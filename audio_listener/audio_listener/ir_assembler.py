"""
audio_listener.ir_assembler — Assemble MusicalAnalysis from all analysis results.

Pitch encoding in voice_sequences
----------------------------------
Each pitched note is stored as::

    {"degree": int, "alter": int, "octave": int, "duration": float, "velocity": int}

where:
  degree  — diatonic scale degree 1-7, or 0 for chromatic notes that do not
            sit within ±1 semitone of any scale degree
  alter   — chromatic offset from the diatonic position of that degree
            (0 = pure diatonic; -1 = flat; +1 = sharp)
            For degree=0, alter holds the raw semitones above the tonic (0-11).
  octave  — absolute MIDI octave (MIDI 60 = C4 → octave 4)

Rests are stored as::

    {"degree": "rest", "alter": 0, "octave": 0, "duration": float, "velocity": 0}

Drum hits use a separate schema::

    {"pitch": str, "duration": float, "velocity": int}   (drum name like "kick")

Why this encoding is transform-safe
-------------------------------------
- transpose(+n): changes analysis.key → tonic_pc shifts → every decoded pitch
  shifts by exactly n semitones automatically. No modification to stored events.
- mode_swap("minor"): changes analysis.mode → at decode time SCALE_INTERVALS[mode]
  is used to look up degree offsets. Degree 3 in major decodes to a major third
  (4 st), but the same degree 3 with the same alter in minor decodes to a minor
  third (3 st). Parallel-mode melody substitution is automatic.
- modulate_section(): changes the key for one section. voice_sequences are cleared
  for that section in transforms.modulate_section() so the generator takes over.
"""

from __future__ import annotations

from audio_listener.midi_parser      import MidiData
from audio_listener.channel_router   import ChannelAssignment
from audio_listener.pitch_tracker    import TrackedNote, extract_notes, quantise_beats
from audio_listener.phrase_segmenter import PhraseSegment, SectionBoundary
from audio_listener.chord_builder    import build_harmonic_plan, detect_harmonic_rhythm
from audio_listener.texture_profiler import profile_section


# ── Drum name lookup ───────────────────────────────────────────────────────────

_DRUM_NAMES: dict[int, str] = {
    35: "kick",    36: "kick",
    37: "snare_rim", 38: "snare", 39: "snare", 40: "snare_electric",
    41: "tom_lo",  42: "hihat_closed", 43: "tom_floor",
    44: "hihat_pedal", 45: "tom_lo_mid", 46: "hihat_open",
    47: "tom_lo_mid",  48: "tom_hi_mid", 49: "crash",
    50: "tom_hi",  51: "ride",   52: "crash2",  53: "ride_bell",
    54: "tambourine",  56: "cowbell", 57: "crash2", 59: "ride",
}


# ── Pitch encoding helpers ─────────────────────────────────────────────────────

def _midi_to_degree_event(
    pitch: int, tonic_pc: int, scale_ivs: list[int]
) -> dict:
    """
    Convert a MIDI pitch to a degree-encoded note event dict.

    Parameters
    ----------
    pitch      MIDI note number (0-127)
    tonic_pc   Pitch class of the tonic (0=C, 1=C#, …, 11=B)
    scale_ivs  SCALE_INTERVALS[mode] — list of 7 diatonic semitone offsets

    Returns a dict with keys: degree, alter, octave.
    """
    relative_pc = (pitch % 12 - tonic_pc) % 12   # 0-11 above tonic
    octave      = pitch // 12 - 1                 # standard MIDI octave

    best_degree = 0
    best_alter  = relative_pc      # fallback: chromatic escape (degree=0)
    best_abs    = relative_pc      # abs(alter) for the fallback

    for d_idx, iv in enumerate(scale_ivs):
        alter = relative_pc - iv
        # Prefer alter in range -2..+2 and pick smallest abs value
        if abs(alter) < best_abs:
            best_abs    = abs(alter)
            best_degree = d_idx + 1   # 1-based degree
            best_alter  = alter
        elif abs(alter) == best_abs and alter < best_alter:
            # Tie-break: prefer flat over sharp
            best_degree = d_idx + 1
            best_alter  = alter

    return {"degree": best_degree, "alter": best_alter, "octave": octave}


def _split_into_layers(
    notes: list[TrackedNote],
    epsilon: float = 0.02,
    max_layers: int = 4,
    lowest_first: bool = False,
) -> list[list[TrackedNote]]:
    """
    Split a polyphonic list of TrackedNotes into monophonic layers.

    Notes that start within *epsilon* beats of each other are grouped as a chord.
    By default (lowest_first=False) layer 0 gets the highest pitch per chord so
    that the primary layer carries the melodic top voice (correct for soprano/alto).
    Pass lowest_first=True for bass channels so layer 0 carries the root/bass tone.

    Only the primary layer is tagged with the canonical voice name (soprano, alto,
    bass); extra layers get suffix names (alto_layer_2, bass_layer_2, …) and are
    captured as additional inner voices so replay_section can play them all.
    """
    if not notes:
        return []
    sorted_notes = sorted(notes, key=lambda n: (n.start_beat, -n.pitch))

    # Group into chord events (notes within epsilon beats share a start)
    chords: list[list[TrackedNote]] = []
    current: list[TrackedNote] = []
    for note in sorted_notes:
        if not current or note.start_beat - current[0].start_beat < epsilon:
            current.append(note)
        else:
            chords.append(current)
            current = [note]
    if current:
        chords.append(current)

    n_layers = min(max_layers, max(len(c) for c in chords))
    layers: list[list[TrackedNote]] = [[] for _ in range(n_layers)]
    for chord in chords:
        chord_sorted = sorted(chord, key=lambda n: n.pitch if lowest_first else -n.pitch)
        for i, note in enumerate(chord_sorted[:n_layers]):
            layers[i].append(note)

    return [layer for layer in layers if layer]


def _extract_pitched_sequence(
    notes: list[TrackedNote],
    start:     float,
    end:       float,
    tonic_pc:  int,
    scale_ivs: list[int],
) -> list[dict]:
    """
    Convert TrackedNotes for one voice into a list of degree-encoded NoteEvent dicts.

    Gaps between notes are filled with rest events so the sequence can be
    rendered to a track without timing drift.
    """
    in_sec = sorted(
        [n for n in notes if start <= n.start_beat < end],
        key=lambda n: n.start_beat,
    )
    result: list[dict] = []
    cursor = start

    for n in in_sec:
        # Use max(cursor, n.start_beat) so cursor never goes backward.
        # If the previous note's quantized duration overlaps this note's start
        # (common with NES staccato that rounds up to a full grid cell), we
        # treat the note as starting immediately after the previous one ends.
        effective_start = max(cursor, n.start_beat)
        gap = effective_start - cursor
        if gap > 0.02:
            result.append({
                "degree": "rest", "alter": 0, "octave": 0,
                "duration": round(gap, 4), "velocity": 0,
            })
        # Clip duration to section end; skip notes that start at/past the end.
        remaining = end - effective_start
        if remaining <= 0.02:
            break
        dur = min(n.duration_beats, remaining)
        dur = max(dur, 0.05)
        # Don't let minimum-duration padding push cursor past section end.
        dur = min(dur, remaining)
        ev  = _midi_to_degree_event(n.pitch, tonic_pc, scale_ivs)
        ev["duration"] = round(dur, 4)
        ev["velocity"] = n.velocity
        result.append(ev)
        cursor = effective_start + dur

    if end - cursor > 0.02:
        result.append({
            "degree": "rest", "alter": 0, "octave": 0,
            "duration": round(end - cursor, 4), "velocity": 0,
        })
    return result


def _extract_drum_sequence(
    notes: list[TrackedNote], start: float, end: float
) -> list[dict]:
    """
    Convert drum TrackedNotes → rest-encoded NoteEvent dicts.

    Drum hits use the {"pitch": name, "duration": gap_to_next, "velocity": v}
    schema (not degree-encoded, since drum pitches are not tonal).
    """
    in_sec = sorted(
        [n for n in notes if start <= n.start_beat < end],
        key=lambda n: n.start_beat,
    )
    result: list[dict] = []
    cursor = start

    for i, n in enumerate(in_sec):
        gap = n.start_beat - cursor
        if gap > 0.001:
            result.append({
                "degree": "rest", "alter": 0, "octave": 0,
                "duration": round(gap, 4), "velocity": 0,
            })
        next_start = in_sec[i + 1].start_beat if i + 1 < len(in_sec) else end
        dur = max(next_start - n.start_beat, 0.001)
        drum_name = _DRUM_NAMES.get(n.pitch, str(n.pitch))
        result.append({
            "pitch":    drum_name,
            "duration": round(dur, 4),
            "velocity": n.velocity,
        })
        cursor = next_start

    if end - cursor > 0.01:
        result.append({
            "degree": "rest", "alter": 0, "octave": 0,
            "duration": round(end - cursor, 4), "velocity": 0,
        })
    return result


def _get_clean_channel_notes(
    data: MidiData, ch: int | None, quantise: bool = True
) -> list[TrackedNote]:
    """
    Extract and optionally quantise notes for a single MIDI channel,
    bypassing any supplementation or mixing done in the main pipeline.

    Used so voice_sequences contain only the notes that actually appeared
    on each channel, not a blend of multiple channels.
    """
    if ch is None:
        return []
    raw   = data.notes_for_channel(ch)
    notes = extract_notes(raw, data.ticks_per_beat, data.tempo_changes)
    return quantise_beats(notes) if quantise else notes


# ── Top-level assembler ────────────────────────────────────────────────────────

def assemble(
    data:           MidiData,
    assignment:     ChannelAssignment,
    melody:         list[TrackedNote],
    bass:           list[TrackedNote],
    counter:        list[TrackedNote],
    drums:          list[TrackedNote] | None,
    tempo_bpm:      int,
    time_sig:       list[int],
    key:            str,
    mode:           str,
    phrases:        list[PhraseSegment],
    sections:       list[SectionBoundary],
    section_labels: list[tuple[str, str]],
    motifs:         list,
) -> object:
    """
    Build and return a MusicalAnalysis from all pre-computed results.

    The passed melody/bass/counter/drums lists (which may include supplemental
    notes from other channels) are used for structural analysis — phrase
    segmentation, harmonic plan, texture profiling.

    For voice_sequences, raw per-channel notes (one channel per voice) are
    extracted from MidiData directly, so the stored sequences are clean.
    """
    from music_generator.ir import MusicalAnalysis, SectionSpec
    from music_generator    import ROOT_TO_PC, SCALE_INTERVALS

    beats_per_bar = time_sig[0]
    tonic_pc      = ROOT_TO_PC.get(key, 0)
    scale_ivs     = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])
    harm_rhythm   = detect_harmonic_rhythm(melody, beats_per_bar, bass=bass, counter=counter)

    # Raw per-channel notes for voice_sequences (no supplementation)
    clean_sop   = _get_clean_channel_notes(data, assignment.melody)
    clean_alto  = _get_clean_channel_notes(data, assignment.countermelody)
    clean_bass  = _get_clean_channel_notes(data, assignment.bass)
    clean_drums = (_get_clean_channel_notes(data, assignment.drums)
                   if assignment.drums is not None else [])

    # Inner voices: sort by mean pitch descending (highest-register first)
    def _ch_mean_pitch(ch: int) -> float:
        raw = data.notes_for_channel(ch)
        return sum(n.pitch for n in raw) / len(raw) if raw else 0.0

    inner_channels = sorted(
        assignment.inner or [],
        key=_ch_mean_pitch,
        reverse=True,   # highest pitch first → inner_1 is the highest inner voice
    )
    clean_inner = [_get_clean_channel_notes(data, ch) for ch in inner_channels]

    # ── Phrase map helper ──────────────────────────────────────────────────────
    def _phrase_map_for_section(sec: SectionBoundary) -> list[tuple[float, float, int]]:
        pm = []
        for local_idx, ph_idx in enumerate(range(sec.start_phrase, sec.end_phrase)):
            if ph_idx < len(phrases):
                ph = phrases[ph_idx]
                pm.append((ph.start_beat, ph.end_beat, local_idx))
        return pm

    # ── Form hint ─────────────────────────────────────────────────────────────
    def _form_hint() -> str:
        n = len(sections)
        if n <= 2:
            return "binary"
        roles = [r for r, _ in section_labels]
        if roles.count("chorus") >= 2 and "verse" in roles:
            return "rondo"
        if n == 3 and roles[0] == roles[2]:
            return "ternary"
        return "through_composed"

    # ── Build sections ─────────────────────────────────────────────────────────
    section_specs = []
    for i, (sec, (role, label)) in enumerate(zip(sections, section_labels)):
        pm        = _phrase_map_for_section(sec)
        harm_plan = build_harmonic_plan(
            melody=melody, bass=bass, counter=counter,
            tonic_pc=tonic_pc, mode=mode,
            harm_rhythm=harm_rhythm,
            section_start_beat=sec.start_beat,
            section_end_beat=sec.end_beat,
            phrase_map=pm,
        )
        texture, energy = profile_section(
            section=sec, melody=melody, bass=bass, counter=counter,
            drum_notes=drums, beats_per_bar=beats_per_bar,
            all_sections=sections,
        )

        # Motif assignment
        motif_id = None
        if motifs:
            if role in ("chorus", "verse", "bridge", "generic"):
                motif_id = motifs[0].id
            elif len(motifs) > 1:
                motif_id = motifs[1].id

        # repeat_of
        repeat_of_id = None
        if sec.is_repeat_of is not None and sec.is_repeat_of < len(section_labels):
            orig_role, _ = section_labels[sec.is_repeat_of]
            repeat_of_id = f"{orig_role}_{sec.is_repeat_of}"

        n_phrases   = sec.end_phrase - sec.start_phrase
        phrase_bars = phrases[sec.start_phrase].bar_count if sec.start_phrase < len(phrases) else 4

        # ── Voice sequences (clean per-channel, degree-encoded) ────────────────
        s_beat, e_beat = sec.start_beat, sec.end_beat
        voice_seqs: dict = {}

        if clean_sop:
            seq = _extract_pitched_sequence(clean_sop, s_beat, e_beat, tonic_pc, scale_ivs)
            if seq:
                voice_seqs["soprano"] = seq

        # For polyphonic channels (e.g. piano chords), split into monophonic layers so
        # every chord tone is captured.  Layer 0 keeps the canonical voice name; extra
        # layers get a suffix (_layer_2, _layer_3, …) and are replayed as additional
        # tracks.  Monophonic channels produce a single layer with no suffix.
        if clean_alto:
            alto_layers = _split_into_layers(clean_alto, lowest_first=False)
            for li, layer_notes in enumerate(alto_layers):
                seq = _extract_pitched_sequence(layer_notes, s_beat, e_beat, tonic_pc, scale_ivs)
                if seq:
                    vname = "alto" if li == 0 else f"alto_layer_{li + 1}"
                    voice_seqs[vname] = seq

        if clean_bass:
            # lowest_first=True: primary bass layer carries the lowest pitch (root/bass tone)
            bass_layers = _split_into_layers(clean_bass, lowest_first=True)
            for li, layer_notes in enumerate(bass_layers):
                seq = _extract_pitched_sequence(layer_notes, s_beat, e_beat, tonic_pc, scale_ivs)
                if seq:
                    vname = "bass" if li == 0 else f"bass_layer_{li + 1}"
                    voice_seqs[vname] = seq

        if clean_drums:
            seq = _extract_drum_sequence(clean_drums, s_beat, e_beat)
            if seq:
                voice_seqs["drums"] = seq

        # Capture all inner channels, sorted by mean pitch descending (highest first)
        for inner_idx, inner_notes in enumerate(clean_inner):
            if not inner_notes:
                continue
            inner_layers = _split_into_layers(inner_notes, lowest_first=False)
            for li, layer_notes in enumerate(inner_layers):
                seq = _extract_pitched_sequence(layer_notes, s_beat, e_beat, tonic_pc, scale_ivs)
                if seq:
                    vname = (f"inner_{inner_idx + 1}" if li == 0
                             else f"inner_{inner_idx + 1}_layer_{li + 1}")
                    voice_seqs[vname] = seq

        # Build source channel map: voice name → MIDI channel number.
        # Extra polyphonic layers share the same MIDI channel as the primary voice
        # so replay_section can look up their instrument program correctly.
        source_ch_map: dict[str, int] = {}
        if assignment.melody is not None:
            source_ch_map["soprano"] = assignment.melody
        if assignment.countermelody is not None:
            source_ch_map["alto"] = assignment.countermelody
            for vname in voice_seqs:
                if vname.startswith("alto_layer_"):
                    source_ch_map[vname] = assignment.countermelody
        if assignment.bass is not None:
            source_ch_map["bass"] = assignment.bass
            for vname in voice_seqs:
                if vname.startswith("bass_layer_"):
                    source_ch_map[vname] = assignment.bass
        if assignment.drums is not None:
            source_ch_map["drums"] = assignment.drums
        for inner_idx, ch in enumerate(inner_channels):
            base = f"inner_{inner_idx + 1}"
            source_ch_map[base] = ch
            for vname in voice_seqs:
                if vname.startswith(f"{base}_layer_"):
                    source_ch_map[vname] = ch

        # Build source program map: voice name → GM program number (0-indexed)
        # Used by replay_section to restore the original MIDI instrument colour.
        progs = getattr(data, "program_numbers", {})
        source_prog_map: dict[str, int] = {}
        for vname, ch in source_ch_map.items():
            if ch in progs:
                source_prog_map[vname] = progs[ch]

        section_specs.append(SectionSpec(
            id=f"{role}_{i}",
            label=label,
            role=role,
            harmonic_plan=harm_plan,
            motif_id=motif_id,
            texture=texture,
            energy=energy,
            repeat_of=repeat_of_id,
            extra_params={
                "num_phrases":    max(2, n_phrases),
                "phrase_length":  max(2, phrase_bars),
                "_source_channels": source_ch_map,   # prefixed _ → skipped by generate_section
                "_source_programs": source_prog_map,  # prefixed _ → skipped by generate_section
            },
            voice_sequences=voice_seqs,
        ))

    motif_dict = {m.id: m for m in motifs}

    return MusicalAnalysis(
        key=key,
        mode=mode,
        tempo_bpm=tempo_bpm,
        time_signature=time_sig,
        sections=section_specs,
        motifs=motif_dict,
        form_hint=_form_hint(),
    )
