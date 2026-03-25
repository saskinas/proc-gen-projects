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
from audio_listener.channel_router   import ChannelAssignment, SectionChannelAssignment
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
        gap = n.start_beat - cursor
        if gap > 0.02:
            # Insert rest to fill the gap before this note.
            result.append({
                "degree": "rest", "alter": 0, "octave": 0,
                "duration": round(gap, 4), "velocity": 0,
            })
            cursor += gap
        elif gap < -0.02 and result and result[-1]["degree"] != "rest":
            # Previous note's duration overlaps this note's start.
            # Clip the previous note so timing stays anchored to the original
            # beat positions instead of accumulating drift.
            clip = cursor - n.start_beat
            prev_dur = result[-1]["duration"]
            new_dur  = max(0.05, round(prev_dur - clip, 4))
            result[-1]["duration"] = new_dur
            cursor -= (prev_dur - new_dur)
            # Re-check: there may now be a small gap
            gap2 = n.start_beat - cursor
            if gap2 > 0.02:
                result.append({
                    "degree": "rest", "alter": 0, "octave": 0,
                    "duration": round(gap2, 4), "velocity": 0,
                })
                cursor += gap2
        # Clip duration to section end; skip notes that start at/past the end.
        remaining = end - n.start_beat
        if remaining <= 0.02:
            break
        dur = min(n.duration_beats, remaining)
        dur = max(dur, 0.05)
        dur = min(dur, remaining)
        ev  = _midi_to_degree_event(n.pitch, tonic_pc, scale_ivs)
        ev["duration"] = round(dur, 4)
        ev["velocity"] = n.velocity
        result.append(ev)
        cursor = n.start_beat + dur

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
        dur = max(next_start - n.start_beat, 0.0)
        drum_name = _DRUM_NAMES.get(n.pitch, str(n.pitch))
        result.append({
            "pitch":    drum_name,
            "duration": round(dur, 4),
            "velocity": n.velocity,
        })
        # Anchor cursor to actual beat position to prevent drift
        cursor = n.start_beat + dur

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


def _pitch_similarity(notes_a: list, notes_b: list) -> float:
    """Cosine similarity of per-MIDI-pitch count histograms."""
    from collections import Counter
    hA = Counter(n.pitch for n in notes_a)
    hB = Counter(n.pitch for n in notes_b)
    all_p = set(hA) | set(hB)
    if not all_p:
        return 0.0
    a = [hA.get(p, 0) for p in all_p]
    b = [hB.get(p, 0) for p in all_p]
    dot  = sum(x * y for x, y in zip(a, b))
    na   = sum(x * x for x in a) ** 0.5
    nb   = sum(x * x for x in b) ** 0.5
    return dot / (na * nb + 1e-9)


def _mean_velocity(notes: list) -> float:
    if not notes:
        return 0.0
    return sum(n.velocity for n in notes) / len(notes)


def _fill_melody_gaps(
    primary:   list[TrackedNote],
    alt_pools: list[list[TrackedNote]],
    start:     float,
    end:       float,
    min_gap:   float = 2.0,
) -> list[TrackedNote]:
    """
    Fill silent gaps in the primary melody with notes from alternative voice pools.

    For each gap in *primary* that is >= *min_gap* beats long, the alternative
    pool with the most notes covering that window is merged in.  This lets the
    soprano track follow the melody even when it shifts to a different MIDI
    channel (e.g. rhythm channel carries the main theme for one section).

    Parameters
    ----------
    primary    Primary soprano TrackedNote list.
    alt_pools  Other pitched channels to borrow from during gaps.
    start/end  Section beat boundaries (only notes within [start, end) are used).
    min_gap    Minimum silence duration in beats before gap-filling is attempted.
    """
    if not primary or not alt_pools:
        return primary

    sorted_primary = sorted(primary, key=lambda n: n.start_beat)

    # Identify silent gaps in the primary melody
    gaps: list[tuple[float, float]] = []
    cursor = start
    for n in sorted_primary:
        if n.start_beat > end:
            break
        gap_end   = n.start_beat
        gap_start = cursor
        if gap_end - gap_start >= min_gap:
            gaps.append((gap_start, gap_end))
        cursor = max(cursor, n.start_beat + n.duration_beats)
    # Tail gap
    if end - cursor >= min_gap:
        gaps.append((cursor, end))

    if not gaps:
        return primary

    # For each gap, pick the alt_pool with the most notes in that window
    extra: list[TrackedNote] = []
    for gap_start, gap_end in gaps:
        best_pool:  list[TrackedNote] | None = None
        best_count: int = 0
        for pool in alt_pools:
            count = sum(1 for n in pool if gap_start <= n.start_beat < gap_end)
            if count > best_count:
                best_count = count
                best_pool  = pool
        if best_pool and best_count > 0:
            extra.extend(n for n in best_pool if gap_start <= n.start_beat < gap_end)

    if not extra:
        return primary

    return sorted(primary + extra, key=lambda n: n.start_beat)


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
    section_assignments: list[SectionChannelAssignment] | None = None,
    tempo_map: list[tuple[float, int]] | None = None,
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

    # Raw per-channel notes for voice_sequences (no supplementation).
    # When per-section assignments are available, extraction is deferred to the
    # section loop so each section can use its own channel mapping.  We cache
    # per-channel results to avoid redundant extraction.
    _clean_cache: dict[int | None, list[TrackedNote]] = {}

    def _cached_clean(ch: int | None) -> list[TrackedNote]:
        if ch not in _clean_cache:
            _clean_cache[ch] = _get_clean_channel_notes(data, ch)
        return _clean_cache[ch]

    # Pre-extract global notes (used when section_assignments is None)
    clean_sop   = _cached_clean(assignment.melody)
    clean_alto  = _cached_clean(assignment.countermelody)
    clean_bass  = _cached_clean(assignment.bass)
    clean_drums = (_cached_clean(assignment.drums)
                   if assignment.drums is not None else [])

    # Inner voices: sort by mean pitch descending (highest-register first)
    def _ch_mean_pitch(ch: int) -> float:
        raw = data.notes_for_channel(ch)
        return sum(n.pitch for n in raw) / len(raw) if raw else 0.0

    # Detect and deduplicate stereo-layer channels: identical pitch/timing content
    # on multiple channels (common in game music for stereo effects).  Keeps the
    # first occurrence and drops exact duplicates so they don't waste voice slots.
    def _channel_fingerprint(ch: int) -> tuple:
        """Quick fingerprint: (note_count, sorted first-10 pitch+tick pairs)."""
        raw_notes = data.notes_for_channel(ch)
        if not raw_notes:
            return ()
        sorted_n = sorted(raw_notes, key=lambda n: (n.tick_on, n.pitch))
        return (len(raw_notes),
                tuple((n.pitch, n.tick_on) for n in sorted_n[:10]))

    # Build set of fingerprints for melody/counter/bass (already-assigned channels)
    _assigned_fps: set[tuple] = set()
    for _ach in (assignment.melody, assignment.countermelody, assignment.bass):
        if _ach is not None:
            _assigned_fps.add(_channel_fingerprint(_ach))

    # Precompute melody raw notes + velocity for echo detection
    _melody_raw = data.notes_for_channel(assignment.melody) if assignment.melody is not None else []
    _melody_vel  = _mean_velocity(_melody_raw)

    # Filter inner channels: skip exact duplicates of already-assigned channels
    # or of earlier inner channels.
    _seen_fps: set[tuple] = set(_assigned_fps)
    _deduped_inner: list[int] = []
    for ch in (assignment.inner or []):
        fp = _channel_fingerprint(ch)
        if fp and fp in _seen_fps:
            continue  # stereo duplicate — skip
        # Echo channel suppression: skip channels whose pitch content is nearly
        # identical to the melody channel but at significantly lower velocity.
        # These are SNES echo-hardware copies that create "ghost note" artifacts.
        if _melody_raw:
            ch_raw = data.notes_for_channel(ch)
            if ch_raw:
                sim = _pitch_similarity(ch_raw, _melody_raw)
                ch_vel = _mean_velocity(ch_raw)
                if sim > 0.90 and _melody_vel > 0 and ch_vel < _melody_vel * 0.70:
                    continue  # echo/reverb copy — skip
        _seen_fps.add(fp)
        _deduped_inner.append(ch)

    global_inner_channels = sorted(
        _deduped_inner,
        key=_ch_mean_pitch,
        reverse=True,   # highest pitch first → inner_1 is the highest inner voice
    )
    clean_inner = [_cached_clean(ch) for ch in global_inner_channels]

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

        # Map motif occurrences to this section's phrase range
        motif_occs = []
        if motif_id:
            motif_def = next((m for m in motifs if m.id == motif_id), None)
            if motif_def:
                for occ in motif_def.occurrences:
                    ph = occ.get("phrase_idx", -1)
                    if sec.start_phrase <= ph < sec.end_phrase:
                        motif_occs.append({
                            "phrase_local_idx": ph - sec.start_phrase,
                            "transform":       occ.get("transform", "original"),
                            "transposition":   occ.get("transposition", 0),
                            "duration_factor": occ.get("duration_factor", 1.0),
                        })

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

        # Determine which channels provide each role for this section.
        #
        # Soprano/alto/bass are ALWAYS pinned to the global assignment so their
        # channel identity is stable across sections.  If the melody shifts to a
        # different channel in some section, that channel is captured as an inner
        # voice (inner_chN) and the gap-filler borrows its notes into soprano.
        #
        # Per-section routing only affects which non-primary channels appear as
        # inner voices for this section.
        sec_sop   = clean_sop
        sec_alto  = clean_alto
        sec_bass  = clean_bass
        sec_drums = clean_drums
        sec_melody_ch   = assignment.melody
        sec_counter_ch  = assignment.countermelody
        sec_bass_ch     = assignment.bass
        sec_drums_ch    = assignment.drums

        # Build inner channel list: all non-drum, non-primary channels active
        # in this section.  Use per-section assignment for inner ordering when
        # available, otherwise fall back to global.
        if section_assignments is not None and i < len(section_assignments):
            sec_asgn = section_assignments[i]
            # Collect all channels that appear in the per-section assignment
            # but are NOT the global primary voices → these become inner voices.
            _primary = {assignment.melody, assignment.countermelody,
                        assignment.bass, assignment.drums}
            _sec_all = set()
            for ch in [sec_asgn.melody, sec_asgn.countermelody, sec_asgn.bass]:
                if ch is not None:
                    _sec_all.add(ch)
            _sec_all.update(sec_asgn.inner or [])
            sec_inner_channels = sorted(
                [ch for ch in _sec_all if ch not in _primary],
                key=_ch_mean_pitch,
                reverse=True,
            )
            sec_inner = [_cached_clean(ch) for ch in sec_inner_channels]
        else:
            sec_inner_channels = global_inner_channels
            sec_inner = clean_inner

        if sec_sop:
            # Do NOT gap-fill soprano here — it would move notes from their
            # original channel onto the soprano channel, breaking reproduction.
            # Melody shifts to other channels are captured as inner_chN voices
            # with channel-based naming, preserving per-channel fidelity.
            sop_layers = _split_into_layers(sec_sop, lowest_first=False, max_layers=4)
            for li, layer_notes in enumerate(sop_layers):
                seq = _extract_pitched_sequence(layer_notes, s_beat, e_beat, tonic_pc, scale_ivs)
                if seq:
                    vname = "soprano" if li == 0 else f"soprano_layer_{li + 1}"
                    voice_seqs[vname] = seq

        # For polyphonic channels (e.g. piano chords), split into monophonic layers.
        # Limit to 2 layers to avoid track explosion — layer 0 carries the primary
        # voice, layer 1 captures chord seconds.
        if sec_alto:
            alto_layers = _split_into_layers(sec_alto, lowest_first=False, max_layers=4)
            for li, layer_notes in enumerate(alto_layers):
                seq = _extract_pitched_sequence(layer_notes, s_beat, e_beat, tonic_pc, scale_ivs)
                if seq:
                    vname = "alto" if li == 0 else f"alto_layer_{li + 1}"
                    voice_seqs[vname] = seq

        if sec_bass:
            # lowest_first=True: primary bass layer carries the lowest pitch (root/bass tone)
            bass_layers = _split_into_layers(sec_bass, lowest_first=True, max_layers=4)
            for li, layer_notes in enumerate(bass_layers):
                seq = _extract_pitched_sequence(layer_notes, s_beat, e_beat, tonic_pc, scale_ivs)
                if seq:
                    vname = "bass" if li == 0 else f"bass_layer_{li + 1}"
                    voice_seqs[vname] = seq

        if sec_drums:
            seq = _extract_drum_sequence(sec_drums, s_beat, e_beat)
            if seq:
                voice_seqs["drums"] = seq

        # Capture all inner channels, keyed by MIDI channel number for stability.
        # Using channel-based names (inner_ch2, inner_ch14) instead of rank-based
        # (inner_1, inner_2) ensures the same MIDI channel maps to the same voice
        # across sections, preventing concatenate_scores from mixing channels.
        for inner_idx, ch in enumerate(sec_inner_channels):
            inner_notes = sec_inner[inner_idx]
            if not inner_notes:
                continue
            # Limit inner voice layers to 2 to avoid track explosion
            inner_layers = _split_into_layers(inner_notes, lowest_first=False, max_layers=4)
            for li, layer_notes in enumerate(inner_layers):
                seq = _extract_pitched_sequence(layer_notes, s_beat, e_beat, tonic_pc, scale_ivs)
                if seq:
                    vname = (f"inner_ch{ch}" if li == 0
                             else f"inner_ch{ch}_layer_{li + 1}")
                    voice_seqs[vname] = seq

        # Build source channel map: voice name → MIDI channel number.
        # Extra polyphonic layers share the same MIDI channel as the primary voice
        # so replay_section can look up their instrument program correctly.
        source_ch_map: dict[str, int] = {}
        if sec_melody_ch is not None:
            source_ch_map["soprano"] = sec_melody_ch
            for vname in voice_seqs:
                if vname.startswith("soprano_layer_"):
                    source_ch_map[vname] = sec_melody_ch
        if sec_counter_ch is not None:
            source_ch_map["alto"] = sec_counter_ch
            for vname in voice_seqs:
                if vname.startswith("alto_layer_"):
                    source_ch_map[vname] = sec_counter_ch
        if sec_bass_ch is not None:
            source_ch_map["bass"] = sec_bass_ch
            for vname in voice_seqs:
                if vname.startswith("bass_layer_"):
                    source_ch_map[vname] = sec_bass_ch
        if sec_drums_ch is not None:
            source_ch_map["drums"] = sec_drums_ch
        for ch in sec_inner_channels:
            base = f"inner_ch{ch}"
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

        # ── CC and pitch bend data for this section ────────────────────────────
        # Store per-channel CC and pitch bend events as beat-timed lists so
        # replay_section → export.py can reproduce expression/sustain/modulation.
        source_cc: dict[int, list[dict]] = {}
        source_pb: dict[int, list[dict]] = {}
        if hasattr(data, "cc_events"):
            s_tick_lo = int(s_beat * data.ticks_per_beat)
            e_tick_hi = int(e_beat * data.ticks_per_beat)
            for ev in data.cc_events:
                if s_tick_lo <= ev.tick < e_tick_hi:
                    beat = data.tick_to_beat(ev.tick) - s_beat
                    source_cc.setdefault(ev.channel, []).append({
                        "beat": round(beat, 4),
                        "cc": ev.controller,
                        "value": ev.value,
                    })
        if hasattr(data, "pitch_bends"):
            s_tick_lo = int(s_beat * data.ticks_per_beat)
            e_tick_hi = int(e_beat * data.ticks_per_beat)
            for ev in data.pitch_bends:
                if s_tick_lo <= ev.tick < e_tick_hi:
                    beat = data.tick_to_beat(ev.tick) - s_beat
                    source_pb.setdefault(ev.channel, []).append({
                        "beat": round(beat, 4),
                        "value": ev.value,
                    })

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
                "num_phrases":         max(2, n_phrases),
                "phrase_length":       max(2, phrase_bars),
                "_source_channels":    source_ch_map,    # prefixed _ → skipped by generate_section
                "_source_programs":    source_prog_map,   # prefixed _ → skipped by generate_section
                "_motif_occurrences":  motif_occs,        # prefixed _ → skipped by generate_section
                "_source_cc":          source_cc,         # CC events per channel (beat-relative)
                "_source_pitch_bends": source_pb,         # pitch bend events per channel
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
        tempo_map=tempo_map or [],
    )
