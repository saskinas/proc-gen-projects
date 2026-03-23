"""
audio_listener.ir_assembler — Assemble MusicalAnalysis from all analysis results.
"""

from __future__ import annotations

from audio_listener.midi_parser      import MidiData
from audio_listener.channel_router   import ChannelAssignment
from audio_listener.pitch_tracker    import TrackedNote
from audio_listener.phrase_segmenter import PhraseSegment, SectionBoundary
from audio_listener.chord_builder    import build_harmonic_plan, detect_harmonic_rhythm
from audio_listener.texture_profiler import profile_section

# ── Note-name helpers ──────────────────────────────────────────────────────────

_MIDI_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_DRUM_NAMES: dict[int, str] = {
    35: "kick", 36: "kick",
    37: "snare_rim", 38: "snare", 39: "snare", 40: "snare_electric",
    41: "tom_lo", 42: "hihat_closed", 43: "tom_floor",
    44: "hihat_pedal", 45: "tom_lo_mid", 46: "hihat_open",
    47: "tom_lo_mid", 48: "tom_hi_mid", 49: "crash",
    50: "tom_hi", 51: "ride", 52: "crash2", 53: "ride_bell",
    54: "tambourine", 56: "cowbell", 57: "crash2", 59: "ride",
}


def _midi_to_name(pitch: int) -> str:
    octave = pitch // 12 - 1
    return f"{_MIDI_NAMES[pitch % 12]}{octave}"


def _extract_pitched_sequence(
    notes: list[TrackedNote], start: float, end: float
) -> list[dict]:
    """Convert pitched TrackedNotes → sequential NoteEvent dicts with rest gaps."""
    in_sec = sorted(
        [n for n in notes if start <= n.start_beat < end],
        key=lambda n: n.start_beat,
    )
    result: list[dict] = []
    cursor = start
    for n in in_sec:
        gap = n.start_beat - cursor
        if gap > 0.02:
            result.append({"pitch": "rest", "duration": round(gap, 4), "velocity": 0})
        dur = min(n.duration_beats, end - n.start_beat)
        dur = max(dur, 0.05)
        result.append({
            "pitch":    _midi_to_name(n.pitch),
            "duration": round(dur, 4),
            "velocity": n.velocity,
        })
        cursor = n.start_beat + dur
    if end - cursor > 0.02:
        result.append({"pitch": "rest", "duration": round(end - cursor, 4), "velocity": 0})
    return result


def _extract_drum_sequence(
    notes: list[TrackedNote], start: float, end: float
) -> list[dict]:
    """Convert drum TrackedNotes → rest-encoded NoteEvent dicts."""
    in_sec = sorted(
        [n for n in notes if start <= n.start_beat < end],
        key=lambda n: n.start_beat,
    )
    result: list[dict] = []
    cursor = start
    for i, n in enumerate(in_sec):
        gap = n.start_beat - cursor
        if gap > 0.001:
            result.append({"pitch": "rest", "duration": round(gap, 4), "velocity": 0})
        # Duration = gap to next event (or section end)
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
        result.append({"pitch": "rest", "duration": round(end - cursor, 4), "velocity": 0})
    return result


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
    """
    from music_generator.ir import MusicalAnalysis, SectionSpec
    from music_generator    import ROOT_TO_PC

    beats_per_bar = time_sig[0]
    tonic_pc      = ROOT_TO_PC.get(key, 0)
    harm_rhythm   = detect_harmonic_rhythm(melody, beats_per_bar, bass=bass, counter=counter)

    # Build phrase_map: list[(start_beat, end_beat, phrase_idx_within_section)]
    def _phrase_map_for_section(sec: SectionBoundary) -> list[tuple[float, float, int]]:
        pm = []
        for local_idx, ph_idx in enumerate(range(sec.start_phrase, sec.end_phrase)):
            if ph_idx < len(phrases):
                ph = phrases[ph_idx]
                pm.append((ph.start_beat, ph.end_beat, local_idx))
        return pm

    # Detect form hint from section structure
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

    section_specs = []
    for i, (sec, (role, label)) in enumerate(zip(sections, section_labels)):
        pm         = _phrase_map_for_section(sec)
        harm_plan  = build_harmonic_plan(
            melody=melody,
            bass=bass,
            counter=counter,
            tonic_pc=tonic_pc,
            mode=mode,
            harm_rhythm=harm_rhythm,
            section_start_beat=sec.start_beat,
            section_end_beat=sec.end_beat,
            phrase_map=pm,
        )

        texture, energy = profile_section(
            section=sec,
            melody=melody,
            bass=bass,
            counter=counter,
            drum_notes=drums,
            beats_per_bar=beats_per_bar,
            all_sections=sections,
        )

        # Assign motif: primary motif to all non-intro/outro sections;
        # additional motifs to bridge/verse sections when available.
        motif_id = None
        if motifs:
            if role in ("chorus", "verse", "bridge", "generic"):
                motif_id = motifs[0].id          # primary motif for main sections
            elif len(motifs) > 1:
                motif_id = motifs[1].id          # secondary motif for transitions

        # repeat_of: id of earlier section this repeats
        repeat_of_id = None
        if sec.is_repeat_of is not None and sec.is_repeat_of < len(section_labels):
            orig_role, orig_label = section_labels[sec.is_repeat_of]
            repeat_of_id = f"{orig_role}_{sec.is_repeat_of}"

        n_phrases = sec.end_phrase - sec.start_phrase
        phrase_bars = phrases[sec.start_phrase].bar_count if sec.start_phrase < len(phrases) else 4

        # Capture per-voice note sequences so the generator can replay the original
        s_beat, e_beat = sec.start_beat, sec.end_beat
        voice_seqs: dict = {}
        if melody:
            seq = _extract_pitched_sequence(melody, s_beat, e_beat)
            if seq:
                voice_seqs["soprano"] = seq
        if counter:
            seq = _extract_pitched_sequence(counter, s_beat, e_beat)
            if seq:
                voice_seqs["alto"] = seq
        if bass:
            seq = _extract_pitched_sequence(bass, s_beat, e_beat)
            if seq:
                voice_seqs["bass"] = seq
        if drums is not None:
            seq = _extract_drum_sequence(drums, s_beat, e_beat)
            if seq:
                voice_seqs["drums"] = seq

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
                "num_phrases":  max(2, n_phrases),
                "phrase_length": max(2, phrase_bars),
            },
            voice_sequences=voice_seqs,
        ))

    # Build motifs dict
    motif_dict = {m.id: m for m in motifs}

    analysis = MusicalAnalysis(
        key=key,
        mode=mode,
        tempo_bpm=tempo_bpm,
        time_signature=time_sig,
        sections=section_specs,
        motifs=motif_dict,
        form_hint=_form_hint(),
    )

    return analysis
