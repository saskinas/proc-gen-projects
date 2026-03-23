"""
audio_listener — Convert MIDI (and future audio) files to MusicalAnalysis symbolic IR.

Primary API
-----------
    from audio_listener import listen_midi

    analysis = listen_midi("zelda_overworld.mid")
    # Returns MusicalAnalysis ready for music_generator.compose.generate_from_analysis()

Future
------
    listen_audio("song.mp3")   # raises NotImplementedError until implemented
"""

from audio_listener._paths import ensure_paths
ensure_paths()

import pathlib
from typing import Union


def listen_midi(
    source: Union[bytes, str, "pathlib.Path"],
    *,
    quantise: bool = True,
    min_motif_occurrences: int = 3,
) -> object:
    """
    Analyse a MIDI file and return a MusicalAnalysis IR.

    Parameters
    ----------
    source
        Raw MIDI bytes, a file path string, or a pathlib.Path.
    quantise
        Snap note timings to a 1/32-note grid. Recommended for NES MIDI.
    min_motif_occurrences
        Minimum phrase count a motif must appear in to be registered.

    Returns
    -------
    MusicalAnalysis
        Fully populated IR. Pass directly to generate_from_analysis().

    Example
    -------
        from audio_listener import listen_midi
        from music_generator.compose import generate_from_analysis
        from procgen import export

        analysis  = listen_midi("game_music.mid")
        score     = generate_from_analysis(analysis, base_params=NES_BASE, seed=0)
        midi_out  = export.to_midi(score)
        open("reproduction.mid", "wb").write(midi_out)
    """
    from audio_listener.midi_parser      import parse_midi
    from audio_listener.channel_router   import assign_channels
    from audio_listener.pitch_tracker    import extract_notes, quantise_beats
    from audio_listener.tempo_meter      import detect_tempo, detect_time_signature
    from audio_listener.key_detector     import detect_key_mode
    from audio_listener.phrase_segmenter import segment_phrases, detect_sections
    from audio_listener.section_labeler  import label_sections
    from audio_listener.motif_finder     import find_motifs
    from audio_listener.ir_assembler     import assemble

    # Load source
    if isinstance(source, (str, pathlib.Path)):
        data_bytes = pathlib.Path(source).read_bytes()
    else:
        data_bytes = bytes(source)

    # ── 1. Parse raw MIDI ────────────────────────────────────────────────────
    raw = parse_midi(data_bytes)

    # ── 2. Global meter ──────────────────────────────────────────────────────
    tempo_bpm = detect_tempo(raw)
    time_sig  = detect_time_signature(raw)
    beats_per_bar = time_sig[0]

    # ── 3. Channel role assignment ───────────────────────────────────────────
    assignment = assign_channels(raw)

    # ── 4. Extract and quantise notes per channel ────────────────────────────
    def _get_notes(ch):
        if ch is None:
            return []
        raw_ch = raw.notes_for_channel(ch)
        notes  = extract_notes(raw_ch, raw.ticks_per_beat, raw.tempo_changes)
        return quantise_beats(notes) if quantise else notes

    melody  = _get_notes(assignment.melody)
    bass    = _get_notes(assignment.bass)
    counter = _get_notes(assignment.countermelody)
    drums   = _get_notes(assignment.drums) if assignment.drums is not None else None

    if not melody:
        # If no melody detected, pick the non-drum channel with the most notes
        non_drum = raw.channels() - ({assignment.drums} if assignment.drums is not None else set())
        if non_drum:
            ch     = max(non_drum, key=lambda c: len(raw.notes_for_channel(c)))
            melody = _get_notes(ch)

    # If melody doesn't start near beat 0, supplement with the highest-count
    # non-drum channel that covers the beginning (for phrase detection purposes).
    if melody:
        total_beats_piece = raw.total_ticks / raw.ticks_per_beat
        melody_start = min(n.start_beat for n in melody)
        if melody_start > total_beats_piece * 0.10:
            assigned = {assignment.melody, assignment.bass,
                        assignment.countermelody, assignment.drums}
            for ch in sorted(raw.channels(), key=lambda c: -len(raw.notes_for_channel(c))):
                if ch in assigned:
                    continue
                supp = _get_notes(ch)
                if supp and min(n.start_beat for n in supp) < melody_start * 0.5:
                    melody = sorted(melody + supp, key=lambda n: n.start_beat)
                    break

    # ── 5. Key / mode ────────────────────────────────────────────────────────
    # Build a supplemental note pool from non-drum ignored channels so that
    # key detection is not misled by a sparse melody assignment.
    def _all_non_drum_notes() -> list:
        all_notes: list = []
        drum_ch = assignment.drums
        for ch in raw.channels():
            if ch == drum_ch:
                continue
            all_notes.extend(_get_notes(ch))
        return all_notes

    # Use all non-drum notes for key detection.
    # Bass notes are passed separately so detect_key_mode can apply 1.5× weight
    # to the bass channel — important since bass roots are the strongest tonal signal.
    _bass_key_notes = _get_notes(assignment.bass) if assignment.bass is not None else []
    _all_non_drum   = _all_non_drum_notes()
    key, mode = detect_key_mode(_all_non_drum, _bass_key_notes, [])

    # ── 6. Phrase and section segmentation ───────────────────────────────────
    phrases  = segment_phrases(melody, beats_per_bar)
    sections = detect_sections(phrases)

    # ── 7. Section labels ────────────────────────────────────────────────────
    total_beats    = raw.total_ticks / raw.ticks_per_beat
    section_labels = label_sections(sections, phrases, melody, total_beats)

    # ── 8. Motifs ────────────────────────────────────────────────────────────
    # Run motif detection on the top two highest-pitched channels (≥25 notes).
    # Combining multiple voices catches motifs that aren't in the soprano alone
    # (e.g. Kirby inner voices that carry the main theme in some sections).
    def _melodic_channels_for_motifs(n_top: int = 2) -> list:
        drum_ch = assignment.drums
        candidates = [
            (ch, raw.notes_for_channel(ch))
            for ch in raw.channels()
            if ch != drum_ch and len(raw.notes_for_channel(ch)) >= 25
        ]
        if not candidates:
            return [melody]
        candidates.sort(key=lambda x: -(sum(n.pitch for n in x[1]) / len(x[1])))
        return [_get_notes(ch) for ch, _ in candidates[:n_top]]

    all_motif_voices = _melodic_channels_for_motifs()
    seen_patterns: set = set()
    motifs: list = []
    for voice_notes in all_motif_voices:
        for m in find_motifs(voice_notes, phrases, min_motif_occurrences):
            pattern_key = tuple(m.intervals)
            if pattern_key not in seen_patterns:
                seen_patterns.add(pattern_key)
                motifs.append(m)
    # Renumber ids to avoid collisions between voices
    _next_motif  = [0]
    _next_seq    = [0]
    _next_cad    = [0]
    for m in motifs:
        if m.type == "motif":
            m.id = f"motif_{_next_motif[0]}"; _next_motif[0] += 1
        elif m.type == "sequence":
            m.id = f"seq_{_next_seq[0]}";    _next_seq[0]   += 1
        else:
            m.id = f"cadence_{_next_cad[0]}"; _next_cad[0]  += 1

    # ── 9. Assemble ──────────────────────────────────────────────────────────
    return assemble(
        data=raw,
        assignment=assignment,
        melody=melody, bass=bass, counter=counter, drums=drums,
        tempo_bpm=tempo_bpm, time_sig=time_sig,
        key=key, mode=mode,
        phrases=phrases, sections=sections, section_labels=section_labels,
        motifs=motifs,
    )


def listen_audio(
    source: Union[bytes, str, "pathlib.Path"],
    **kwargs,
) -> object:
    """
    Stub for future audio (WAV/MP3/OGG) analysis.

    Raises NotImplementedError. Planned implementation requires:
      - Pitch tracking (pYIN or CREPE for monophonic; basic-pitch for polyphonic)
      - Onset detection (librosa)
      - Beat tracking (librosa.beat.beat_track)
    then feeds into the same key_detector / chord_builder / phrase_segmenter pipeline.
    """
    raise NotImplementedError(
        "listen_audio() is not yet implemented. "
        "For MIDI files use listen_midi() instead.\n"
        "Audio support will require: librosa, numpy, and optionally basic-pitch."
    )
