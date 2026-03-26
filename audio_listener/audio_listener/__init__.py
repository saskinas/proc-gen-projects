"""
audio_listener — Convert MIDI and audio files to MusicalAnalysis symbolic IR.

Primary API
-----------
    from audio_listener import listen_midi, listen_audio

    analysis = listen_midi("zelda_overworld.mid")
    analysis = listen_audio("song.wav")
    # Returns MusicalAnalysis ready for music_generator.compose.generate_from_analysis()

Audio input requires: basic-pitch, librosa, soundfile, numpy
Stem separation requires: demucs, torch, torchaudio
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
    from audio_listener.channel_router   import assign_channels, assign_channels_per_section
    from audio_listener.pitch_tracker    import extract_notes, quantise_beats
    from audio_listener.tempo_meter      import detect_tempo, detect_tempo_map, detect_time_signature
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
    tempo_map = detect_tempo_map(raw)
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

    # If melody doesn't cover the full piece, supplement with the most active
    # non-assigned channel that extends beyond the melody's range.
    # This ensures the phrase segmenter creates phrases for the entire piece,
    # so per-section routing can capture all channels everywhere.
    if melody:
        total_beats_piece = raw.total_ticks / raw.ticks_per_beat
        melody_start = min(n.start_beat for n in melody)
        melody_end   = max(n.start_beat + n.duration_beats for n in melody)
        assigned = {assignment.melody, assignment.bass,
                    assignment.countermelody, assignment.drums}

        # Supplement start: merge notes from a channel that covers the beginning
        if melody_start > total_beats_piece * 0.10:
            for ch in sorted(raw.channels(), key=lambda c: -len(raw.notes_for_channel(c))):
                if ch in assigned:
                    continue
                supp = _get_notes(ch)
                if supp and min(n.start_beat for n in supp) < melody_start * 0.5:
                    melody = sorted(melody + supp, key=lambda n: n.start_beat)
                    assigned.add(ch)
                    break

        # Supplement end: merge notes from a channel that covers the tail
        if melody_end < total_beats_piece * 0.90:
            for ch in sorted(raw.channels(), key=lambda c: -len(raw.notes_for_channel(c))):
                if ch in assigned:
                    continue
                supp = _get_notes(ch)
                if supp and max(n.start_beat + n.duration_beats for n in supp) > melody_end:
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
    total_beats    = raw.total_ticks / raw.ticks_per_beat
    phrases  = segment_phrases(melody, beats_per_bar, piece_total_beats=total_beats)
    sections = detect_sections(phrases)

    # ── 6b. Per-section channel assignment ───────────────────────────────────
    # Extract TrackedNote lists for all non-drum channels and re-score roles
    # per section so the melody/bass/counter channels can shift between sections.
    _drum_ch = assignment.drums
    _all_pitched_channels: dict[int, list] = {}
    for ch in raw.channels():
        if ch == _drum_ch:
            continue
        notes_ch = _get_notes(ch)
        if notes_ch:
            _all_pitched_channels[ch] = notes_ch

    section_assignments = assign_channels_per_section(
        _all_pitched_channels, assignment, sections,
    )

    # ── 7. Section labels ────────────────────────────────────────────────────
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
        section_assignments=section_assignments,
        tempo_map=tempo_map,
    )


def listen_audio(
    source: Union[bytes, str, "pathlib.Path"],
    *,
    quantise: bool = True,
    min_motif_occurrences: int = 3,
    separate_stems: bool = True,
    stem_output_dir: Union[str, "pathlib.Path", None] = None,
) -> object:
    """
    Analyse an audio file (WAV/MP3/OGG/FLAC) and return a MusicalAnalysis IR.

    Uses basic-pitch for polyphonic transcription, librosa for tempo/beat
    detection, and optionally Demucs for source separation (vocals, bass,
    drums, other).

    The returned MusicalAnalysis is identical in structure to listen_midi()
    output and can be passed to generate_from_analysis() / presets / etc.

    Parameters
    ----------
    source
        Path to audio file, or raw audio bytes (WAV format).
    quantise
        Snap note timings to a 1/32-note grid after transcription.
    min_motif_occurrences
        Minimum phrase count a motif must appear in to be registered.
    separate_stems
        If True, run Demucs stem separation for better channel assignment
        and to store stems for faithful reproduction output.  Falls back
        to HPSS (librosa-only) if Demucs is not installed.
    stem_output_dir
        Directory to save separated stems.  If None and separate_stems
        is True, stems are kept in memory only (stored in analysis.extra).

    Returns
    -------
    MusicalAnalysis
        Fully populated IR with stem data attached (if separated).

    Example
    -------
        from audio_listener import listen_audio
        from music_generator.compose import generate_from_analysis
        from procgen import export

        analysis = listen_audio("song.wav")
        score    = generate_from_analysis(analysis, base_params={}, seed=0)
        midi_out = export.to_midi(score)
        open("transcription.mid", "wb").write(midi_out)
    """
    from audio_listener.audio_parser     import parse_audio, assign_channels_from_stems
    from audio_listener.channel_router   import assign_channels, assign_channels_per_section
    from audio_listener.pitch_tracker    import extract_notes, quantise_beats
    from audio_listener.tempo_meter      import detect_tempo, detect_tempo_map, detect_time_signature
    from audio_listener.key_detector     import detect_key_mode
    from audio_listener.phrase_segmenter import segment_phrases, detect_sections
    from audio_listener.section_labeler  import label_sections
    from audio_listener.motif_finder     import find_motifs
    from audio_listener.ir_assembler     import assemble

    # ── 1. Transcribe audio → synthetic MidiData ──────────────────────────────
    raw, audio_sr, audio_array = parse_audio(source)

    # ── 1b. Stem separation (optional) ────────────────────────────────────────
    stem_data = None
    if separate_stems:
        try:
            from audio_listener.stem_separator import separate_stems as _sep_stems
            stem_data = _sep_stems(source)
        except ImportError:
            # Demucs not available — fall back to HPSS
            try:
                from audio_listener.stem_separator import separate_hpss
                stem_data = separate_hpss(audio_array, audio_sr)
            except ImportError:
                pass  # No stem separation available — single-channel mode

        # Re-assign channels based on stem information
        if stem_data and stem_data.has_stems:
            raw = assign_channels_from_stems(raw, stem_data.as_dict(), audio_sr)

            # Save stems if requested
            if stem_output_dir:
                from audio_listener.stem_separator import save_stems
                save_stems(stem_data, stem_output_dir)

    # ── 2. Global meter ───────────────────────────────────────────────────────
    # Use the tempo already embedded in the synthetic MidiData from librosa
    tempo_bpm = detect_tempo(raw)
    tempo_map = detect_tempo_map(raw)
    time_sig  = detect_time_signature(raw)
    beats_per_bar = time_sig[0]

    # ── 3. Channel role assignment ────────────────────────────────────────────
    assignment = assign_channels(raw)

    # ── 4. Extract and quantise notes per channel ─────────────────────────────
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
        non_drum = raw.channels() - ({assignment.drums} if assignment.drums is not None else set())
        if non_drum:
            ch     = max(non_drum, key=lambda c: len(raw.notes_for_channel(c)))
            melody = _get_notes(ch)

    # Supplement melody coverage (same logic as listen_midi)
    if melody:
        total_beats_piece = raw.total_ticks / raw.ticks_per_beat
        melody_start = min(n.start_beat for n in melody)
        melody_end   = max(n.start_beat + n.duration_beats for n in melody)
        assigned = {assignment.melody, assignment.bass,
                    assignment.countermelody, assignment.drums}

        if melody_start > total_beats_piece * 0.10:
            for ch in sorted(raw.channels(), key=lambda c: -len(raw.notes_for_channel(c))):
                if ch in assigned:
                    continue
                supp = _get_notes(ch)
                if supp and min(n.start_beat for n in supp) < melody_start * 0.5:
                    melody = sorted(melody + supp, key=lambda n: n.start_beat)
                    assigned.add(ch)
                    break

        if melody_end < total_beats_piece * 0.90:
            for ch in sorted(raw.channels(), key=lambda c: -len(raw.notes_for_channel(c))):
                if ch in assigned:
                    continue
                supp = _get_notes(ch)
                if supp and max(n.start_beat + n.duration_beats for n in supp) > melody_end:
                    melody = sorted(melody + supp, key=lambda n: n.start_beat)
                    break

    # ── 5. Key / mode ─────────────────────────────────────────────────────────
    def _all_non_drum_notes() -> list:
        all_notes: list = []
        drum_ch = assignment.drums
        for ch in raw.channels():
            if ch == drum_ch:
                continue
            all_notes.extend(_get_notes(ch))
        return all_notes

    _bass_key_notes = _get_notes(assignment.bass) if assignment.bass is not None else []
    _all_non_drum   = _all_non_drum_notes()
    key, mode = detect_key_mode(_all_non_drum, _bass_key_notes, [])

    # ── 6. Phrase and section segmentation ────────────────────────────────────
    total_beats = raw.total_ticks / raw.ticks_per_beat
    phrases  = segment_phrases(melody, beats_per_bar, piece_total_beats=total_beats)
    sections = detect_sections(phrases)

    # ── 6b. Per-section channel assignment ────────────────────────────────────
    _drum_ch = assignment.drums
    _all_pitched_channels: dict[int, list] = {}
    for ch in raw.channels():
        if ch == _drum_ch:
            continue
        notes_ch = _get_notes(ch)
        if notes_ch:
            _all_pitched_channels[ch] = notes_ch

    section_assignments = assign_channels_per_section(
        _all_pitched_channels, assignment, sections,
    )

    # ── 7. Section labels ─────────────────────────────────────────────────────
    section_labels = label_sections(sections, phrases, melody, total_beats)

    # ── 8. Motifs ─────────────────────────────────────────────────────────────
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

    # ── 9. Assemble ───────────────────────────────────────────────────────────
    analysis = assemble(
        data=raw,
        assignment=assignment,
        melody=melody, bass=bass, counter=counter, drums=drums,
        tempo_bpm=tempo_bpm, time_sig=time_sig,
        key=key, mode=mode,
        phrases=phrases, sections=sections, section_labels=section_labels,
        motifs=motifs,
        section_assignments=section_assignments,
        tempo_map=tempo_map,
    )

    # Attach stem data and source format info for output routing
    if not hasattr(analysis, '_audio_metadata'):
        analysis._audio_metadata = {}
    analysis._audio_metadata["source_format"] = "audio"
    analysis._audio_metadata["sample_rate"] = audio_sr
    if stem_data and stem_data.has_stems:
        analysis._audio_metadata["stem_data"] = stem_data
        analysis._audio_metadata["has_stems"] = True
    else:
        analysis._audio_metadata["has_stems"] = False

    return analysis
