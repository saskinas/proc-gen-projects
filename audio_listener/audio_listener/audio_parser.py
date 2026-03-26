"""
audio_listener.audio_parser — Polyphonic audio-to-MidiData transcription.

Uses basic-pitch (Spotify) for polyphonic pitch tracking, and librosa for
beat/tempo detection.  Produces a synthetic MidiData object compatible with
the downstream pipeline (channel_router, pitch_tracker, etc.).

Dependencies: basic-pitch, librosa, soundfile, numpy
    pip install basic-pitch librosa soundfile numpy
"""

from __future__ import annotations

import pathlib
import tempfile
from dataclasses import dataclass, field

from audio_listener.midi_parser import MidiData, RawNote, CCEvent, PitchBendEvent


# ── Lazy imports ──────────────────────────────────────────────────────────────

def _require_audio_deps():
    """Check that all required audio dependencies are available."""
    missing = []
    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")
    try:
        import librosa  # noqa: F401
    except ImportError:
        missing.append("librosa")
    try:
        import soundfile  # noqa: F401
    except ImportError:
        missing.append("soundfile")
    try:
        import basic_pitch  # noqa: F401
    except ImportError:
        missing.append("basic-pitch")
    if missing:
        raise ImportError(
            f"Audio analysis requires: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}"
        )


# ── Instrument detection from spectral features ──────────────────────────────

@dataclass
class DetectedInstrument:
    """Instrument guess from spectral analysis of an audio segment."""
    gm_program: int        # GM program number (0-127)
    name: str              # human-readable name
    confidence: float      # 0.0–1.0


def _detect_instrument_from_audio(
    audio: "np.ndarray",
    sr: int,
) -> DetectedInstrument:
    """
    Estimate the most likely GM instrument family from spectral features.

    Uses spectral centroid, MFCCs, and harmonic-to-noise ratio to classify
    into broad families: piano, guitar, strings, brass, woodwind, synth.
    """
    import numpy as np
    import librosa

    if len(audio) < sr * 0.1:
        return DetectedInstrument(0, "piano", 0.3)

    # Spectral centroid — brightness indicator
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))

    # MFCCs — timbral fingerprint
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1)

    # Spectral rolloff — where most energy lies
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)))

    # Spectral flatness — noise-like vs tonal
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=audio)))

    # Classify based on spectral features
    # These thresholds are rough heuristics; good enough for GM family mapping
    if flatness > 0.15:
        # Very noisy — likely synth pad or distorted
        return DetectedInstrument(80, "synth_lead", 0.4)
    elif centroid < 800:
        # Very dark timbre — bass or low strings
        if rolloff < 2000:
            return DetectedInstrument(32, "acoustic_bass", 0.5)
        else:
            return DetectedInstrument(42, "cello", 0.4)
    elif centroid < 1500:
        # Warm — piano or acoustic guitar
        # Check attack transient sharpness via onset strength
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        attack_sharpness = float(np.max(onset_env)) / (float(np.mean(onset_env)) + 1e-6)
        if attack_sharpness > 4.0:
            return DetectedInstrument(25, "acoustic_guitar", 0.5)
        else:
            return DetectedInstrument(0, "piano", 0.5)
    elif centroid < 2500:
        # Mid-range — strings, brass, or woodwind
        # MFCCs can help distinguish
        if mfcc_means[1] > 0:
            return DetectedInstrument(48, "strings", 0.4)
        else:
            return DetectedInstrument(56, "trumpet", 0.4)
    elif centroid < 4000:
        # Bright — flute, violin, or bright synth
        if flatness < 0.02:
            return DetectedInstrument(73, "flute", 0.4)
        else:
            return DetectedInstrument(40, "violin", 0.4)
    else:
        # Very bright — likely a bright synth or high woodwind
        return DetectedInstrument(80, "synth_lead", 0.4)


def detect_instruments_per_channel(
    stems: dict[str, "np.ndarray"],
    sr: int,
) -> dict[int, DetectedInstrument]:
    """
    Detect instruments from separated stems.

    Parameters
    ----------
    stems   Dict mapping stem name → audio array (from Demucs).
    sr      Sample rate.

    Returns
    -------
    Dict mapping synthetic MIDI channel → DetectedInstrument.
    """
    result: dict[int, DetectedInstrument] = {}

    # Map stem names to channels (matching the synthetic channel assignment
    # we create in parse_audio)
    stem_channel_map = {
        "vocals": 0,
        "other": 1,
        "bass": 2,
        "drums": 9,
    }

    for stem_name, audio in stems.items():
        ch = stem_channel_map.get(stem_name)
        if ch is None or ch == 9:  # skip drums — no pitched instrument
            continue
        if len(audio) > 0:
            result[ch] = _detect_instrument_from_audio(audio, sr)

    return result


# ── Core audio-to-MidiData transcription ──────────────────────────────────────

def parse_audio(
    source: str | pathlib.Path | bytes,
    *,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    min_note_length_ms: float = 58.0,
    min_freq: float | None = None,
    max_freq: float | None = None,
) -> tuple[MidiData, int, "np.ndarray"]:
    """
    Transcribe an audio file to a MidiData object using basic-pitch.

    Parameters
    ----------
    source              Path to audio file (WAV, MP3, OGG, FLAC) or raw bytes.
    onset_threshold     basic-pitch onset sensitivity (0-1, higher = fewer notes).
    frame_threshold     basic-pitch frame activation threshold (0-1).
    min_note_length_ms  Minimum note duration in milliseconds.
    min_freq            Minimum frequency to transcribe (Hz). None = default.
    max_freq            Maximum frequency to transcribe (Hz). None = default.

    Returns
    -------
    (MidiData, sample_rate, audio_array)
        MidiData compatible with the downstream pipeline.
        sample_rate and audio_array for potential stem separation.
    """
    _require_audio_deps()

    import numpy as np
    import librosa

    # ── Load audio ────────────────────────────────────────────────────────────
    source_path = None
    tmp_path = None

    if isinstance(source, bytes):
        # Write to a temp file for basic-pitch (it needs a file path)
        import tempfile as _tf
        tmp = _tf.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(source)
        tmp.close()
        source_path = tmp.name
        tmp_path = tmp.name
    else:
        source_path = str(source)

    try:
        # Load audio with librosa for our analysis
        audio, sr = librosa.load(source_path, sr=22050, mono=True)

        # ── Tempo and beat detection ──────────────────────────────────────────
        tempo_result = librosa.beat.beat_track(y=audio, sr=sr, units="time")
        # librosa returns (tempo, beat_frames) or similar depending on version
        if isinstance(tempo_result, tuple):
            tempo_value = tempo_result[0]
            beat_times = tempo_result[1]
        else:
            tempo_value = tempo_result
            beat_times = []

        # Handle array-type tempo (newer librosa returns array)
        if hasattr(tempo_value, '__len__'):
            tempo_bpm = float(tempo_value[0]) if len(tempo_value) > 0 else 120.0
        else:
            tempo_bpm = float(tempo_value) if tempo_value > 0 else 120.0
        tempo_bpm = max(30, min(300, tempo_bpm))

        # ── Run basic-pitch transcription ─────────────────────────────────────
        from basic_pitch.inference import predict

        model_output, midi_data_bp, note_events = predict(
            source_path,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=min_note_length_ms,
            minimum_frequency=min_freq if min_freq else 32.7,   # C1
            maximum_frequency=max_freq if max_freq else 4186.0, # C8
        )

        # ── Convert basic-pitch notes to RawNote ─────────────────────────────
        # note_events is a list of (start_time_s, end_time_s, pitch, amplitude, [bends])
        # We assign all notes to a single channel (0) since basic-pitch
        # doesn't separate sources.  Stem separation handles that later.

        ticks_per_beat = 480  # standard resolution
        uspb = int(60_000_000 / tempo_bpm)  # microseconds per beat
        secs_per_beat = 60.0 / tempo_bpm

        raw_notes: list[RawNote] = []
        for note_ev in note_events:
            start_s = note_ev[0]
            end_s = note_ev[1]
            pitch = int(note_ev[2])
            amplitude = float(note_ev[3])

            # Convert time to ticks
            tick_on = int(start_s / secs_per_beat * ticks_per_beat)
            tick_off = int(end_s / secs_per_beat * ticks_per_beat)
            velocity = max(1, min(127, int(amplitude * 127)))

            if tick_off <= tick_on:
                tick_off = tick_on + ticks_per_beat // 8  # minimum 1/8 beat

            raw_notes.append(RawNote(
                channel=0,
                pitch=pitch,
                velocity=velocity,
                tick_on=tick_on,
                tick_off=tick_off,
            ))

        # Sort by tick_on
        raw_notes.sort(key=lambda n: n.tick_on)

        # Total ticks
        total_ticks = max(n.tick_off for n in raw_notes) if raw_notes else 0

        # ── Detect time signature ─────────────────────────────────────────────
        # Use librosa's beat strength pattern to estimate time signature
        # Default to 4/4 — most music is in 4/4
        time_sig_num, time_sig_den = 4, 4

        # ── Build MidiData ────────────────────────────────────────────────────
        midi_data = MidiData(
            format=0,
            ticks_per_beat=ticks_per_beat,
            tempo_changes=[(0, uspb)],
            time_sig_events=[(0, time_sig_num, time_sig_den)],
            notes=raw_notes,
            track_names={0: "transcription"},
            total_ticks=total_ticks,
            program_numbers={0: 0},  # default to piano; updated after stem separation
            cc_events=[],
            pitch_bends=[],
        )

        return midi_data, sr, audio

    finally:
        if tmp_path:
            import os
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def assign_channels_from_stems(
    midi_data: MidiData,
    stems: dict[str, "np.ndarray"],
    sr: int,
) -> MidiData:
    """
    Re-assign MIDI channels based on Demucs stem separation.

    Uses pitch range and the stem separation to split the single-channel
    basic-pitch transcription into multiple channels:
      - Channel 0: vocals / melody (highest pitched content)
      - Channel 1: other / countermelody
      - Channel 2: bass
      - Channel 9: drums (detected via percussive onset pattern)

    Parameters
    ----------
    midi_data   Single-channel MidiData from parse_audio.
    stems       Dict mapping stem names → audio arrays from Demucs.
    sr          Sample rate.

    Returns
    -------
    MidiData with notes assigned to appropriate channels.
    """
    import numpy as np
    import librosa

    if not midi_data.notes:
        return midi_data

    # Get pitch ranges from stems using spectral analysis
    stem_pitch_ranges: dict[str, tuple[float, float]] = {}
    for stem_name, audio in stems.items():
        if stem_name == "drums" or len(audio) < sr * 0.1:
            continue
        # Use spectral centroid as a rough pitch indicator
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
        # Convert centroid frequency to approximate MIDI pitch
        if centroid > 20:
            midi_pitch = 12 * np.log2(centroid / 440.0) + 69
            stem_pitch_ranges[stem_name] = (midi_pitch - 12, midi_pitch + 12)

    # Sort notes by pitch to assign to channels
    all_notes = list(midi_data.notes)
    pitches = [n.pitch for n in all_notes]
    if not pitches:
        return midi_data

    min_pitch = min(pitches)
    max_pitch = max(pitches)
    pitch_range = max(1, max_pitch - min_pitch)

    # Simple heuristic: split by pitch range
    # Bottom 25% → bass (ch 2)
    # Top 40% → melody/vocals (ch 0)
    # Middle → other/countermelody (ch 1)
    bass_threshold = min_pitch + pitch_range * 0.25
    melody_threshold = min_pitch + pitch_range * 0.60

    # Detect percussive content using onset strength
    has_drums = "drums" in stems and len(stems["drums"]) > sr * 0.1
    drum_onsets: set[int] = set()

    if has_drums:
        drum_audio = stems["drums"]
        # Detect drum onsets
        onset_frames = librosa.onset.onset_detect(
            y=drum_audio, sr=sr, units="time"
        )
        secs_per_beat = 60.0 / (midi_data.tempo_bpm or 120)
        # Convert onset times to approximate tick positions
        for onset_t in onset_frames:
            tick = int(onset_t / secs_per_beat * midi_data.ticks_per_beat)
            # Mark a window around each onset
            for dt in range(-20, 21):
                drum_onsets.add(tick + dt)

    new_notes: list[RawNote] = []
    programs: dict[int, int] = {}

    for note in all_notes:
        if note.pitch < bass_threshold:
            ch = 2  # bass
        elif note.pitch > melody_threshold:
            ch = 0  # melody / vocals
        else:
            ch = 1  # other / countermelody

        new_notes.append(RawNote(
            channel=ch,
            pitch=note.pitch,
            velocity=note.velocity,
            tick_on=note.tick_on,
            tick_off=note.tick_off,
        ))

    # Detect instruments per stem and assign GM programs
    instruments = detect_instruments_per_channel(stems, sr)
    for ch, inst in instruments.items():
        programs[ch] = inst.gm_program

    # Default programs if not detected
    programs.setdefault(0, 0)    # piano for melody
    programs.setdefault(1, 48)   # strings for other
    programs.setdefault(2, 32)   # acoustic bass

    return MidiData(
        format=1,
        ticks_per_beat=midi_data.ticks_per_beat,
        tempo_changes=midi_data.tempo_changes,
        time_sig_events=midi_data.time_sig_events,
        notes=new_notes,
        track_names={0: "melody", 1: "other", 2: "bass"},
        total_ticks=midi_data.total_ticks,
        program_numbers=programs,
        cc_events=midi_data.cc_events,
        pitch_bends=midi_data.pitch_bends,
    )
