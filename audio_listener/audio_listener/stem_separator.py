"""
audio_listener.stem_separator — Audio source separation using Demucs.

Separates an audio file into stems (vocals, bass, drums, other) using
Meta's Demucs model.  Stems are stored alongside the IR for faithful
reproduction and hybrid output routing.

Dependencies: demucs, torch, torchaudio
    pip install demucs
"""

from __future__ import annotations

import pathlib
import io
import wave
from dataclasses import dataclass, field

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class StemData:
    """Container for separated audio stems."""
    vocals: "np.ndarray | None" = None   # vocals stem (mono float32)
    bass: "np.ndarray | None" = None     # bass stem (mono float32)
    drums: "np.ndarray | None" = None    # drums stem (mono float32)
    other: "np.ndarray | None" = None    # other / harmony stem (mono float32)
    sample_rate: int = 44100
    source_path: str = ""                # original audio path

    def as_dict(self) -> dict[str, "np.ndarray"]:
        """Return stems as a dict (only non-None stems)."""
        result = {}
        if self.vocals is not None and len(self.vocals) > 0:
            result["vocals"] = self.vocals
        if self.bass is not None and len(self.bass) > 0:
            result["bass"] = self.bass
        if self.drums is not None and len(self.drums) > 0:
            result["drums"] = self.drums
        if self.other is not None and len(self.other) > 0:
            result["other"] = self.other
        return result

    @property
    def has_stems(self) -> bool:
        return bool(self.as_dict())


def _require_demucs():
    """Check that Demucs and its dependencies are available."""
    missing = []
    if not _HAS_NUMPY:
        missing.append("numpy")
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    try:
        import demucs  # noqa: F401
    except ImportError:
        missing.append("demucs")
    if missing:
        raise ImportError(
            f"Stem separation requires: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}"
        )


# ── HPSS fallback (librosa-only, no Demucs needed) ───────────────────────────

def separate_hpss(
    audio: "np.ndarray",
    sr: int,
) -> StemData:
    """
    Fallback stem separation using librosa's HPSS (Harmonic-Percussive
    Source Separation).  Only separates into harmonic + percussive.

    Much less accurate than Demucs but requires no GPU or large model.
    """
    import librosa

    # HPSS → harmonic vs percussive
    harmonic, percussive = librosa.effects.hpss(audio)

    # Split harmonic into bass vs treble using a simple frequency filter
    # Apply a low-pass filter for bass (< ~250 Hz)
    bass = librosa.effects.preemphasis(harmonic, coef=-0.97)
    # Simple approach: low-pass via STFT
    stft_h = librosa.stft(harmonic)
    n_bins = stft_h.shape[0]
    freqs = librosa.fft_frequencies(sr=sr, n_fft=(n_bins - 1) * 2)

    # Bass: keep only bins below 250 Hz
    bass_mask = np.zeros(n_bins)
    bass_mask[freqs < 250] = 1.0
    bass_stft = stft_h * bass_mask[:, np.newaxis]
    bass_audio = librosa.istft(bass_stft, length=len(harmonic))

    # Other: everything above 250 Hz in harmonic
    other_mask = 1.0 - bass_mask
    other_stft = stft_h * other_mask[:, np.newaxis]
    other_audio = librosa.istft(other_stft, length=len(harmonic))

    return StemData(
        vocals=None,    # HPSS can't separate vocals
        bass=bass_audio.astype(np.float32),
        drums=percussive.astype(np.float32),
        other=other_audio.astype(np.float32),
        sample_rate=sr,
    )


# ── Demucs separation ────────────────────────────────────────────────────────

def separate_stems(
    source: str | pathlib.Path | bytes,
    *,
    model_name: str = "htdemucs",
    device: str | None = None,
) -> StemData:
    """
    Separate an audio file into stems using Demucs.

    Parameters
    ----------
    source       Path to audio file or raw bytes.
    model_name   Demucs model name (default: htdemucs — hybrid transformer).
    device       "cuda", "cpu", or None for auto-detect.

    Returns
    -------
    StemData with separated stems as mono float32 numpy arrays.
    """
    _require_demucs()

    import torch
    import torchaudio
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    # ── Load model ────────────────────────────────────────────────────────────
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_model(model_name)
    model.to(device)
    model.eval()

    # ── Load audio ────────────────────────────────────────────────────────────
    source_path = ""
    tmp_path = None

    if isinstance(source, bytes):
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(source)
        tmp.close()
        source_path = tmp.name
        tmp_path = tmp.name
    else:
        source_path = str(source)

    try:
        waveform, sr = torchaudio.load(source_path)

        # Resample to model's expected sample rate if needed
        model_sr = model.samplerate
        if sr != model_sr:
            resampler = torchaudio.transforms.Resample(sr, model_sr)
            waveform = resampler(waveform)
            sr = model_sr

        # Ensure stereo (Demucs expects 2 channels)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]

        # Add batch dimension: (batch, channels, samples)
        waveform = waveform.unsqueeze(0).to(device)

        # ── Run separation ────────────────────────────────────────────────────
        with torch.no_grad():
            estimates = apply_model(model, waveform, device=device)

        # estimates shape: (batch, sources, channels, samples)
        # Sources for htdemucs: drums, bass, other, vocals
        source_names = model.sources  # e.g., ['drums', 'bass', 'other', 'vocals']

        stems_dict: dict[str, np.ndarray] = {}
        for i, name in enumerate(source_names):
            # Convert to mono numpy
            stem = estimates[0, i].cpu().numpy()  # (channels, samples)
            mono = np.mean(stem, axis=0).astype(np.float32)
            stems_dict[name] = mono

        return StemData(
            vocals=stems_dict.get("vocals"),
            bass=stems_dict.get("bass"),
            drums=stems_dict.get("drums"),
            other=stems_dict.get("other"),
            sample_rate=sr,
            source_path=source_path,
        )

    finally:
        if tmp_path:
            import os
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ── Stem I/O ──────────────────────────────────────────────────────────────────

def save_stems(
    stem_data: StemData,
    output_dir: str | pathlib.Path,
    prefix: str = "stem",
) -> dict[str, pathlib.Path]:
    """
    Save stems as individual WAV files in output_dir.

    Returns dict mapping stem name → file path.
    """
    if not _HAS_NUMPY:
        raise ImportError("numpy is required to save stems")

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, pathlib.Path] = {}
    for name, audio in stem_data.as_dict().items():
        path = output_dir / f"{prefix}_{name}.wav"
        _write_wav(audio, stem_data.sample_rate, path)
        saved[name] = path

    return saved


def _write_wav(audio: "np.ndarray", sr: int, path: pathlib.Path) -> None:
    """Write a mono float32 numpy array as a 16-bit PCM WAV file."""
    import numpy as np

    peak = float(np.max(np.abs(audio)))
    if peak > 1e-6:
        audio = audio * (0.95 / peak)

    pcm = (audio * 32767.0).clip(-32768, 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


# ── Stem-based output routing ────────────────────────────────────────────────

def remix_with_stems(
    stem_data: StemData,
    *,
    tempo_factor: float = 1.0,
    pitch_shift_semitones: int = 0,
) -> "np.ndarray":
    """
    Re-mix stems with optional tempo change and pitch shift.

    This is the "faithful reproduction" path — uses actual audio stems
    rather than re-synthesis.  Suitable for untransformed or simple
    (transpose/tempo) transforms.

    Parameters
    ----------
    stem_data               Separated stems.
    tempo_factor            Speed multiplier (1.0 = original, 1.5 = 50% faster).
    pitch_shift_semitones   Pitch shift in semitones (0 = original).

    Returns
    -------
    np.ndarray  Mixed audio as mono float32.
    """
    import numpy as np
    import librosa

    stems = stem_data.as_dict()
    if not stems:
        return np.zeros(1, dtype=np.float32)

    sr = stem_data.sample_rate

    # Process each stem
    processed: list[np.ndarray] = []
    for name, audio in stems.items():
        a = audio.copy()

        # Time stretch if needed
        if abs(tempo_factor - 1.0) > 0.01:
            a = librosa.effects.time_stretch(a, rate=tempo_factor)

        # Pitch shift if needed
        if pitch_shift_semitones != 0 and name != "drums":
            a = librosa.effects.pitch_shift(
                a, sr=sr, n_steps=pitch_shift_semitones
            )

        processed.append(a)

    # Mix to common length
    max_len = max(len(a) for a in processed)
    mixed = np.zeros(max_len, dtype=np.float32)
    for a in processed:
        mixed[:len(a)] += a

    # Normalize
    peak = float(np.max(np.abs(mixed)))
    if peak > 1e-6:
        mixed = mixed * (0.90 / peak)

    return mixed


def stems_to_wav(
    stem_data: StemData,
    *,
    tempo_factor: float = 1.0,
    pitch_shift_semitones: int = 0,
) -> bytes:
    """
    Remix stems and return as WAV bytes.

    This is the public API for stem-based output.
    """
    import numpy as np

    audio = remix_with_stems(
        stem_data,
        tempo_factor=tempo_factor,
        pitch_shift_semitones=pitch_shift_semitones,
    )

    buf = io.BytesIO()
    pcm = (audio * 32767.0).clip(-32768, 32767).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(stem_data.sample_rate)
        wf.writeframes(pcm.tobytes())

    return buf.getvalue()
