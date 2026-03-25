"""
procgen.synthesize — Pure-Python MIDI-to-WAV synthesis using numpy + mido.

Provides:
    midi_to_wav(midi_bytes, sample_rate=44100) -> bytes  (WAV file bytes)

Waveforms are chosen per GM instrument family:
    Piano          — sine blend
    Organ          — square
    Guitar/Bass    — triangle
    Strings        — sawtooth
    Brass/Reed     — square + sawtooth blend
    Pipe/Flute     — sine
    Synth Lead/Pad — square
    Drums (ch 9)   — noise/sine synthesis per drum type
"""
from __future__ import annotations

import io
import wave

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    import mido
    _HAS_MIDO = True
except ImportError:
    _HAS_MIDO = False


_SAMPLE_RATE = 44100


def _require_deps() -> None:
    missing = []
    if not _HAS_NUMPY:
        missing.append("numpy")
    if not _HAS_MIDO:
        missing.append("mido")
    if missing:
        raise ImportError(
            f"WAV synthesis requires: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}"
        )


# ── Waveform generators ────────────────────────────────────────────────────────

def _midi_to_freq(pitch: int) -> float:
    return 440.0 * (2.0 ** ((pitch - 69) / 12.0))


def _square(freq: float, n: int, sr: int) -> "np.ndarray":
    t = np.arange(n) / sr
    return np.sign(np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _sine(freq: float, n: int, sr: int) -> "np.ndarray":
    t = np.arange(n) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _triangle(freq: float, n: int, sr: int) -> "np.ndarray":
    t     = np.arange(n) / sr
    phase = (freq * t) % 1.0
    return np.where(phase < 0.5, 4 * phase - 1, 3 - 4 * phase).astype(np.float32)


def _sawtooth(freq: float, n: int, sr: int) -> "np.ndarray":
    t     = np.arange(n) / sr
    phase = (freq * t) % 1.0
    return (2 * phase - 1).astype(np.float32)


def _waveform_for_program(prog: int, freq: float, n: int, sr: int) -> "np.ndarray":
    """Select waveform based on GM program family (prog 0-127)."""
    fam = prog // 8
    if fam == 0:               # Piano
        return _sine(freq, n, sr) * 0.65 + _triangle(freq, n, sr) * 0.35
    elif fam in (1, 2):        # Chromatic perc, Organ
        return _square(freq, n, sr)
    elif fam in (3, 4):        # Guitar, Bass
        return _triangle(freq, n, sr)
    elif fam in (5, 6):        # Strings, Ensemble
        return _sawtooth(freq, n, sr)
    elif fam in (7, 8):        # Brass, Reed
        return _square(freq, n, sr) * 0.6 + _sawtooth(freq, n, sr) * 0.4
    elif fam == 9:             # Pipe / Flute
        return _sine(freq, n, sr)
    elif fam in (10, 11):      # Synth Lead, Synth Pad
        return _square(freq, n, sr)
    else:
        return _sine(freq, n, sr)


def _adsr(
    n: int,
    sr: int,
    attack:  float = 0.01,
    decay:   float = 0.05,
    sustain: float = 0.75,
    release: float = 0.08,
) -> "np.ndarray":
    """Build ADSR amplitude envelope of length n samples."""
    a = min(int(attack  * sr), n)
    d = min(int(decay   * sr), n - a)
    r = min(int(release * sr), n - a - d)
    s = n - a - d - r
    if s < 0:
        s, r = 0, max(0, n - a - d)
        if a + d + r > n:
            d = max(0, n - a - r)

    env = np.zeros(n, dtype=np.float32)
    if a > 0:
        env[:a]           = np.linspace(0.0, 1.0, a)
    if d > 0:
        env[a:a+d]        = np.linspace(1.0, sustain, d)
    if s > 0:
        env[a+d:a+d+s]    = sustain
    if r > 0:
        env[a+d+s:n]      = np.linspace(sustain, 0.0, n - (a + d + s))
    return env


def _adsr_for_program(prog: int) -> dict:
    fam = prog // 8
    if fam == 0:       return dict(attack=0.005, decay=0.10, sustain=0.60, release=0.15)
    elif fam == 1:     return dict(attack=0.002, decay=0.08, sustain=0.30, release=0.10)
    elif fam == 2:     return dict(attack=0.02,  decay=0.01, sustain=0.90, release=0.05)
    elif fam == 3:     return dict(attack=0.003, decay=0.15, sustain=0.40, release=0.10)
    elif fam == 4:     return dict(attack=0.005, decay=0.10, sustain=0.70, release=0.08)
    elif fam in (5,6): return dict(attack=0.05,  decay=0.05, sustain=0.80, release=0.15)
    elif fam in (7,8): return dict(attack=0.02,  decay=0.05, sustain=0.80, release=0.08)
    elif fam == 9:     return dict(attack=0.03,  decay=0.02, sustain=0.90, release=0.10)
    elif fam in(10,11):return dict(attack=0.01,  decay=0.05, sustain=0.80, release=0.10)
    else:              return dict(attack=0.01,  decay=0.05, sustain=0.70, release=0.10)


# ── Drum synthesis ─────────────────────────────────────────────────────────────

_DRUM_PARAMS: dict[int, dict] = {
    35: {"kind": "kick",  "dur": 0.25},
    36: {"kind": "kick",  "dur": 0.25},
    37: {"kind": "snare", "dur": 0.12},
    38: {"kind": "snare", "dur": 0.15},
    39: {"kind": "snare", "dur": 0.12},
    40: {"kind": "snare", "dur": 0.12},
    41: {"kind": "tom",   "dur": 0.20, "freq": 80},
    42: {"kind": "hihat", "dur": 0.04},
    43: {"kind": "tom",   "dur": 0.20, "freq": 90},
    44: {"kind": "hihat", "dur": 0.04},
    45: {"kind": "tom",   "dur": 0.20, "freq": 100},
    46: {"kind": "hihat", "dur": 0.25},
    47: {"kind": "tom",   "dur": 0.20, "freq": 110},
    48: {"kind": "tom",   "dur": 0.20, "freq": 130},
    49: {"kind": "crash", "dur": 0.50},
    50: {"kind": "tom",   "dur": 0.20, "freq": 150},
    51: {"kind": "crash", "dur": 0.35},
    52: {"kind": "crash", "dur": 0.50},
    57: {"kind": "crash", "dur": 0.50},
    59: {"kind": "crash", "dur": 0.35},
}


def _synth_drum(pitch: int, velocity: int, sr: int) -> "np.ndarray":
    """Synthesize a single drum hit. Returns float32 array."""
    params = _DRUM_PARAMS.get(pitch, {"kind": "snare", "dur": 0.10})
    kind   = params["kind"]
    dur    = params.get("dur", 0.10)
    n      = int(dur * sr)
    vel_f  = velocity / 127.0
    rng    = np.random.default_rng(pitch)   # deterministic per drum type

    t = np.arange(n) / sr

    if kind == "kick":
        freq_sweep = 150 * np.exp(-12 * t)
        phase_arr  = np.cumsum(2 * np.pi * freq_sweep / sr)
        osc = np.sin(phase_arr).astype(np.float32)
        env = np.exp(-8 * t).astype(np.float32)
        return osc * env * vel_f

    elif kind == "snare":
        noise = rng.standard_normal(n).astype(np.float32)
        body  = _sine(200, n, sr) * 0.3
        env   = np.exp(-18 * t).astype(np.float32)
        return (noise * 0.7 + body) * env * vel_f

    elif kind == "hihat":
        noise = rng.standard_normal(n).astype(np.float32)
        env   = np.exp(-40 * t).astype(np.float32)
        return noise * env * vel_f * 0.4

    elif kind == "tom":
        freq = params.get("freq", 100)
        body = _sine(freq, n, sr) + _sine(freq * 1.5, n, sr) * 0.3
        env  = np.exp(-10 * t).astype(np.float32)
        return body * env * vel_f * 0.7

    elif kind == "crash":
        noise = rng.standard_normal(n).astype(np.float32)
        env   = np.exp(-5 * t).astype(np.float32)
        return noise * env * vel_f * 0.35

    # fallback
    noise = rng.standard_normal(n).astype(np.float32)
    env   = np.exp(-15 * t).astype(np.float32)
    return noise * env * vel_f


# ── Core synthesis ─────────────────────────────────────────────────────────────

def midi_to_wav(midi_bytes: bytes, sample_rate: int = _SAMPLE_RATE) -> bytes:
    """
    Synthesize a MIDI file to WAV audio.

    Parameters
    ----------
    midi_bytes    Raw bytes of a standard MIDI (.mid) file.
    sample_rate   Output sample rate in Hz (default 44100).

    Returns
    -------
    bytes  WAV file data (16-bit PCM, mono, at *sample_rate* Hz).

    Raises
    ------
    ImportError  If numpy or mido are not installed.
    """
    _require_deps()

    sr  = sample_rate
    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))

    # ── 1. Build absolute-time event list ────────────────────────────────────
    tempo    = 500_000   # 120 BPM default
    cur_time = 0.0
    abs_events: list[tuple[float, object]] = []

    for msg in mido.merge_tracks(mid.tracks):
        if not hasattr(msg, "time"):
            continue
        dt        = mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
        cur_time += dt
        if msg.type == "set_tempo":
            tempo = msg.tempo
        abs_events.append((cur_time, msg))

    if not abs_events:
        return _empty_wav(sr)

    total_secs    = abs_events[-1][0] + 2.0     # 2s tail for release
    total_samples = int(total_secs * sr)
    buf           = np.zeros(total_samples, dtype=np.float32)

    # ── 2. Track program changes and build note event list ───────────────────
    ch_program: dict[int, int] = {}
    active_on:  dict[tuple[int, int], tuple[float, int]] = {}   # (ch,pitch) → (on_t, vel)
    note_events: list[tuple[int, int, float, float, int]] = []  # ch,pitch,on,off,vel

    for t, msg in abs_events:
        if msg.type == "program_change":
            ch_program[msg.channel] = msg.program
        elif msg.type in ("note_on", "note_off"):
            ch    = msg.channel
            pitch = msg.note
            vel   = getattr(msg, "velocity", 0)
            key   = (ch, pitch)
            is_on = msg.type == "note_on" and vel > 0

            if is_on:
                # If already on (re-trigger), close previous
                if key in active_on:
                    on_t, v = active_on.pop(key)
                    note_events.append((ch, pitch, on_t, t, v))
                active_on[key] = (t, vel)
            else:
                if key in active_on:
                    on_t, v = active_on.pop(key)
                    note_events.append((ch, pitch, on_t, t, v))

    # Close notes still open at end
    for (ch, pitch), (on_t, v) in active_on.items():
        note_events.append((ch, pitch, on_t, total_secs - 1.0, v))

    # ── 3. Render each note into the buffer ──────────────────────────────────
    for ch, pitch, on_t, off_t, vel in note_events:
        dur = max(0.01, off_t - on_t)

        start_samp = int(on_t * sr)
        if start_samp >= total_samples:
            continue

        if ch == 9:
            # Percussion channel — noise/sine synthesis
            drum_buf = _synth_drum(pitch, vel, sr)
            n        = min(len(drum_buf), total_samples - start_samp)
            if n > 0:
                buf[start_samp : start_samp + n] += drum_buf[:n] * 0.45
        else:
            prog  = ch_program.get(ch, 0)
            freq  = _midi_to_freq(pitch)
            vel_f = vel / 127.0

            # Note buffer: duration + short release tail
            adsr_p     = _adsr_for_program(prog)
            rel_tail   = adsr_p["release"]
            n          = min(int((dur + rel_tail) * sr), total_samples - start_samp)
            if n <= 0:
                continue

            wave_arr = _waveform_for_program(prog, freq, n, sr)
            # Override dur for ADSR so envelope knows the sustain portion
            env = _adsr(n, sr, **adsr_p)
            buf[start_samp : start_samp + n] += wave_arr * env * vel_f * 0.12

    # ── 4. Normalise and convert to 16-bit PCM ───────────────────────────────
    peak = float(np.max(np.abs(buf)))
    if peak > 1e-6:
        buf = buf * (0.90 / peak)

    pcm = (buf * 32767.0).clip(-32768, 32767).astype(np.int16)

    # ── 5. Pack into WAV container ───────────────────────────────────────────
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)       # 16-bit
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

    return wav_io.getvalue()


def _empty_wav(sr: int) -> bytes:
    """Return a valid WAV file containing 1 second of silence."""
    if _HAS_NUMPY:
        pcm = np.zeros(sr, dtype=np.int16)
        data = pcm.tobytes()
    else:
        data = b"\x00\x00" * sr
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data)
    return wav_io.getvalue()
