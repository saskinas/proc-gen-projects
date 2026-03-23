"""
audio_listener demo — Listen to a NES MIDI file and generate a reproduction.

Usage:
    python demo.py <path/to/nes_music.mid>

If no file is given, a synthetic test MIDI is generated internally.
"""

from __future__ import annotations

import sys
import pathlib

HERE     = pathlib.Path(__file__).resolve().parent
PROJECTS = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECTS / "music_generator"))
sys.path.insert(0, str(PROJECTS / "procgen"))

from audio_listener import listen_midi
from music_generator.compose import generate_from_analysis
from procgen import export

OUT_DIR = HERE / "demo_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── NES-appropriate base params ───────────────────────────────────────────────

NES_BASE = dict(
    num_voices=3,
    texture="polyphonic",
    counterpoint=0.35,
    imitation=0.0,
    voice_independence=0.5,
    chord_complexity=0.20,
    harmonic_rhythm=0.4,
    circle_of_fifths=0.70,
    modal_mixture=0.05,
    pedal_point=0.0,
    melodic_range=2,
    step_leap_ratio=0.60,
    sequence_probability=0.35,
    use_ornaments=False,
    swing=0.0,
    rhythmic_regularity=0.70,
    rhythmic_density=0.55,
    melodic_theme="motif",
    melodic_theme_dev="flat",
    form="binary",
    num_phrases=2,
    phrase_length=4,
    bass_type="walking",
    inner_voice_style="countermelody",
    instruments=["square"],   # NES pulse-wave feel
    ending="held",
    intro="none",
    power_chords=False,
    drum_style="none",     # overridden from analysis
    drum_intensity=0.5,
)


def _make_test_midi() -> bytes:
    """
    Build a minimal but realistic SMF test MIDI in C major.

    Two channels:
      - Channel 0 (melody): simple C major scale fragment × 4 phrases
      - Channel 1 (bass):   root notes on downbeats

    4/4 time, 120 BPM, 4 phrases of 4 bars each.
    """
    import struct

    tpb = 480     # ticks per beat
    uspb = 500_000  # 120 BPM

    def varlen(v: int) -> bytes:
        result = [v & 0x7F]; v >>= 7
        while v:
            result.insert(0, (v & 0x7F) | 0x80); v >>= 7
        return bytes(result)

    def note_events(ch: int, pitch: int, vel: int, start: int, dur: int) -> bytes:
        b = bytearray()
        b += varlen(start) + bytes([0x90 | ch, pitch, vel])
        b += varlen(dur)   + bytes([0x80 | ch, pitch, 0])
        return bytes(b)

    # Melody: C D E F G A G F E D C (quarter notes) repeated 4×
    melody_pitches = [60, 62, 64, 65, 67, 69, 67, 65, 64, 62, 60, 60,
                      64, 67, 69, 71, 72, 71, 69, 67, 65, 64, 62, 60,
                      60, 62, 64, 65, 67, 65, 64, 62, 60, 62, 64, 65,
                      67, 69, 71, 72, 71, 69, 67, 65, 64, 62, 60, 60]
    # Bass: root on each bar downbeat
    bass_pitches   = [48, 48, 43, 43, 48, 48, 53, 53,
                      48, 48, 43, 43, 48, 48, 55, 55,
                      48, 48, 43, 43, 41, 41, 43, 43,
                      48, 48, 43, 43, 48, 48, 48, 48]

    mel_track = bytearray()
    tick = 0
    for p in melody_pitches:
        mel_track += note_events(0, p, 80, tick if mel_track == b"" else 0, tpb)
        tick = 0   # delta after first

    # Actually build properly (delta times need to be from previous event)
    mel_track = bytearray()
    mel_track += varlen(0) + bytes([0xFF, 0x03, len(b"Melody")]) + b"Melody"
    prev_ev_tick = 0
    for i, p in enumerate(melody_pitches):
        abs_tick = i * tpb
        delta_on  = abs_tick - prev_ev_tick
        delta_off = tpb
        mel_track += varlen(delta_on) + bytes([0x90, p, 80])
        mel_track += varlen(delta_off - 1) + bytes([0x80, p, 0])
        prev_ev_tick = abs_tick + tpb

    mel_track += varlen(0) + b"\xFF\x2F\x00"

    bass_track = bytearray()
    bass_track += varlen(0) + bytes([0xFF, 0x03, len(b"Triangle")]) + b"Triangle"
    prev_ev_tick = 0
    for i, p in enumerate(bass_pitches):
        abs_tick  = i * tpb * 2   # half notes
        delta_on  = abs_tick - prev_ev_tick
        delta_off = tpb * 2
        bass_track += varlen(delta_on) + bytes([0x91, p, 70])
        bass_track += varlen(delta_off - 1) + bytes([0x81, p, 0])
        prev_ev_tick = abs_tick + tpb * 2

    bass_track += varlen(0) + b"\xFF\x2F\x00"

    # Tempo track
    tempo_track = bytearray()
    tempo_track += varlen(0) + b"\xFF\x51\x03"
    tempo_track += bytes([(uspb >> 16) & 0xFF, (uspb >> 8) & 0xFF, uspb & 0xFF])
    tempo_track += varlen(0) + b"\xFF\x58\x04" + bytes([4, 2, 24, 8])   # 4/4
    tempo_track += varlen(0) + b"\xFF\x2F\x00"

    def make_chunk(tag: bytes, data: bytes) -> bytes:
        return tag + struct.pack(">I", len(data)) + data

    header = struct.pack(">4sIHHH", b"MThd", 6, 1, 3, tpb)
    body   = (make_chunk(b"MTrk", bytes(tempo_track)) +
              make_chunk(b"MTrk", bytes(mel_track)) +
              make_chunk(b"MTrk", bytes(bass_track)))
    return header + body


def run_demo(midi_path: pathlib.Path | None = None) -> None:
    print()
    print("=" * 66)
    print("  AUDIO LISTENER DEMO")
    print("  MIDI -> MusicalAnalysis -> Reproduction")
    print("=" * 66)

    if midi_path is None:
        print("\n  [No input file given — using built-in C major test MIDI]")
        midi_bytes = _make_test_midi()
        # Save the test MIDI for reference
        test_path = OUT_DIR / "test_input.mid"
        test_path.write_bytes(midi_bytes)
        print(f"  Test MIDI written to: {test_path}")
        source = midi_bytes
        stem   = "test_c_major"
    else:
        source = midi_path
        stem   = midi_path.stem

    # ── 1. Analyse ────────────────────────────────────────────────────────────
    print("\n  [1] Analysing MIDI...")
    analysis = listen_midi(source)

    print(f"\n  Key/Mode    : {analysis.key} {analysis.mode}")
    print(f"  Tempo       : {analysis.tempo_bpm} BPM")
    print(f"  Time sig    : {analysis.time_signature[0]}/{analysis.time_signature[1]}")
    print(f"  Sections    : {len(analysis.sections)}")
    print(f"  Motifs      : {len(analysis.motifs)}")

    print(f"\n  {'Section':<16} {'Role':<10} {'Key':<5} {'Chords':>6}  TextureHints excerpt")
    print("  " + "-" * 66)
    for sec in analysis.sections:
        ep       = sec.extra_params
        key_str  = ep.get("key", analysis.key)
        n_chords = len(sec.harmonic_plan)
        tx       = sec.texture
        tex_str  = f"bass={tx.bass_type or 'inh'}  density={tx.rhythmic_density or 'inh'}"
        print(f"  {sec.label:<16} {sec.role:<10} {key_str:<5} {n_chords:>6}  {tex_str}")

    if analysis.motifs:
        print("\n  Detected motifs:")
        for m in analysis.motifs.values():
            print(f"    {m.id}: intervals={m.intervals}  durations={m.durations}")

    # ── 2. Generate reproduction ──────────────────────────────────────────────
    print("\n  [2] Generating reproduction...")

    # Adapt base params from analysis
    base = dict(NES_BASE)
    n_voices = max(2, min(4, max(
        (sec.texture.num_voices or 2) for sec in analysis.sections
    )))
    base["num_voices"] = n_voices
    base["melodic_theme"] = "motif" if analysis.motifs else "none"

    # Determine drum style from analysis — use most common non-"none" style
    from collections import Counter
    style_counts = Counter(
        sec.texture.drum_style
        for sec in analysis.sections
        if sec.texture.drum_style and sec.texture.drum_style != "none"
    )
    if style_counts:
        base["drum_style"] = style_counts.most_common(1)[0][0]
        intensities = [sec.texture.drum_intensity for sec in analysis.sections
                       if sec.texture.drum_intensity and sec.texture.drum_intensity > 0]
        base["drum_intensity"] = sum(intensities) / len(intensities) if intensities else 0.5

    # Use instruments from section texture hints if available
    all_instrs = [
        instr for sec in analysis.sections
        if sec.texture.instruments
        for instr in sec.texture.instruments
    ]
    # (For now, harpsichord is the NES-appropriate default — keep unless analysis overrides)

    score = generate_from_analysis(analysis, base_params=base, seed=42)

    total_notes = sum(len(t.notes) for t in score.tracks)
    print(f"\n  Generated: {len(score.tracks)} tracks, {total_notes} notes")
    for t in score.tracks:
        print(f"    {t.instrument:<16} {len(t.notes):>5} notes")

    # ── 3. Save MIDI ─────────────────────────────────────────────────────────
    out_path = OUT_DIR / f"{stem}_reproduction.mid"
    out_path.write_bytes(export.to_midi(score))
    print(f"\n  -> Reproduction MIDI: {out_path}")

    # Also save the analysis as JSON for inspection
    json_path = OUT_DIR / f"{stem}_analysis.json"
    json_path.write_text(analysis.to_json(), encoding="utf-8")
    print(f"  -> Analysis JSON   : {json_path}")


if __name__ == "__main__":
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run_demo(path)
