"""
demo.py - Showcase the MusicDomain procedural composer.

Generates eight named style presets, prints musical analysis for each,
then runs a variety test showing how the same preset sounds across
different seeds.

MIDI files are written to  demo_output/midi/  relative to this script.

Run:
    python demo_music.py
"""

from __future__ import annotations
import os
import sys
import pathlib
import statistics

HERE     = pathlib.Path(__file__).resolve().parent
PROJECTS = HERE.parent
sys.path.insert(0, str(HERE))                  # music_generator package
sys.path.insert(0, str(PROJECTS / "procgen"))  # procgen package

from procgen import Designer, export
from music_generator import MusicDomain
from music_generator import NOTE_NAMES, ROOT_TO_PC, SCALE_INTERVALS, note_str_to_midi
from music_generator.ir import (
    MusicalAnalysis, SectionSpec, MotifDef, TextureHints, EnergyProfile,
)
from music_generator.compose import generate_from_analysis
from music_generator.transforms import (
    mode_swap, transpose, apply_style_preset, assign_motif, compose as ir_compose,
)

OUT_DIR = HERE / "demo_output" / "midi"


# -- Analysis helpers ----------------------------------------------------------

def _to_midi_safe(pitch: str) -> int:
    try:
        return note_str_to_midi(pitch)
    except Exception:
        return 60


def analyze_track(notes) -> dict:
    """Compute musical statistics for a list of Note objects."""
    if not notes:
        return {}

    midis = [_to_midi_safe(n.pitch) for n in notes]
    durs  = [n.duration for n in notes]

    intervals = [abs(midis[i + 1] - midis[i]) for i in range(len(midis) - 1)]
    avg_interval   = statistics.mean(intervals)    if intervals else 0.0
    step_pct       = (sum(1 for x in intervals if x <= 2) / len(intervals) * 100
                      if intervals else 0.0)
    leap_pct       = (sum(1 for x in intervals if x >= 5) / len(intervals) * 100
                      if intervals else 0.0)

    total_beats    = sum(durs)
    note_density   = len(notes) / total_beats if total_beats > 0 else 0

    dur_counts: dict[float, int] = {}
    for d in durs:
        dur_counts[d] = dur_counts.get(d, 0) + 1
    dominant_dur = max(dur_counts, key=dur_counts.get) if dur_counts else 0

    pc_counts: dict[int, int] = {}
    for m in midis:
        pc = m % 12
        pc_counts[pc] = pc_counts.get(pc, 0) + 1
    top_pcs = sorted(pc_counts, key=pc_counts.get, reverse=True)[:4]
    top_pc_names = [NOTE_NAMES[pc] for pc in top_pcs]

    pitch_range = max(midis) - min(midis) if midis else 0

    return {
        "note_count":    len(notes),
        "total_beats":   round(total_beats, 2),
        "avg_interval":  round(avg_interval, 2),
        "step_pct":      round(step_pct, 1),
        "leap_pct":      round(leap_pct, 1),
        "note_density":  round(note_density, 2),
        "dominant_dur":  dominant_dur,
        "top_pcs":       top_pc_names,
        "pitch_range":   pitch_range,
    }


def diatonic_pct(notes, key: str, mode: str) -> float:
    """Percentage of notes that belong to the key's scale."""
    if not notes:
        return 0.0
    root_pc = ROOT_TO_PC[key]
    ivs     = set(SCALE_INTERVALS[mode])
    count   = sum(1 for n in notes
                  if (_to_midi_safe(n.pitch) % 12 - root_pc) % 12 in ivs)
    return round(count / len(notes) * 100, 1)


def voice_ordering_ok(score) -> bool:
    """Check soprano mean pitch > bass mean pitch."""
    means = {}
    for track_idx, track in enumerate(score.tracks):
        role = score.metadata.get("voices", [])[track_idx] if track_idx < len(score.metadata.get("voices", [])) else "?"
        if track.notes:
            means[role] = statistics.mean(_to_midi_safe(n.pitch) for n in track.notes)
    if "soprano" in means and "bass" in means:
        return means["soprano"] > means["bass"]
    return True


def note_line(notes, max_n: int = 16) -> str:
    """One-line rendering of the first N note names + durations."""
    seen = notes[:max_n]
    tokens = []
    for n in seen:
        dur_s = {4.0: "W", 2.0: "H", 1.0: "Q", 0.5: "E", 0.25: "S", 0.125: "T"}.get(n.duration, f"{n.duration:.3g}b")
        tokens.append(f"{n.pitch}{dur_s}")
    suffix = " ..." if len(notes) > max_n else ""
    return "  " + "  ".join(tokens) + suffix


def _dur_label(d: float) -> str:
    return {4.0: "whole", 2.0: "half", 1.0: "quarter",
            0.5: "8th", 0.25: "16th", 0.125: "32nd"}.get(d, f"{d}b")


def print_score(score, name: str, key: str, mode: str, show_voices: int = 1):
    cols = 66
    print()
    print("-" * cols)
    print(f"  {name}")
    print("-" * cols)

    intent = score.metadata.get("intent", {})
    print(f"  Key   : {key} {mode}  .  Tempo: {score.tempo_bpm} BPM  "
          f".  {score.time_signature[0]}/{score.time_signature[1]}")
    print(f"  Form  : {intent.get('form','?')}  .  "
          f"{intent.get('num_phrases','?')} phrases x {intent.get('phrase_length','?')} bars")
    print(f"  Voices: {len(score.tracks)}  "
          f"({', '.join(t.instrument for t in score.tracks)})")

    roles = score.metadata.get("voices", [f"v{i}" for i in range(len(score.tracks))])
    for idx, track in enumerate(score.tracks[:show_voices]):
        role  = roles[idx] if idx < len(roles) else f"v{idx}"
        stats = analyze_track(track.notes)
        if not stats:
            continue
        diat  = diatonic_pct(track.notes, key, mode)
        print(f"\n  [{role.upper()}]  {track.instrument}")
        print(f"    {stats['note_count']} notes  .  {stats['total_beats']} beats  "
              f".  {stats['note_density']} notes/beat")
        print(f"    Step {stats['step_pct']}%  leap {stats['leap_pct']}%  "
              f"avg interval {stats['avg_interval']} semi  range {stats['pitch_range']} semi")
        print(f"    Dominant duration: {_dur_label(stats['dominant_dur'])}  "
              f".  diatonic {diat}%  .  top PCs: {', '.join(stats['top_pcs'])}")
        print(f"    Melody excerpt:")
        print(note_line(track.notes))

    ok = voice_ordering_ok(score)
    total_notes = sum(len(t.notes) for t in score.tracks)
    print(f"\n  Total notes: {total_notes}  .  Voice ordering OK: {ok}")


# -- Style presets -------------------------------------------------------------

PRESETS: list[dict] = [

    {
        "name":  "Bach Two-Voice Invention  (D minor, harpsichord)",
        "key":   "D",  "mode": "minor",
        "params": dict(
            form="binary", num_phrases=4, phrase_length=4,
            num_voices=2, counterpoint=0.85, imitation=0.6,
            inner_voice_style="running_sixteenths", bass_type="walking",
            circle_of_fifths=0.80, chord_complexity=0.50,
            rhythmic_density=0.80, use_ornaments=True,
            step_leap_ratio=0.72, sequence_probability=0.55,
            instruments=["harpsichord"], tempo_bpm=72,
        ),
        "seed": 17,
    },

    {
        "name":  "Bach-Style Organ Fugue  (C minor, organ, 4 voices)",
        "key":   "C",  "mode": "harmonic_minor",
        "params": dict(
            form="through_composed", num_phrases=6, phrase_length=4,
            num_voices=4, counterpoint=0.90, imitation=0.75,
            inner_voice_style="running_sixteenths", bass_type="walking",
            circle_of_fifths=0.85, chord_complexity=0.60,
            rhythmic_density=0.75, use_ornaments=True,
            step_leap_ratio=0.70, sequence_probability=0.65,
            voice_independence=0.8, pedal_point=0.25,
            instruments=["organ"], tempo_bpm=66,
        ),
        "seed": 42,
    },

    {
        "name":  "Classical Sonata Exposition  (C major, piano, alberti bass)",
        "key":   "C",  "mode": "major",
        "params": dict(
            form="binary", num_phrases=4, phrase_length=4,
            num_voices=3, counterpoint=0.25, imitation=0.0,
            inner_voice_style="block_chords", bass_type="alberti",
            circle_of_fifths=0.65, chord_complexity=0.30,
            rhythmic_density=0.55, use_ornaments=False,
            step_leap_ratio=0.65, sequence_probability=0.2,
            voice_independence=0.3, melodic_direction="arch",
            instruments=["piano"], tempo_bpm=120,
        ),
        "seed": 7,
    },

    {
        "name":  "Romantic Waltz  (A major, strings, 3/4)",
        "key":   "A",  "mode": "major",
        "params": dict(
            form="ternary", num_phrases=6, phrase_length=4,
            num_voices=3, counterpoint=0.35, imitation=0.0,
            inner_voice_style="arpeggiated", bass_type="arpeggiated",
            circle_of_fifths=0.60, chord_complexity=0.45,
            rhythmic_density=0.50, use_ornaments=True,
            step_leap_ratio=0.55, sequence_probability=0.30,
            time_signature=[3, 4], melodic_direction="arch",
            instruments=["violin", "strings", "cello"], tempo_bpm=152,
        ),
        "seed": 3,
    },

    {
        "name":  "Modal Celtic Folk  (D dorian, flute + bass)",
        "key":   "D",  "mode": "dorian",
        "params": dict(
            form="binary", num_phrases=4, phrase_length=4,
            num_voices=2, counterpoint=0.55, imitation=0.0,
            inner_voice_style="countermelody", bass_type="walking",
            circle_of_fifths=0.25, chord_complexity=0.15,
            rhythmic_density=0.60, use_ornaments=True,
            step_leap_ratio=0.80, sequence_probability=0.40,
            time_signature=[6, 8], melodic_direction="free",
            instruments=["flute", "bass"], tempo_bpm=108,
        ),
        "seed": 55,
    },

    {
        "name":  "Jazz Ballad  (F major, piano, swing, rich chords)",
        "key":   "F",  "mode": "major",
        "params": dict(
            form="through_composed", num_phrases=4, phrase_length=4,
            num_voices=3, counterpoint=0.45, imitation=0.0,
            inner_voice_style="countermelody", bass_type="walking",
            circle_of_fifths=0.55, chord_complexity=0.85,
            modal_mixture=0.35, rhythmic_density=0.50,
            use_ornaments=False, step_leap_ratio=0.50,
            sequence_probability=0.25, swing=0.65,
            instruments=["piano"], tempo_bpm=60,
        ),
        "seed": 88,
    },

    {
        "name":  "Baroque Passacaglia  (G minor, pedal bass, organ)",
        "key":   "G",  "mode": "minor",
        "params": dict(
            form="through_composed", num_phrases=8, phrase_length=4,
            num_voices=3, counterpoint=0.70, imitation=0.20,
            inner_voice_style="countermelody", bass_type="pedal",
            circle_of_fifths=0.75, chord_complexity=0.50,
            pedal_point=0.50, rhythmic_density=0.65,
            use_ornaments=True, step_leap_ratio=0.70,
            harmonic_rhythm=0.65, voice_independence=0.70,
            instruments=["organ", "organ", "organ"], tempo_bpm=58,
        ),
        "seed": 200,
    },

    {
        "name":  "Dark Phrygian  (E phrygian, sparse, strings)",
        "key":   "E",  "mode": "phrygian",
        "params": dict(
            form="through_composed", num_phrases=4, phrase_length=8,
            num_voices=3, counterpoint=0.60, imitation=0.0,
            inner_voice_style="sustained", bass_type="sustained",
            circle_of_fifths=0.15, chord_complexity=0.35,
            rhythmic_density=0.20, use_ornaments=False,
            step_leap_ratio=0.60, sequence_probability=0.15,
            melodic_direction="descending", voice_independence=0.65,
            instruments=["strings", "strings", "cello"], tempo_bpm=48,
        ),
        "seed": 777,
    },

    # ── Multi-instrument presets ─────────────────────────────────────────────

    {
        "name":  "String Quartet  (G major, violin/violin/viola/cello, motif)",
        "key":   "G",  "mode": "major",
        "params": dict(
            form="binary", num_phrases=4, phrase_length=4,
            num_voices=4, counterpoint=0.65, imitation=0.3,
            inner_voice_style="countermelody", bass_type="walking",
            circle_of_fifths=0.70, chord_complexity=0.40,
            rhythmic_density=0.60, use_ornaments=True,
            step_leap_ratio=0.68, sequence_probability=0.35,
            voice_independence=0.65, melodic_direction="arch",
            melodic_theme="motif",
            instruments=["violin", "violin", "viola", "cello"], tempo_bpm=112,
        ),
        "seed": 301,
    },

    {
        "name":  "Piano Trio  (D major, violin/cello/piano, fugue subject)",
        "key":   "D",  "mode": "major",
        "params": dict(
            form="ternary", num_phrases=6, phrase_length=4,
            num_voices=3, counterpoint=0.75, imitation=0.45,
            inner_voice_style="running_sixteenths", bass_type="walking",
            circle_of_fifths=0.75, chord_complexity=0.45,
            rhythmic_density=0.70, use_ornaments=True,
            step_leap_ratio=0.70, sequence_probability=0.50,
            voice_independence=0.70, melodic_direction="free",
            melodic_theme="fugue_subject",
            instruments=["violin", "cello", "piano"], tempo_bpm=96,
        ),
        "seed": 404,
    },

    {
        "name":  "Wind Quintet  (F major, flute/oboe/clarinet/bassoon, 4v)",
        "key":   "F",  "mode": "major",
        "params": dict(
            form="rondo", num_phrases=6, phrase_length=4,
            num_voices=4, counterpoint=0.50, imitation=0.20,
            inner_voice_style="countermelody", bass_type="walking",
            circle_of_fifths=0.60, chord_complexity=0.35,
            rhythmic_density=0.55, use_ornaments=True,
            step_leap_ratio=0.75, sequence_probability=0.30,
            voice_independence=0.55, melodic_direction="free",
            melodic_theme="motif",
            instruments=["flute", "oboe", "clarinet", "bassoon"], tempo_bpm=120,
        ),
        "seed": 512,
    },

    {
        "name":  "Baroque Fugue  (B minor, harpsichord 4v, fugue subject)",
        "key":   "B",  "mode": "minor",
        "params": dict(
            form="through_composed", num_phrases=8, phrase_length=4,
            num_voices=4, counterpoint=0.90, imitation=0.80,
            inner_voice_style="running_sixteenths", bass_type="walking",
            circle_of_fifths=0.85, chord_complexity=0.55,
            rhythmic_density=0.78, use_ornaments=True,
            step_leap_ratio=0.72, sequence_probability=0.65,
            voice_independence=0.80, pedal_point=0.20,
            melodic_theme="fugue_subject",
            instruments=["harpsichord"], tempo_bpm=64,
        ),
        "seed": 616,
    },

    # ── Rock presets ──────────────────────────────────────────────────────────

    {
        "name":  "Classic Rock  (A mixolydian, electric guitar + bass)",
        "key":   "A",  "mode": "mixolydian",
        "params": dict(
            form="binary", num_phrases=4, phrase_length=4,
            num_voices=2, counterpoint=0.20, imitation=0.0,
            inner_voice_style="block_chords", bass_type="rock",
            circle_of_fifths=0.35, chord_complexity=0.10,
            rhythmic_density=0.60, rhythmic_regularity=0.75,
            use_ornaments=False, step_leap_ratio=0.55,
            sequence_probability=0.30, melodic_direction="arch",
            instruments=["electric_guitar", "bass_guitar"], tempo_bpm=140,
            ending="held", intro="upbeat",
        ),
        "seed": 701,
    },

    {
        "name":  "Blues Rock  (A blues, shuffle, guitar + bass)",
        "key":   "A",  "mode": "blues",
        "params": dict(
            form="through_composed", num_phrases=4, phrase_length=4,
            num_voices=2, counterpoint=0.30, imitation=0.0,
            inner_voice_style="countermelody", bass_type="rock",
            circle_of_fifths=0.30, chord_complexity=0.25,
            rhythmic_density=0.55, rhythmic_regularity=0.50,
            use_ornaments=False, step_leap_ratio=0.50,
            sequence_probability=0.40, swing=0.40,
            instruments=["electric_guitar", "bass_guitar"], tempo_bpm=96,
            ending="held",
        ),
        "seed": 702,
    },

    {
        "name":  "Alt Rock  (E minor pentatonic, power chords, 3 voices)",
        "key":   "E",  "mode": "pentatonic_minor",
        "params": dict(
            form="binary", num_phrases=4, phrase_length=4,
            num_voices=3, counterpoint=0.25, imitation=0.0,
            inner_voice_style="block_chords", bass_type="rock",
            circle_of_fifths=0.40, chord_complexity=0.0,
            rhythmic_density=0.65, rhythmic_regularity=0.70,
            use_ornaments=False, step_leap_ratio=0.60,
            sequence_probability=0.35, melodic_direction="free",
            power_chords=True,
            instruments=["electric_guitar", "electric_guitar", "bass_guitar"],
            tempo_bpm=130, ending="held",
        ),
        "seed": 703,
    },

    {
        "name":  "Hard Rock  (E phrygian, dist guitar, power chords)",
        "key":   "E",  "mode": "phrygian",
        "params": dict(
            form="through_composed", num_phrases=4, phrase_length=4,
            num_voices=3, counterpoint=0.20, imitation=0.0,
            inner_voice_style="block_chords", bass_type="rock",
            circle_of_fifths=0.25, chord_complexity=0.0,
            rhythmic_density=0.70, rhythmic_regularity=0.80,
            use_ornaments=False, step_leap_ratio=0.55,
            sequence_probability=0.20, melodic_direction="descending",
            power_chords=True,
            instruments=["dist_guitar", "dist_guitar", "bass_guitar"],
            tempo_bpm=170, ending="held", intro="upbeat",
        ),
        "seed": 704,
    },

    {
        "name":  "Pop Rock  (C major pentatonic, guitar + bass)",
        "key":   "C",  "mode": "pentatonic_major",
        "params": dict(
            form="binary", num_phrases=4, phrase_length=4,
            num_voices=2, counterpoint=0.25, imitation=0.0,
            inner_voice_style="block_chords", bass_type="rock",
            circle_of_fifths=0.50, chord_complexity=0.0,
            rhythmic_density=0.55, rhythmic_regularity=0.70,
            use_ornaments=False, step_leap_ratio=0.65,
            sequence_probability=0.30, melodic_direction="arch",
            instruments=["electric_guitar", "bass_guitar"], tempo_bpm=124,
            ending="held", intro="upbeat",
        ),
        "seed": 705,
    },
]


def run_demos(out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    for preset in PRESETS:
        name   = preset["name"]
        key    = preset["key"]
        mode   = preset["mode"]
        seed   = preset["seed"]
        params = preset["params"]

        # Fix "sustained" not being a valid inner_voice_style
        if params.get("inner_voice_style") == "sustained":
            params["inner_voice_style"] = "block_chords"

        d = Designer(domain=MusicDomain(), seed=seed)
        d.set("key", key)
        d.set("mode", mode)
        for k, v in params.items():
            d.set(k, v)

        score = d.generate()
        print_score(score, name, key, mode, show_voices=1)

        fname = name.split("(")[0].strip().replace(" ", "_").lower()[:40] + ".mid"
        midi_path = out_dir / fname
        midi_bytes = export.to_midi(score)
        with open(midi_path, "wb") as f:
            f.write(midi_bytes)
        print(f"  -> saved {midi_path.name}  ({len(midi_bytes)} bytes)")


# -- Variety demo --------------------------------------------------------------

def run_variety_demo(out_dir: pathlib.Path):
    """
    Show how seed variation produces different pieces from the same style.
    Uses the Bach invention preset with three different seeds.
    """
    print()
    print("=" * 66)
    print("  VARIETY DEMO -- Same style, different seeds")
    print("  (Bach Two-Voice Invention in D minor)")
    print("=" * 66)

    bach_params = dict(
        form="binary", num_phrases=4, phrase_length=4,
        num_voices=2, counterpoint=0.85, imitation=0.6,
        inner_voice_style="running_sixteenths", bass_type="walking",
        circle_of_fifths=0.80, chord_complexity=0.50,
        rhythmic_density=0.80, use_ornaments=True,
        step_leap_ratio=0.72, sequence_probability=0.55,
        instruments=["harpsichord"], tempo_bpm=72,
    )

    comparison_rows: list[dict] = []

    for seed in [101, 202, 303]:
        d = Designer(domain=MusicDomain(), seed=seed)
        d.set("key", "D")
        d.set("mode", "minor")
        for k, v in bach_params.items():
            d.set(k, v)
        score = d.generate()

        melody = score.tracks[0].notes
        stats  = analyze_track(melody)
        diat   = diatonic_pct(melody, "D", "minor")

        comparison_rows.append({
            "seed":     seed,
            "notes":    stats["note_count"],
            "step%":    stats["step_pct"],
            "leap%":    stats["leap_pct"],
            "interval": stats["avg_interval"],
            "diatonic": diat,
            "excerpt":  note_line(melody, 10),
        })

        fname = f"bach_variety_seed{seed}.mid"
        with open(out_dir / fname, "wb") as f:
            f.write(export.to_midi(score))
        print(f"\n  Seed {seed}")
        print(f"    Notes {stats['note_count']}  step {stats['step_pct']}%  "
              f"leap {stats['leap_pct']}%  interval {stats['avg_interval']}  "
              f"diatonic {diat}%")
        print(f"    {comparison_rows[-1]['excerpt']}")

    # Cross-seed variety check
    excerpts = [r["excerpt"] for r in comparison_rows]
    unique_excerpts = len(set(excerpts))
    print(f"\n  Distinct melody openings across seeds: {unique_excerpts}/3")
    intervals = [r["interval"] for r in comparison_rows]
    print(f"  Avg-interval range: {min(intervals):.2f}-{max(intervals):.2f} semitones")
    all_step = [r["step%"] for r in comparison_rows]
    print(f"  Step% range       : {min(all_step):.1f}%-{max(all_step):.1f}%")


# -- Parameter sweep -----------------------------------------------------------

def run_param_sweep(out_dir: pathlib.Path):
    """
    Sweep a single parameter while holding everything else constant,
    showing how the musical output changes measurably.
    """
    print()
    print("=" * 66)
    print("  PARAMETER SWEEP -- rhythmic_density  (C major, piano)")
    print("=" * 66)
    print(f"  {'density':>10}  {'dom dur':>10}  {'notes/beat':>10}  {'excerpt'}")
    print("  " + "-" * 62)

    for density in [0.1, 0.3, 0.5, 0.7, 0.9]:
        d = Designer(domain=MusicDomain(), seed=5)
        d.set("key", "C")
        d.set("mode", "major")
        d.set("num_voices", 1)
        d.set("rhythmic_density", density)
        d.set("instruments", ["piano"])
        score = d.generate()
        melody = score.tracks[0].notes
        stats  = analyze_track(melody)
        dd     = _dur_label(stats["dominant_dur"])
        print(f"  {density:>10.1f}  {dd:>10}  {stats['note_density']:>10.2f}  "
              f"{note_line(melody, 6)}")

    print()
    print("=" * 66)
    print("  PARAMETER SWEEP -- circle_of_fifths  (C major, piano, 4 bars)")
    print("=" * 66)
    print(f"  {'cof':>8}  {'diatonic%':>10}  top pitch classes")
    print("  " + "-" * 50)

    for cof in [0.0, 0.2, 0.5, 0.8, 1.0]:
        d = Designer(domain=MusicDomain(), seed=5)
        d.set("key", "C")
        d.set("mode", "major")
        d.set("num_voices", 1)
        d.set("circle_of_fifths", cof)
        d.set("chord_complexity", 0.4)
        d.set("harmonic_rhythm", 0.5)
        score = d.generate()
        melody = score.tracks[0].notes
        stats  = analyze_track(melody)
        diat   = diatonic_pct(melody, "C", "major")
        print(f"  {cof:>8.1f}  {diat:>10.1f}  {', '.join(stats['top_pcs'])}")


def run_theme_demo(out_dir: pathlib.Path):
    """
    Demonstrate how the same melodic motif sounds across phrases
    with its transformations (original, inverted, retrograde, retrograde-inv).
    """
    print()
    print("=" * 66)
    print("  THEME DEMO -- Motif transformations across phrases")
    print("=" * 66)

    configs = [
        ("Motif  (G major, string quartet)",   "G", "major",  "motif",        301),
        ("Fugue  (B minor, harpsichord 4v)",    "B", "minor",  "fugue_subject", 616),
        ("No theme (D minor, baseline)",        "D", "minor",  "none",          17),
    ]

    for label, key, mode, theme_type, seed in configs:
        d = Designer(domain=MusicDomain(), seed=seed)
        d.set("key", key)
        d.set("mode", mode)
        d.set("melodic_theme", theme_type)
        d.set("num_voices", 4)
        d.set("counterpoint", 0.80)
        d.set("imitation", 0.50)
        d.set("inner_voice_style", "running_sixteenths")
        d.set("bass_type", "walking")
        d.set("circle_of_fifths", 0.80)
        d.set("chord_complexity", 0.50)
        d.set("rhythmic_density", 0.75)
        d.set("use_ornaments", True)
        d.set("instruments", ["harpsichord"])
        d.set("tempo_bpm", 70)

        score = d.generate()
        m = score.tracks[0].notes
        diat = diatonic_pct(m, key, mode) if m else 0.0
        density = len(m) / sum(n.duration for n in m) if m else 0.0

        print(f"\n  {label}")
        print(f"    Soprano notes: {len(m)}  .  diatonic {diat:.1f}%  .  "
              f"density {density:.2f} notes/beat")
        print(f"    First 12 notes: {note_line(m, 12)}")

        fname  = f"theme_{theme_type}_{key}_{mode}.mid"
        midi_b = export.to_midi(score)
        with open(out_dir / fname, "wb") as f:
            f.write(midi_b)
        print(f"    -> {fname}  ({len(midi_b)} bytes)")


# -- Pipeline demo -------------------------------------------------------------
#
# This section demonstrates the procgen framework pipeline directly:
#   1. designer.explain()  -- inspect the GenerationPlan before running
#   2. designer.trace()    -- generate AND collect every step's raw output
#   3. designer.override() -- swap one generator class without touching the domain


def _summarise_step(name: str, output) -> str:
    """One-line summary of a generator's raw output for pipeline display."""
    if name == "harmonic_plan":
        from collections import Counter
        types = Counter(ev.chord_type for ev in output)
        top   = "  ".join(f"{t}({n})" for t, n in types.most_common(4))
        return (f"{len(output)} events  "
                f"phrases 0-{max(ev.phrase_idx for ev in output)}  "
                f"final_degree={output[-1].degree}  "
                f"types: {top}")
    if name == "theme":
        if output.get("type") == "none":
            return "type=none  (no motif)"
        ivs  = output.get("intervals", [])
        durs = output.get("durations", [])
        iv_s = " ".join(f"{'+' if v >= 0 else ''}{v}" for v in ivs)
        return (f"type={output['type']}  "
                f"intervals=[{iv_s}]  "
                f"dur={durs[0] if durs else '?'}b x{len(durs)}")
    if name == "voice_lines":
        parts = "  ".join(
            f"{v}:{len(events)}" for v, events in sorted(output.items())
        )
        return f"{len(output)} voices  events: {parts}"
    if name == "elaboration":
        parts = "  ".join(
            f"{v}:{len(notes)}" for v, notes in sorted(output.items())
        )
        total = sum(len(n) for n in output.values())
        return f"total={total} notes  per voice: {parts}"
    if name == "score":
        return (f"{output.tempo_bpm} BPM  "
                f"{output.time_signature[0]}/{output.time_signature[1]}  "
                f"{len(output.tracks)} tracks  "
                f"{sum(len(t.notes) for t in output.tracks)} notes")
    return repr(output)[:80]


class _SimpleCadenceGenerator:
    """
    Minimal drop-in replacement for HarmonicPlanGenerator.

    Produces only I-IV-V-I cadence patterns — boring harmonically but
    useful to show how a generator override changes the musical result
    without touching the domain or any other generator.
    """
    def __init__(self, seed):
        from procgen.seeds import Seed
        self.seed = seed if isinstance(seed, Seed) else Seed(seed)

    def generate(self, params: dict, context: dict):
        from music_generator import (
            HarmonicEvent, ROOT_TO_PC, SCALE_INTERVALS,
            DIATONIC_TRIADS, degree_root_pc,
        )
        tonic_pc    = ROOT_TO_PC[params["key"]]
        mode        = params["mode"]
        num_phrases = params["num_phrases"]
        phrase_bars = params["phrase_length"]
        beats_bar   = params["time_signature"][0]
        phrase_beats = phrase_bars * beats_bar

        # Fixed I-IV-V-I progression repeated each phrase
        pattern = [1, 4, 5, 1]
        beats_per_chord = phrase_beats / len(pattern)

        events = []
        for p in range(num_phrases):
            for deg in pattern:
                root = degree_root_pc(tonic_pc, mode, deg)
                ctype = DIATONIC_TRIADS[mode][deg - 1]
                events.append(HarmonicEvent(root, ctype, beats_per_chord, deg, p))
        return events


def run_pipeline_demo(out_dir: pathlib.Path):
    """
    Demonstrate the procgen pipeline visibly using the music domain.

    Shows three framework features not exposed by the other demos:
      1. designer.explain()  — read the plan before generating
      2. designer.trace()    — inspect every step's raw output
      3. designer.override() — hot-swap a generator without touching the domain
    """
    print()
    print("=" * 66)
    print("  PIPELINE DEMO -- procgen framework internals")
    print("  (D minor, harpsichord, 4 voices, fugue_subject theme)")
    print("=" * 66)

    # Shared configuration
    def _build(seed=42, **extra):
        d = Designer(domain=MusicDomain(), seed=seed)
        d.set("key",               "D")
        d.set("mode",              "minor")
        d.set("form",              "binary")
        d.set("num_phrases",       4)
        d.set("phrase_length",     4)
        d.set("num_voices",        4)
        d.set("counterpoint",      0.85)
        d.set("imitation",         0.70)
        d.set("inner_voice_style", "running_sixteenths")
        d.set("bass_type",         "walking")
        d.set("circle_of_fifths",  0.82)
        d.set("chord_complexity",  0.55)
        d.set("rhythmic_density",  0.78)
        d.set("use_ornaments",     True)
        d.set("melodic_theme",     "fugue_subject")
        d.set("instruments",       ["harpsichord"])
        d.set("tempo_bpm",         66)
        for k, v in extra.items():
            d.set(k, v)
        return d

    # ── [1] Plan inspection via dry_run() ─────────────────────────────────
    print()
    print("  [1]  dry_run() -- plan before any generator runs")
    print("  " + "-" * 62)
    d = _build()
    plan = d.dry_run()
    print(f"       Domain  : {plan.metadata['domain']}")
    print(f"       Seed    : {plan.seed}")
    key_params = ["key", "mode", "form", "num_voices", "melodic_theme", "tempo_bpm"]
    intent_s = plan.metadata["intent_summary"]
    print(f"       Intent  : " + "  ".join(
        f"{k}={intent_s[k]}" for k in key_params if k in intent_s
    ))
    print(f"       Steps ({len(plan.steps)}):")
    for i, (name, _params) in enumerate(plan.steps, 1):
        print(f"         {i}. {name}")

    # ── [2] Trace every step's output ──────────────────────────────────────
    print()
    print("  [2]  trace() -- step-by-step raw output")
    print("  " + "-" * 62)
    d2 = _build()
    score_full, trace = d2.trace()

    step_order = ["harmonic_plan", "theme", "voice_lines", "elaboration", "score"]
    for i, step in enumerate(step_order, 1):
        if step in trace:
            summary = _summarise_step(step, trace[step])
            print(f"  {i}. {step:<16} {summary}")

    midi_bytes = export.to_midi(score_full)
    with open(out_dir / "pipeline_full.mid", "wb") as f:
        f.write(midi_bytes)
    print(f"\n       -> pipeline_full.mid  ({len(midi_bytes)} bytes)")

    # ── [3] Generator override ─────────────────────────────────────────────
    print()
    print("  [3]  override() -- swap HarmonicPlanGenerator for SimpleCadenceGenerator")
    print("       (I-IV-V-I only; all other generators unchanged)")
    print("  " + "-" * 62)

    # Original: measure chord variety (unique chord types in harmonic plan)
    from collections import Counter as _Counter
    harm_orig = trace["harmonic_plan"]
    types_orig = len(set(ev.chord_type for ev in harm_orig))
    deg_orig   = len(set(ev.degree    for ev in harm_orig))
    print(f"       Original  : {len(harm_orig)} events  "
          f"{types_orig} unique chord types  "
          f"{deg_orig} unique degrees")

    d3 = _build().override("harmonic_plan", _SimpleCadenceGenerator)
    score_override, trace_override = d3.trace()
    harm_new = trace_override["harmonic_plan"]
    types_new = len(set(ev.chord_type for ev in harm_new))
    deg_new   = len(set(ev.degree    for ev in harm_new))
    print(f"       Overridden: {len(harm_new)} events  "
          f"{types_new} unique chord types  "
          f"{deg_new} unique degrees  (I-IV-V-I only)")

    # Show how the soprano melody changed despite same theme/voice/elaboration config
    sop_orig     = score_full.tracks[0].notes
    sop_override = score_override.tracks[0].notes
    diat_orig     = diatonic_pct(sop_orig,     "D", "minor") if sop_orig     else 0.0
    diat_override = diatonic_pct(sop_override, "D", "minor") if sop_override else 0.0
    print(f"\n       Soprano diatonic: original={diat_orig:.1f}%  overridden={diat_override:.1f}%")
    print(f"       Original   excerpt: {note_line(sop_orig,     8)}")
    print(f"       Overridden excerpt: {note_line(sop_override, 8)}")

    midi_bytes2 = export.to_midi(score_override)
    with open(out_dir / "pipeline_override.mid", "wb") as f:
        f.write(midi_bytes2)
    print(f"\n       -> pipeline_override.mid  ({len(midi_bytes2)} bytes)")


# -- Rock demo -----------------------------------------------------------------

def run_rock_demo(out_dir: pathlib.Path):
    """
    Demonstrate rock styles across multiple seeds — pentatonic/blues scales,
    power chords, rock bass, and modern rock instrumentation.
    """
    print()
    print("=" * 66)
    print("  ROCK DEMO -- Modern rock styles (multi-seed)")
    print("=" * 66)

    rock_configs = [
        ("Classic Rock A mixolydian",   "A", "mixolydian",      [801, 802, 803],
         dict(bass_type="rock", circle_of_fifths=0.35, chord_complexity=0.10,
              rhythmic_density=0.60, instruments=["electric_guitar", "bass_guitar"],
              tempo_bpm=140, num_voices=2, inner_voice_style="block_chords",
              use_ornaments=False, step_leap_ratio=0.55, ending="held")),
        ("Blues Rock A blues",          "A", "blues",            [811, 812, 813],
         dict(bass_type="rock", circle_of_fifths=0.30, chord_complexity=0.25,
              rhythmic_density=0.55, swing=0.40,
              instruments=["electric_guitar", "bass_guitar"],
              tempo_bpm=96, num_voices=2, inner_voice_style="countermelody",
              use_ornaments=False, step_leap_ratio=0.50, ending="held")),
        ("Pentatonic Minor E",          "E", "pentatonic_minor", [821, 822, 823],
         dict(bass_type="rock", power_chords=True, chord_complexity=0.0,
              circle_of_fifths=0.40, rhythmic_density=0.65,
              instruments=["electric_guitar", "electric_guitar", "bass_guitar"],
              tempo_bpm=128, num_voices=3, inner_voice_style="block_chords",
              use_ornaments=False, step_leap_ratio=0.60, ending="held")),
    ]

    for label, key, mode, seeds, base_params in rock_configs:
        print(f"\n  {label}  ({len(seeds)} seeds)")
        print("  " + "-" * 60)

        for seed in seeds:
            d = Designer(domain=MusicDomain(), seed=seed)
            d.set("key", key)
            d.set("mode", mode)
            d.set("form", "binary")
            d.set("num_phrases", 4)
            d.set("phrase_length", 4)
            for k, v in base_params.items():
                d.set(k, v)

            score = d.generate()
            melody = score.tracks[0].notes
            stats  = analyze_track(melody)
            diat   = diatonic_pct(melody, key, mode)

            fname = f"rock_{label.replace(' ','_').lower()[:24]}_s{seed}.mid"
            with open(out_dir / fname, "wb") as f:
                f.write(export.to_midi(score))

            print(f"    seed {seed}: {stats['note_count']:3d} notes  "
                  f"diatonic {diat:5.1f}%  "
                  f"avg-ivl {stats['avg_interval']:.1f}  "
                  f"density {stats['note_density']:.2f}  -> {fname}")


# -- Shared base params for symbolic demos ------------------------------------

_SYMBOLIC_BASE = dict(
    form="binary",
    num_phrases=2,
    phrase_length=4,
    num_voices=3,
    texture="polyphonic",
    counterpoint=0.55,
    imitation=0.0,
    voice_independence=0.5,
    inner_voice_style="countermelody",
    bass_type="walking",
    circle_of_fifths=0.65,
    chord_complexity=0.40,
    modal_mixture=0.10,
    pedal_point=0.0,
    melodic_range=2,
    step_leap_ratio=0.70,
    sequence_probability=0.30,
    use_ornaments=False,
    melodic_direction="arch",
    rhythmic_density=0.55,
    rhythmic_regularity=0.60,
    swing=0.0,
    instruments=["piano"],
    melodic_theme="none",
    power_chords=False,
    ending="held",
    intro="none",
)


# -- Demo 1: Multi-section composition ----------------------------------------

def run_multisection_demo(out_dir: pathlib.Path):
    """
    Compose a four-section piece (Intro -> Verse -> Chorus -> Outro) using the
    MusicalAnalysis IR.  Each section has independent TextureHints and an
    EnergyProfile; the generator is called once per section and the results
    are concatenated into a single MIDI file.

    This demonstrates the symbolic pipeline's section-level controls without
    any pre-built harmonic plans — HarmonicPlanGenerator runs normally for each
    section but receives section-specific texture and energy overrides.
    """
    print()
    print("=" * 66)
    print("  MULTI-SECTION DEMO -- Intro -> Verse -> Chorus -> Outro")
    print("  (A minor, piano, 4 distinct sections)")
    print("=" * 66)

    analysis = MusicalAnalysis(
        key="A", mode="minor", tempo_bpm=96, time_signature=[4, 4],
        form_hint="through_composed",
        sections=[
            SectionSpec(
                id="intro", label="Intro", role="intro",
                harmonic_plan=[],
                texture=TextureHints(
                    bass_type="sustained", inner_voice_style="block_chords",
                    rhythmic_density=0.25, use_ornaments=False,
                    melodic_direction="ascending", counterpoint=0.2,
                ),
                energy=EnergyProfile(level=0.25, arc="ascending"),
                extra_params={"num_phrases": 2, "phrase_length": 2,
                              "circle_of_fifths": 0.4, "intro": "upbeat"},
            ),
            SectionSpec(
                id="verse", label="Verse", role="verse",
                harmonic_plan=[],
                texture=TextureHints(
                    bass_type="walking", inner_voice_style="countermelody",
                    rhythmic_density=0.55, use_ornaments=False,
                    melodic_direction="arch", counterpoint=0.55,
                ),
                energy=EnergyProfile(level=0.55, arc="arch"),
                extra_params={"num_phrases": 2, "phrase_length": 4,
                              "circle_of_fifths": 0.65},
            ),
            SectionSpec(
                id="chorus", label="Chorus", role="chorus",
                harmonic_plan=[],
                texture=TextureHints(
                    bass_type="arpeggiated", inner_voice_style="arpeggiated",
                    rhythmic_density=0.70, use_ornaments=True,
                    melodic_direction="arch", counterpoint=0.70,
                ),
                energy=EnergyProfile(level=0.85, arc="arch"),
                extra_params={"num_phrases": 2, "phrase_length": 4,
                              "circle_of_fifths": 0.75, "chord_complexity": 0.55},
            ),
            SectionSpec(
                id="outro", label="Outro", role="outro",
                harmonic_plan=[],
                texture=TextureHints(
                    bass_type="sustained", inner_voice_style="block_chords",
                    rhythmic_density=0.20, use_ornaments=False,
                    melodic_direction="descending", counterpoint=0.15,
                ),
                energy=EnergyProfile(level=0.20, arc="descending"),
                extra_params={"num_phrases": 2, "phrase_length": 2,
                              "ending": "ritardando"},
            ),
        ],
    )

    score = generate_from_analysis(analysis, base_params=_SYMBOLIC_BASE, seed=900)

    total_notes = sum(len(t.notes) for t in score.tracks)
    total_beats = sum(n.duration for n in score.tracks[0].notes) if score.tracks else 0

    print(f"\n  Structure : {' -> '.join(s.label for s in analysis.sections)}")
    print(f"  Key/Mode  : {analysis.key} {analysis.mode}  .  {analysis.tempo_bpm} BPM")
    print(f"  Voices    : {len(score.tracks)}  ({', '.join(t.instrument for t in score.tracks)})")
    print(f"  Total     : {total_notes} notes  .  ~{total_beats:.0f} beats")

    roles = score.metadata.get("voices", [])
    for i, track in enumerate(score.tracks):
        role   = roles[i] if i < len(roles) else f"v{i}"
        stats  = analyze_track(track.notes)
        if stats:
            print(f"\n  [{role.upper()}]  {stats['note_count']} notes  "
                  f"density {stats['note_density']:.2f}/beat  "
                  f"step {stats['step_pct']:.0f}%")
            print(f"    {note_line(track.notes, 14)}")

    fname = "multisection_intro_verse_chorus_outro.mid"
    with open(out_dir / fname, "wb") as f:
        f.write(export.to_midi(score))
    print(f"\n  -> {fname}  ({sum(len(t.notes) for t in score.tracks)} notes total)")


# -- Demo 2: Mode-swap remix ---------------------------------------------------

def run_mode_swap_demo(out_dir: pathlib.Path):
    """
    Generate a harmonic plan in G major, package it as a MusicalAnalysis,
    then use mode_swap() to derive a parallel G minor version.  Both pieces
    share the exact same scale-degree skeleton and texture; only the chord
    quality and root pitches change.

    Shows how a single procedurally-generated harmonic structure can be
    recoloured across modes as a remix operation.
    """
    print()
    print("=" * 66)
    print("  MODE-SWAP DEMO -- Same degree progression, parallel modes")
    print("  (G major -> G minor, piano, 3 voices)")
    print("=" * 66)

    # Step 1: generate a harmonic plan in G major via trace()
    d = Designer(domain=MusicDomain(), seed=42)
    d.set("key", "G"); d.set("mode", "major")
    d.set("form", "binary"); d.set("num_phrases", 4); d.set("phrase_length", 4)
    d.set("num_voices", 3); d.set("circle_of_fifths", 0.70)
    d.set("chord_complexity", 0.40); d.set("harmonic_rhythm", 0.5)
    d.set("instruments", ["piano"]); d.set("tempo_bpm", 108)
    _, trace = d.trace()
    source_plan = trace["harmonic_plan"]

    print(f"\n  Source plan: {len(source_plan)} events  "
          f"degrees {sorted(set(ev.degree for ev in source_plan))}  "
          f"types {sorted(set(ev.chord_type for ev in source_plan))}")

    # Step 2: build a MusicalAnalysis from the traced plan
    base_section = SectionSpec(
        id="main", label="Main", role="generic",
        harmonic_plan=source_plan,      # inject pre-built plan
        texture=TextureHints(
            bass_type="walking", inner_voice_style="countermelody",
            rhythmic_density=0.60, use_ornaments=True,
        ),
        energy=EnergyProfile(level=0.60, arc="arch"),
    )

    major_analysis = MusicalAnalysis(
        key="G", mode="major", tempo_bpm=108, time_signature=[4, 4],
        sections=[base_section],
    )

    # Step 3: derive minor version via transform
    minor_analysis = mode_swap(major_analysis, "minor")

    print(f"\n  Major plan types: {sorted(set(ev.chord_type for ev in major_analysis.sections[0].harmonic_plan))}")
    print(f"  Minor plan types: {sorted(set(ev.chord_type for ev in minor_analysis.sections[0].harmonic_plan))}")

    base = dict(_SYMBOLIC_BASE)
    base.update(num_voices=3, instruments=["piano"], tempo_bpm=108,
                use_ornaments=True, rhythmic_density=0.60)

    for label, analysis in [("major", major_analysis), ("minor", minor_analysis)]:
        score = generate_from_analysis(analysis, base_params=base, seed=300)
        melody = score.tracks[0].notes
        stats  = analyze_track(melody)
        diat   = diatonic_pct(melody, analysis.key, analysis.mode)
        print(f"\n  {analysis.key} {analysis.mode.upper():8s}  "
              f"{stats['note_count']:3d} notes  "
              f"step {stats['step_pct']:.0f}%  "
              f"diatonic {diat:.0f}%")
        print(f"    {note_line(melody, 12)}")

        fname = f"mode_swap_G_{label}.mid"
        with open(out_dir / fname, "wb") as f:
            f.write(export.to_midi(score))
        print(f"    -> {fname}")


# -- Demo 3: Theme threading ---------------------------------------------------

def run_theme_thread_demo(out_dir: pathlib.Path):
    """
    Define a single MotifDef and thread it through three contrasting sections
    (baroque -> jazz -> rock).  Each section has a completely different texture
    but plays the same motivic intervals (in its transformation for that phrase).

    This demonstrates how the symbolic layer separates thematic identity from
    style, which is the core of remix: keep the musical idea, change the dress.
    """
    print()
    print("=" * 66)
    print("  THEME THREADING DEMO -- One motif, three styles")
    print("  (C major, motif woven through baroque / jazz / rock sections)")
    print("=" * 66)

    # A memorable four-note motif: up a third, step up, step down (da-da-da-DUM shape)
    motif = MotifDef.from_intervals(
        id="main_motif",
        type="motif",
        intervals=[3, 2, -2],        # +minor 3rd, +whole step, -whole step
        durations=[0.5, 0.5, 0.5, 1.0],
    )
    print(f"\n  Motif intervals : {motif.intervals}")
    print(f"  Inverted        : {motif.inverted}")
    print(f"  Retrograde      : {motif.retrograde}")

    analysis = MusicalAnalysis(
        key="C", mode="major", tempo_bpm=100, time_signature=[4, 4],
        sections=[
            SectionSpec(
                id="baroque", label="Baroque", role="generic",
                harmonic_plan=[],
                motif_id="main_motif",
                texture=TextureHints(
                    bass_type="walking", inner_voice_style="running_sixteenths",
                    rhythmic_density=0.78, use_ornaments=True, swing=0.0,
                    counterpoint=0.85, step_leap_ratio=0.72,
                    instruments=["harpsichord"],
                ),
                energy=EnergyProfile(level=0.70, arc="arch"),
                extra_params={"num_phrases": 2, "phrase_length": 4,
                              "circle_of_fifths": 0.82, "tempo_bpm": 72},
            ),
            SectionSpec(
                id="jazz", label="Jazz", role="generic",
                harmonic_plan=[],
                motif_id="main_motif",
                texture=TextureHints(
                    bass_type="walking", inner_voice_style="countermelody",
                    rhythmic_density=0.50, swing=0.65, use_ornaments=False,
                    counterpoint=0.45, step_leap_ratio=0.50,
                    instruments=["piano"],
                ),
                energy=EnergyProfile(level=0.60, arc="arch"),
                extra_params={"num_phrases": 2, "phrase_length": 4,
                              "chord_complexity": 0.80, "modal_mixture": 0.35,
                              "tempo_bpm": 60},
            ),
            SectionSpec(
                id="rock", label="Rock", role="generic",
                harmonic_plan=[],
                motif_id="main_motif",
                texture=TextureHints(
                    bass_type="rock", inner_voice_style="block_chords",
                    rhythmic_density=0.65, swing=0.0, use_ornaments=False,
                    counterpoint=0.20, rhythmic_regularity=0.80,
                    instruments=["electric_guitar", "bass_guitar"],
                ),
                energy=EnergyProfile(level=0.85, arc="arch"),
                extra_params={"num_phrases": 2, "phrase_length": 4,
                              "num_voices": 2, "chord_complexity": 0.10,
                              "tempo_bpm": 140, "ending": "held"},
            ),
        ],
    )
    analysis.add_motif(motif)

    score = generate_from_analysis(analysis, base_params=_SYMBOLIC_BASE, seed=500)

    roles = score.metadata.get("voices", [])
    print()
    section_labels = [s.label for s in analysis.sections]
    styles = ["baroque", "jazz", "rock"]
    for i, (label, style) in enumerate(zip(section_labels, styles)):
        print(f"  [{label.upper()} / {style}]")

    if score.tracks:
        melody = score.tracks[0].notes
        print(f"\n  Combined soprano: {len(melody)} notes")
        print(f"    {note_line(melody, 16)}")

    fname = "theme_thread_baroque_jazz_rock.mid"
    with open(out_dir / fname, "wb") as f:
        f.write(export.to_midi(score))
    total = sum(len(t.notes) for t in score.tracks)
    print(f"\n  -> {fname}  ({total} notes, {len(score.tracks)} tracks)")


# -- Demo 4: Energy arc --------------------------------------------------------

def run_energy_arc_demo(out_dir: pathlib.Path):
    """
    Build a five-section piece with an explicit energy arc:
        calm -> building -> peak -> release -> resolution

    Each section gets the same harmonic style but its EnergyProfile.level drives
    rhythmic_density automatically.  This shows how the IR's energy layer provides
    a high-level compositional shape that the generator then realises in detail.
    """
    print()
    print("=" * 66)
    print("  ENERGY ARC DEMO -- Shaped energy across five sections")
    print("  (E dorian, strings, calm -> peak -> resolution)")
    print("=" * 66)

    ENERGY_LEVELS = {
        "calm":       (0.20, "ascending"),
        "building":   (0.50, "ascending"),
        "peak":       (0.90, "arch"),
        "release":    (0.55, "descending"),
        "resolution": (0.20, "descending"),
    }

    sections = []
    for section_id, (level, arc) in ENERGY_LEVELS.items():
        sections.append(SectionSpec(
            id=section_id,
            label=section_id.capitalize(),
            role="generic",
            harmonic_plan=[],
            texture=TextureHints(
                bass_type="arpeggiated" if level > 0.5 else "sustained",
                inner_voice_style="countermelody" if level > 0.4 else "block_chords",
                use_ornaments=(level > 0.6),
            ),
            energy=EnergyProfile(level=level, arc=arc),
            extra_params={
                "num_phrases": 2,
                "phrase_length": 4,
                "circle_of_fifths": 0.5 + level * 0.3,
                "chord_complexity": 0.2 + level * 0.5,
            },
        ))

    analysis = MusicalAnalysis(
        key="E", mode="dorian", tempo_bpm=80, time_signature=[4, 4],
        sections=sections,
    )

    base = dict(_SYMBOLIC_BASE)
    base.update(
        num_voices=3,
        instruments=["strings", "viola", "cello"],
        step_leap_ratio=0.65,
        voice_independence=0.60,
        counterpoint=0.60,
    )

    score = generate_from_analysis(analysis, base_params=base, seed=700)

    print(f"\n  {'Section':<14} {'Energy':>7}  {'Arc':<12}  {'Notes':>6}  Excerpt")
    print("  " + "-" * 62)

    # Split the combined score back into per-section note counts (approximate)
    if score.tracks:
        soprano = score.tracks[0].notes
        # Estimate section boundaries from cumulative beat count
        section_beats_list = []
        for sec in analysis.sections:
            np = sec.extra_params.get("num_phrases", 2)
            pl = sec.extra_params.get("phrase_length", 4)
            ts = analysis.time_signature[0]
            section_beats_list.append(np * pl * ts)

        cursor = 0.0
        note_idx = 0
        for sec, target_beats in zip(analysis.sections, section_beats_list):
            sec_notes = []
            accum = 0.0
            while note_idx < len(soprano) and accum < target_beats - 0.01:
                n = soprano[note_idx]
                sec_notes.append(n)
                accum += n.duration
                note_idx += 1
            level, arc = ENERGY_LEVELS[sec.id]
            excerpt = "  ".join(
                f"{n.pitch}{['W','H','Q','E','S'][min(4,int(-0.5+1/max(n.duration,0.125)))]}"
                for n in sec_notes[:6]
            )
            print(f"  {sec.label:<14} {level:>7.2f}  {arc:<12}  {len(sec_notes):>6}  {excerpt}")

    fname = "energy_arc_dorian.mid"
    with open(out_dir / fname, "wb") as f:
        f.write(export.to_midi(score))
    total = sum(len(t.notes) for t in score.tracks)
    print(f"\n  -> {fname}  ({total} notes total)")


# -- Entry point ---------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("=" * 66)
    print("  procgen  --  MusicDomain Demo")
    print("=" * 66)

    run_demos(OUT_DIR)
    run_variety_demo(OUT_DIR)
    run_param_sweep(OUT_DIR)
    run_theme_demo(OUT_DIR)
    run_pipeline_demo(OUT_DIR)
    run_rock_demo(OUT_DIR)

    # ── Symbolic IR demos ────────────────────────────────────────────────
    print()
    print("=" * 66)
    print("  SYMBOLIC IR DEMOS")
    print("  (MusicalAnalysis -> generate_from_analysis)")
    print("=" * 66)
    run_multisection_demo(OUT_DIR)
    run_mode_swap_demo(OUT_DIR)
    run_theme_thread_demo(OUT_DIR)
    run_energy_arc_demo(OUT_DIR)

    print()
    print(f"All MIDI files written to: {OUT_DIR}")
    print("Done.")
