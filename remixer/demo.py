"""
remixer demo — Remix Kirby MIDI files using the symbolic IR pipeline.

Usage
-----
    python demo.py                    # remix all 4 Kirby MIDIs
    python demo.py grape-garden       # remix only grape-garden
    python demo.py --list             # print available songs and presets

Output
------
    demo_output/<song>_<preset>.mid

Each output MIDI is a remix of the original with a different musical treatment.
Run a reproduction baseline first (no transforms) to compare with the originals.
"""

from __future__ import annotations

import sys
import pathlib

# ── Path bootstrap ─────────────────────────────────────────────────────────────
HERE     = pathlib.Path(__file__).resolve().parent
PROJECTS = HERE.parent
for pkg in ("remixer", "audio_listener", "music_generator", "procgen"):
    sys.path.insert(0, str(PROJECTS / pkg))

from remixer import analyze, describe, remix_to_file
from remixer.presets import (
    JAZZ, BAROQUE, CLASSICAL, FOLK, AMBIENT, ROCK,
    DARK_MINOR, RELATIVE_MINOR,
    EPIC, MIRROR_WORLD, RETROGRADE, KALEIDOSCOPE,
    FRAGMENTED, REHARMONIZED,
)
from music_generator.transforms import (
    transpose, mode_swap, tempo_scale, apply_style_preset,
    modulate_section, sequence_motif, invert_melody,
    harmonize_in_thirds, octave_shift,
    compose,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
KIRBY_DIR = PROJECTS / "audio_listener" / "sample_input"
OUT_DIR   = HERE / "demo_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Song catalogue ────────────────────────────────────────────────────────────
SONGS = {
    "grape-garden":             KIRBY_DIR / "grape-garden.mid",
    "vegetable-valley":         KIRBY_DIR / "vegetable-valley.mid",
    "rainbow-resort":           KIRBY_DIR / "rainbow-resort-stage-select.mid",
    "final-boss":               KIRBY_DIR / "final-boss.mid",
}


def _out(song: str, preset: str) -> pathlib.Path:
    return OUT_DIR / f"{song}_{preset}.mid"


# ── Per-song demo suites ───────────────────────────────────────────────────────

def demo_grape_garden(analysis):
    song = "grape-garden"
    src  = SONGS[song]

    print(f"\n{'-'*66}")
    print(f"  GRAPE GARDEN  ({src.name})")
    print(f"{'-'*66}")

    # Baseline: pure reproduction (no transforms)
    remix_to_file(src, _out(song, "00_original"),        seed=42)

    # Tonal colour changes — showcase mode_swap / transpose
    remix_to_file(src, _out(song, "01_dark_minor"),      DARK_MINOR,     seed=42)
    remix_to_file(src, _out(song, "02_relative_minor"),  RELATIVE_MINOR, seed=42)

    # Style transplants — what if this were a different genre?
    remix_to_file(src, _out(song, "03_jazz"),            JAZZ,           seed=42)
    remix_to_file(src, _out(song, "04_baroque"),         BAROQUE,        seed=42)
    remix_to_file(src, _out(song, "05_folk"),            FOLK,           seed=42)

    # Energy extremes
    remix_to_file(src, _out(song, "06_epic"),            EPIC,           seed=42)
    remix_to_file(src, _out(song, "07_ambient"),         AMBIENT,        seed=42)

    # Voice-level transforms — melodic manipulation
    remix_to_file(src, _out(song, "08_mirror_world"),    MIRROR_WORLD,   seed=42)
    remix_to_file(src, _out(song, "09_kaleidoscope"),    KALEIDOSCOPE,   seed=42)

    # Motivic development — work with the detected patterns
    remix_to_file(src, _out(song, "10_fragmented"),      FRAGMENTED,     seed=42)

    # Compound remix: jazz + dark minor
    remix_to_file(src, _out(song, "11_dark_jazz"),
        DARK_MINOR, JAZZ, seed=42)

    # Compound remix: baroque + invert + high harmonic complexity
    remix_to_file(src, _out(song, "12_baroque_mirror"),
        MIRROR_WORLD, BAROQUE, seed=42)

    # Compound: epic rock
    remix_to_file(src, _out(song, "13_epic_rock"),
        EPIC, ROCK, seed=42)


def demo_vegetable_valley(analysis):
    song = "vegetable-valley"
    src  = SONGS[song]

    print(f"\n{'-'*66}")
    print(f"  VEGETABLE VALLEY  ({src.name})")
    print(f"{'-'*66}")

    remix_to_file(src, _out(song, "00_original"),       seed=42)
    remix_to_file(src, _out(song, "01_jazz"),           JAZZ,           seed=42)
    remix_to_file(src, _out(song, "02_dark_minor"),     DARK_MINOR,     seed=42)
    remix_to_file(src, _out(song, "03_baroque"),        BAROQUE,        seed=42)
    remix_to_file(src, _out(song, "04_retrograde"),     RETROGRADE,     seed=42)
    remix_to_file(src, _out(song, "05_epic"),           EPIC,           seed=42)
    remix_to_file(src, _out(song, "06_folk"),           FOLK,           seed=42)
    # Compound: minor + retrograde (haunted garden)
    remix_to_file(src, _out(song, "07_haunted"),
        RELATIVE_MINOR, RETROGRADE, seed=42)


def demo_rainbow_resort(analysis):
    song = "rainbow-resort"
    src  = SONGS[song]

    print(f"\n{'-'*66}")
    print(f"  RAINBOW RESORT  ({src.name})")
    print(f"{'-'*66}")

    remix_to_file(src, _out(song, "00_original"),       seed=42)
    remix_to_file(src, _out(song, "01_ambient"),        AMBIENT,        seed=42)
    remix_to_file(src, _out(song, "02_jazz"),           JAZZ,           seed=42)
    remix_to_file(src, _out(song, "03_dark_minor"),     DARK_MINOR,     seed=42)
    remix_to_file(src, _out(song, "04_baroque"),        BAROQUE,        seed=42)
    remix_to_file(src, _out(song, "05_mirror_world"),   MIRROR_WORLD,   seed=42)
    # Compound: dreamy ambient minor
    remix_to_file(src, _out(song, "06_dream_minor"),
        RELATIVE_MINOR, AMBIENT, seed=42)
    # Classical with reharmonization
    remix_to_file(src, _out(song, "07_classical_reharmonized"),
        CLASSICAL, REHARMONIZED, seed=42)


def demo_final_boss(analysis):
    song = "final-boss"
    src  = SONGS[song]

    print(f"\n{'-'*66}")
    print(f"  FINAL BOSS  ({src.name})")
    print(f"{'-'*66}")

    remix_to_file(src, _out(song, "00_original"),       seed=42)
    remix_to_file(src, _out(song, "01_epic"),           EPIC,           seed=42)
    remix_to_file(src, _out(song, "02_dark_minor"),     DARK_MINOR,     seed=42)
    remix_to_file(src, _out(song, "03_jazz"),           JAZZ,           seed=42)
    # Motivic development — final boss themes are usually motif-dense
    remix_to_file(src, _out(song, "04_fragmented"),     FRAGMENTED,     seed=42)
    remix_to_file(src, _out(song, "05_kaleidoscope"),   KALEIDOSCOPE,   seed=42)
    remix_to_file(src, _out(song, "06_reharmonized"),   REHARMONIZED,   seed=42)
    # Compound: epic + reharmonized (maximum tension)
    remix_to_file(src, _out(song, "07_epic_reharmonized"),
        EPIC, REHARMONIZED, seed=42)
    # Compound: ambient minor (a quiet, melancholic version of the boss theme)
    remix_to_file(src, _out(song, "08_lullaby"),
        RELATIVE_MINOR, AMBIENT, seed=7)


# ── Main ──────────────────────────────────────────────────────────────────────

DEMO_FNS = {
    "grape-garden":     demo_grape_garden,
    "vegetable-valley": demo_vegetable_valley,
    "rainbow-resort":   demo_rainbow_resort,
    "final-boss":       demo_final_boss,
}


def main():
    args = sys.argv[1:]

    if "--list" in args:
        print("\nAvailable songs:")
        for name in SONGS:
            path = SONGS[name]
            exists = "✓" if path.exists() else "✗ (missing)"
            print(f"  {name:<30} {exists}")
        print("\nAvailable presets:")
        from remixer.presets import PRESETS
        for name in PRESETS:
            fn = PRESETS[name]
            doc = (fn.__doc__ or "").strip().split("\n")[0]
            print(f"  {name:<20} {doc}")
        return

    # Select songs to run
    if args:
        selected = {}
        for arg in args:
            # Accept partial match (e.g. "grape" matches "grape-garden")
            matches = [k for k in SONGS if arg.lower() in k]
            if not matches:
                print(f"  Unknown song '{arg}'. Use --list to see options.")
                continue
            for m in matches:
                selected[m] = SONGS[m]
    else:
        selected = SONGS

    print()
    print("=" * 66)
    print("  KIRBY MUSIC REMIXER")
    print("=" * 66)
    print(f"\n  Output directory: {OUT_DIR}")
    print()

    for song_name, src_path in selected.items():
        if not src_path.exists():
            print(f"  [SKIP] {song_name}: file not found at {src_path}")
            continue

        # Analyse once and print description
        print(f"\n  Analysing {song_name}...")
        try:
            analysis = analyze(src_path)
        except Exception as exc:
            print(f"  [ERROR] Analysis failed: {exc}")
            continue

        print(describe(analysis))

        # Run demo suite
        try:
            DEMO_FNS[song_name](analysis)
        except Exception as exc:
            import traceback
            print(f"  [ERROR] Demo failed: {exc}")
            traceback.print_exc()

    print()
    print("=" * 66)
    print("  Done.")
    print("=" * 66)
    print()


if __name__ == "__main__":
    main()
