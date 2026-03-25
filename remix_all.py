"""
remix_all.py — Batch remix all MIDI files under sample_input/.

Discovers every MIDI in audio_listener/sample_input/{album}/*.mid,
checks reproduction fidelity, and generates a wide variety of remixes.

Output structure
----------------
    remix_output/
        {album}/
            {song_stem}/
                00_original.mid          ← baseline reproduction
                01_developed.mid
                02_embellished.mid
                ...

Quality check
-------------
For each song, the baseline reproduction is scored against the original.
Songs that score < 50 % are flagged and skipped (remix output unreliable).
The report is printed and also written to remix_output/quality_report.txt.

Usage
-----
    python remix_all.py                    # process everything (WAV output)
    python remix_all.py --midi             # output MIDI instead of WAV
    python remix_all.py --check-only       # quality check, no remixes
    python remix_all.py Kirby              # only albums matching 'Kirby'
    python remix_all.py grape              # only songs matching 'grape'
    python remix_all.py --workers 4        # parallel (default: 1)
    python remix_all.py --skip jazz,folk   # skip these preset names
    python remix_all.py --only lofi,epic   # only run these preset names
"""

from __future__ import annotations

import sys
import pathlib
import time
import traceback
import concurrent.futures
from typing import Callable

# ── Path bootstrap ─────────────────────────────────────────────────────────────
HERE     = pathlib.Path(__file__).resolve().parent
for pkg in ("remixer", "audio_listener", "music_generator", "procgen"):
    sys.path.insert(0, str(HERE / pkg))

from remixer import analyze, describe, remix_to_file
from remixer.presets import PRESETS

# ── Paths ──────────────────────────────────────────────────────────────────────
SAMPLE_DIR = HERE / "audio_listener" / "sample_input"
OUT_DIR    = HERE / "remix_output"

# ── Preset plan ───────────────────────────────────────────────────────────────
# Each entry: (filename_stem, [preset_callable, ...])
# Presets are applied left-to-right (chained transforms).
# Seeds: always 42 for consistency, 7 for alternate-mood variants.

_P = PRESETS   # short alias

REMIX_PLAN: list[tuple[str, list[Callable]]] = [
    # ── Baseline ──────────────────────────────────────────────────────────────
    ("00_original",             []),

    # ── Melodic variation (new) — Suno-focused ────────────────────────────────
    ("01_developed",            [_P["developed"]]),
    ("02_embellished",          [_P["embellished"]]),
    ("03_reimagined",           [_P["reimagined"]]),
    ("04_evolved",              [_P["evolved"]]),
    ("05_grooved",              [_P["grooved"]]),

    # ── Tonal colour ──────────────────────────────────────────────────────────
    ("06_dark_minor",           [_P["dark_minor"]]),
    ("07_relative_minor",       [_P["relative_minor"]]),

    # ── Style transplants ─────────────────────────────────────────────────────
    ("08_jazz",                 [_P["jazz"]]),
    ("09_baroque",              [_P["baroque"]]),
    ("10_classical",            [_P["classical"]]),
    ("11_folk",                 [_P["folk"]]),
    ("12_lofi",                 [_P["lofi"]]),
    ("13_minimalist",           [_P["minimalist"]]),
    ("14_ambient",              [_P["ambient"]]),
    ("15_rock",                 [_P["rock"]]),

    # ── Energy ────────────────────────────────────────────────────────────────
    ("16_epic",                 [_P["epic"]]),

    # ── Voice / melodic transforms ────────────────────────────────────────────
    ("17_mirror_world",         [_P["mirror_world"]]),
    ("18_retrograde",           [_P["retrograde"]]),
    ("19_kaleidoscope",         [_P["kaleidoscope"]]),

    # ── Motivic development ───────────────────────────────────────────────────
    ("20_fragmented",           [_P["fragmented"]]),
    ("21_reharmonized",         [_P["reharmonized"]]),

    # ── Compound: melodic variation + style ───────────────────────────────────
    ("22_developed_jazz",       [_P["developed"],    _P["jazz"]]),
    ("23_embellished_baroque",  [_P["embellished"],  _P["baroque"]]),
    ("24_reimagined_folk",      [_P["reimagined"],   _P["folk"]]),
    ("25_evolved_lofi",         [_P["evolved"],      _P["lofi"]]),
    ("26_grooved_rock",         [_P["grooved"],      _P["rock"]]),

    # ── Compound: melodic variation + tonal ───────────────────────────────────
    ("27_dark_developed",       [_P["dark_minor"],   _P["developed"]]),
    ("28_minor_reimagined",     [_P["relative_minor"], _P["reimagined"]]),
    ("29_dark_embellished",     [_P["dark_minor"],   _P["embellished"]]),

    # ── Compound: classic combinations ────────────────────────────────────────
    ("30_dark_lofi",            [_P["dark_minor"],   _P["lofi"]]),
    ("31_epic_rock",            [_P["epic"],         _P["rock"]]),
    ("32_baroque_mirror",       [_P["mirror_world"], _P["baroque"]]),
    ("33_minor_ambient",        [_P["relative_minor"], _P["ambient"]]),
    ("34_dark_jazz",            [_P["dark_minor"],   _P["jazz"]]),
    ("35_epic_reharmonized",    [_P["epic"],         _P["reharmonized"]]),
]


# ── Quality check ──────────────────────────────────────────────────────────────

def _parse_midi_tracks(midi_bytes: bytes):
    """
    Generator yielding (channel, pitch, velocity) for every note-on with vel>0.
    Handles MIDI running status correctly.
    """
    import struct

    def _read_vlq(data: bytes, pos: int) -> tuple[int, int]:
        val = 0
        while True:
            b = data[pos]; pos += 1
            val = (val << 7) | (b & 0x7F)
            if not (b & 0x80):
                return val, pos

    pos = 14  # skip 14-byte MIDI header chunk
    while pos < len(midi_bytes) - 8:
        if midi_bytes[pos:pos+4] != b'MTrk':
            break
        track_len = struct.unpack('>I', midi_bytes[pos+4:pos+8])[0]
        track_end = pos + 8 + track_len
        pos += 8
        prev_status = 0
        while pos < track_end:
            _, pos = _read_vlq(midi_bytes, pos)   # delta time
            if pos >= track_end:
                break
            b = midi_bytes[pos]
            if b & 0x80:
                status = b; pos += 1; prev_status = status
            else:
                status = prev_status  # running status — data byte, don't advance
            nib = status & 0xF0
            ch  = status & 0x0F
            if nib == 0x90 and pos + 1 < track_end:
                pitch = midi_bytes[pos]; vel = midi_bytes[pos+1]; pos += 2
                if vel > 0:
                    yield ch, pitch, vel
            elif nib == 0x80 and pos + 1 < track_end:
                pos += 2   # note-off
            elif nib in (0xA0, 0xB0, 0xE0) and pos + 1 < track_end:
                pos += 2
            elif nib in (0xC0, 0xD0) and pos < track_end:
                pos += 1
            elif status == 0xFF and pos < track_end:
                pos += 1   # meta type byte
                meta_len, pos = _read_vlq(midi_bytes, pos)
                pos += meta_len
            elif status in (0xF0, 0xF7):
                sl, pos = _read_vlq(midi_bytes, pos)
                pos += sl
            else:
                break       # truly unknown — give up on this track
        pos = track_end


def _midi_pitch_histogram(midi_bytes: bytes) -> list[float]:
    """Return a 12-bin pitch-class histogram (normalised) from raw MIDI bytes.
    Channel 9 (GM drums) is excluded so the histogram reflects pitched content."""
    try:
        bins = [0] * 12
        for ch, pitch, _vel in _parse_midi_tracks(midi_bytes):
            if ch != 9:
                bins[pitch % 12] += 1
        total = sum(bins) or 1
        return [b / total for b in bins]
    except Exception:
        return [1/12] * 12


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot  = sum(x * y for x, y in zip(a, b))
    na   = sum(x * x for x in a) ** 0.5
    nb   = sum(x * x for x in b) ** 0.5
    return dot / (na * nb + 1e-9)


def _count_notes(midi_bytes: bytes) -> int:
    """Count note-on events with velocity > 0 across all channels."""
    try:
        return sum(1 for _ in _parse_midi_tracks(midi_bytes))
    except Exception:
        return 0


def check_quality(src_path: pathlib.Path, analysis=None) -> dict:
    """
    Generate a baseline reproduction and score it against the original.

    Uses the analysis IR (if supplied) to count original notes reliably —
    our simple MIDI byte-parser can fail on non-standard SysEx-heavy files.

    Returns dict with keys: score (0-1), note_ratio, histogram_sim, status.
    """
    try:
        orig_bytes = src_path.read_bytes()
        from remixer import remix
        repro_bytes = remix(src_path)

        # Note count: prefer analysis IR (accurate) over byte parser
        if analysis is not None:
            orig_notes = sum(
                sum(1 for ev in seq if ev.get("degree") != "rest")
                for sec in analysis.sections
                for seq in sec.voice_sequences.values()
            )
        else:
            orig_notes = _count_notes(orig_bytes)

        repro_notes = _count_notes(repro_bytes)

        # If our byte parser returned 0 for the original but analysis has notes,
        # fall back to comparing repro vs analysis note count.
        parser_failed_orig = (_count_notes(orig_bytes) == 0 and orig_notes > 0)

        if parser_failed_orig:
            # Can't compare histograms reliably — compare note counts only
            note_ratio = min(repro_notes, orig_notes) / max(orig_notes, 1)
            hist_sim   = note_ratio  # treat same as note ratio (best we can do)
        else:
            note_ratio = min(repro_notes, orig_notes) / max(orig_notes, 1)
            orig_hist  = _midi_pitch_histogram(orig_bytes)
            repro_hist = _midi_pitch_histogram(repro_bytes)
            hist_sim   = _cosine_similarity(orig_hist, repro_hist)

        # Combined score: 60% note coverage + 40% pitch-class similarity
        score = 0.60 * note_ratio + 0.40 * hist_sim

        return {
            "score":         score,
            "note_ratio":    note_ratio,
            "hist_sim":      hist_sim,
            "orig_notes":    orig_notes,
            "repro_notes":   repro_notes,
            "parser_failed": parser_failed_orig,
            "status":        "ok" if score >= 0.50 else "poor",
        }
    except Exception as e:
        return {
            "score": 0.0, "note_ratio": 0.0, "hist_sim": 0.0,
            "orig_notes": 0, "repro_notes": 0, "parser_failed": False,
            "status": f"error: {e}",
        }


# ── Remix worker ───────────────────────────────────────────────────────────────

def _run_remix(task: tuple) -> tuple[str, str, bool, str]:
    """(src_path, out_path, stem, transforms) → (stem, summary_line, ok, error)"""
    src_path, out_path, stem, transforms = task
    try:
        remix_to_file(src_path, out_path, *transforms, seed=42)
        size = out_path.stat().st_size
        return stem, f"  [{stem}]  {size//1024} KB", True, ""
    except Exception as e:
        return stem, f"  [{stem}]  FAILED: {e}", False, traceback.format_exc()


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover_songs(filter_terms: list[str]) -> list[tuple[str, str, pathlib.Path]]:
    """
    Return list of (album_name, song_stem, midi_path) for all MIDIs under SAMPLE_DIR.
    Filters by any of filter_terms (case-insensitive substring match on album or stem).
    """
    results = []
    for midi_path in sorted(SAMPLE_DIR.rglob("*.mid")):
        album = midi_path.parent.name
        stem  = midi_path.stem
        if filter_terms:
            combined = (album + "/" + stem).lower()
            if not any(t.lower() in combined for t in filter_terms):
                continue
        results.append((album, stem, midi_path))
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    check_only = "--check-only" in args
    use_midi   = "--midi" in args
    args = [a for a in args if a not in ("--check-only", "--midi")]

    workers = 1
    skip_presets: set[str] = set()
    only_presets: set[str] = set()
    i = 0
    while i < len(args):
        if args[i] == "--workers" and i + 1 < len(args):
            workers = int(args[i + 1])
            args = args[:i] + args[i+2:]
        elif args[i] == "--skip" and i + 1 < len(args):
            skip_presets = {s.strip() for s in args[i + 1].split(",")}
            args = args[:i] + args[i+2:]
        elif args[i] == "--only" and i + 1 < len(args):
            only_presets = {s.strip() for s in args[i + 1].split(",")}
            args = args[:i] + args[i+2:]
        else:
            i += 1

    out_ext = ".mid" if use_midi else ".wav"

    # Remaining args are filter terms (album name or song name substrings)
    filter_terms = [a for a in args if not a.startswith("--")]

    songs = discover_songs(filter_terms)
    if not songs:
        print(f"No MIDI files found under {SAMPLE_DIR}")
        if filter_terms:
            print(f"  Filters applied: {filter_terms}")
        return

    # ── Filter remix plan by --skip / --only ──────────────────────────────────
    def _remix_uses_preset(remix_stem: str, transforms: list, name: str) -> bool:
        """Check if a remix entry uses a given preset name (in stem or transform)."""
        if name in remix_stem:
            return True
        for t in transforms:
            if hasattr(t, "__name__") and t.__name__ == name:
                return True
        return False

    if only_presets:
        active_plan = [
            (stem, tfs) for stem, tfs in REMIX_PLAN
            if stem == "00_original"  # always keep baseline
            or any(_remix_uses_preset(stem, tfs, p) for p in only_presets)
        ]
    elif skip_presets:
        active_plan = [
            (stem, tfs) for stem, tfs in REMIX_PLAN
            if not any(_remix_uses_preset(stem, tfs, p) for p in skip_presets)
        ]
    else:
        active_plan = list(REMIX_PLAN)

    print()
    print("=" * 70)
    print("  BATCH REMIXER")
    print("=" * 70)
    print(f"  Source :  {SAMPLE_DIR}")
    print(f"  Output :  {OUT_DIR}")
    print(f"  Format :  {'MIDI' if use_midi else 'WAV'}")
    print(f"  Songs  :  {len(songs)}")
    print(f"  Remixes:  {len(active_plan)} per song  ({len(songs)*len(active_plan)} total)")
    print(f"  Workers:  {workers}")
    if skip_presets:
        print(f"  Skipped:  {', '.join(sorted(skip_presets))}")
    if only_presets:
        print(f"  Only   :  {', '.join(sorted(only_presets))}")
    print()

    quality_lines: list[str] = []
    skipped: set[str] = set()
    total_ok = total_fail = 0
    t0_all = time.time()

    for album, stem, src_path in songs:
        song_out = OUT_DIR / album / stem
        song_out.mkdir(parents=True, exist_ok=True)

        # ── Analyse ────────────────────────────────────────────────────────────
        print(f"  Analysing {album}/{stem} ...", end=" ", flush=True)
        try:
            analysis = analyze(src_path)
        except Exception as exc:
            print(f"ANALYSIS FAILED: {exc}")
            q_line = f"  {album}/{stem:<40}  score=  0%  ANALYSIS FAILED: {exc}"
            quality_lines.append(q_line)
            skipped.add(f"{album}/{stem}")
            continue

        # ── Quality check ──────────────────────────────────────────────────────
        qr = check_quality(src_path, analysis)
        pct = int(qr["score"] * 100)
        pf  = " [~hist]" if qr.get("parser_failed") else ""
        flag = "" if qr["status"] == "ok" else "  *** POOR ***"
        q_line = (
            f"  {album}/{stem:<40}  "
            f"score={pct:3d}%{pf}  "
            f"notes={qr['orig_notes']}->{qr['repro_notes']}  "
            f"hist={qr['hist_sim']:.2f}"
            f"{flag}"
        )
        quality_lines.append(q_line)
        print(q_line.strip())

        if qr["status"] != "ok":
            skipped.add(f"{album}/{stem}")
            if check_only:
                continue

        if check_only:
            continue

        # ── Generate remixes ───────────────────────────────────────────────────
        tasks = []
        for remix_stem, transforms in active_plan:
            out_path = song_out / f"{remix_stem}{out_ext}"
            tasks.append((src_path, out_path, remix_stem, transforms))

        print(f"  Remixing  {album}/{stem}  ({len(tasks)} variants)...")
        t0 = time.time()

        if workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(_run_remix, t): t[2] for t in tasks}
                for fut in concurrent.futures.as_completed(futures):
                    rstem, line, ok, err = fut.result()
                    if not ok:
                        print(f"    FAIL {rstem}: {err.splitlines()[-1]}")
                        total_fail += 1
                    else:
                        total_ok += 1
        else:
            for task in tasks:
                rstem, line, ok, err = _run_remix(task)
                if not ok:
                    print(f"    FAIL  {rstem}: {err.splitlines()[-1]}")
                    total_fail += 1
                else:
                    total_ok += 1

        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s  -> {song_out}")

    # ── Quality report ────────────────────────────────────────────────────────
    report_path = OUT_DIR / "quality_report.txt"
    report_path.write_text(
        "REPRODUCTION QUALITY REPORT\n"
        "============================\n\n"
        + "\n".join(quality_lines)
        + "\n\nLegend: score = 60%*note_coverage + 40%*pitch_histogram_similarity\n"
        + "        *** POOR *** = score < 45% — remixes may not reflect source well\n",
        encoding="utf-8",
    )

    elapsed_all = time.time() - t0_all
    print()
    print("=" * 70)
    print(f"  Total time : {elapsed_all:.1f}s")
    if not check_only:
        print(f"  Remixes OK : {total_ok}")
        print(f"  Failures   : {total_fail}")
    if skipped:
        print(f"  Poor quality ({len(skipped)}) — check {report_path.name} for details:")
        for s in skipped:
            print(f"    {s}")
    print(f"  Quality report: {report_path}")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
