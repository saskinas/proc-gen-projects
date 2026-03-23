"""
remixer — Analyze MIDI, transform its symbolic IR, re-generate music.

Public API
----------
    analyze(source)                 → MusicalAnalysis
    describe(analysis)              → str   (formatted human-readable report)
    remix(source, *transforms, ...) → bytes (MIDI)
    remix_to_file(source, out, ...)         (writes file, prints summary)

    from remixer.presets import JAZZ, DARK_MINOR, BAROQUE, EPIC, AMBIENT, MIRROR_WORLD

Example
-------
    from remixer import remix
    from remixer.presets import JAZZ, DARK_MINOR

    midi_bytes = remix("grape-garden.mid", JAZZ, seed=42)
    open("remix.mid", "wb").write(midi_bytes)
"""

from remixer.pipeline import analyze, describe, remix, remix_to_file

__all__ = ["analyze", "describe", "remix", "remix_to_file"]
