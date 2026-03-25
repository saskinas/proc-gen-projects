"""
remixer — Analyze MIDI, transform its symbolic IR, re-generate music.

Public API
----------
    analyze(source)                 → MusicalAnalysis
    describe(analysis)              → str   (formatted human-readable report)
    remix(source, *transforms, ...) → bytes (MIDI)
    remix_to_file(source, out, ...)         (writes .mid or .wav depending on ext)
    remix_to_wav(source, out, ...)          (always writes .wav)

    from remixer.presets import JAZZ, DARK_MINOR, BAROQUE, EPIC, AMBIENT, MIRROR_WORLD

Example
-------
    from remixer import remix_to_wav
    from remixer.presets import JAZZ

    remix_to_wav("grape-garden.mid", "grape-garden-jazz.wav", JAZZ, seed=42)
"""

from remixer.pipeline import analyze, describe, remix, remix_to_file, remix_to_wav

__all__ = ["analyze", "describe", "remix", "remix_to_file", "remix_to_wav"]
