"""
remixer test suite.

Tests cover the full pipeline (analyze → transform → generate → MIDI) and
each major preset using the audio_listener's built-in test MIDI so no external
files are needed.
"""

from __future__ import annotations

import pathlib
import struct
import sys
import unittest

# ── Path bootstrap ─────────────────────────────────────────────────────────────
HERE     = pathlib.Path(__file__).resolve().parent
PROJECTS = HERE.parent
for pkg in ("remixer", "audio_listener", "music_generator", "procgen"):
    sys.path.insert(0, str(PROJECTS / pkg))


# ── Shared test MIDI (C major, 4/4, 120 BPM) ──────────────────────────────────

def _make_test_midi() -> bytes:
    """Minimal two-channel MIDI in C major for deterministic testing."""
    tpb  = 480
    uspb = 500_000  # 120 BPM

    def varlen(v: int) -> bytes:
        result = [v & 0x7F]; v >>= 7
        while v:
            result.insert(0, (v & 0x7F) | 0x80); v >>= 7
        return bytes(result)

    melody_pitches = [60, 62, 64, 65, 67, 69, 67, 65, 64, 62, 60, 60,
                      64, 67, 69, 71, 72, 71, 69, 67, 65, 64, 62, 60,
                      60, 62, 64, 65, 67, 65, 64, 62, 60, 62, 64, 65,
                      67, 69, 71, 72, 71, 69, 67, 65, 64, 62, 60, 60]
    bass_pitches   = [48, 48, 43, 43, 48, 48, 53, 53,
                      48, 48, 43, 43, 48, 48, 55, 55,
                      48, 48, 43, 43, 41, 41, 43, 43,
                      48, 48, 43, 43, 48, 48, 48, 48]

    mel_track = bytearray()
    mel_track += varlen(0) + bytes([0xFF, 0x03, len(b"Melody")]) + b"Melody"
    prev = 0
    for i, p in enumerate(melody_pitches):
        abs_t = i * tpb
        mel_track += varlen(abs_t - prev) + bytes([0x90, p, 80])
        mel_track += varlen(tpb - 1)      + bytes([0x80, p, 0])
        prev = abs_t + tpb
    mel_track += varlen(0) + b"\xFF\x2F\x00"

    bass_track = bytearray()
    bass_track += varlen(0) + bytes([0xFF, 0x03, len(b"Triangle")]) + b"Triangle"
    prev = 0
    for i, p in enumerate(bass_pitches):
        abs_t = i * tpb * 2
        bass_track += varlen(abs_t - prev) + bytes([0x91, p, 70])
        bass_track += varlen(tpb * 2 - 1)  + bytes([0x81, p, 0])
        prev = abs_t + tpb * 2
    bass_track += varlen(0) + b"\xFF\x2F\x00"

    tempo_track = bytearray()
    tempo_track += varlen(0) + b"\xFF\x51\x03"
    tempo_track += bytes([(uspb >> 16) & 0xFF, (uspb >> 8) & 0xFF, uspb & 0xFF])
    tempo_track += varlen(0) + b"\xFF\x58\x04" + bytes([4, 2, 24, 8])
    tempo_track += varlen(0) + b"\xFF\x2F\x00"

    def chunk(tag, data):
        return tag + struct.pack(">I", len(data)) + data

    header = struct.pack(">4sIHHH", b"MThd", 6, 1, 3, tpb)
    body   = (chunk(b"MTrk", bytes(tempo_track)) +
              chunk(b"MTrk", bytes(mel_track)) +
              chunk(b"MTrk", bytes(bass_track)))
    return header + body


TEST_MIDI = _make_test_midi()


# ── Pipeline tests ─────────────────────────────────────────────────────────────

class TestAnalyze(unittest.TestCase):
    def setUp(self):
        from remixer import analyze
        self.analysis = analyze(TEST_MIDI)

    def test_returns_musical_analysis(self):
        from music_generator.ir import MusicalAnalysis
        self.assertIsInstance(self.analysis, MusicalAnalysis)

    def test_key_is_string(self):
        self.assertIsInstance(self.analysis.key, str)
        self.assertGreater(len(self.analysis.key), 0)

    def test_tempo_positive(self):
        self.assertGreater(self.analysis.tempo_bpm, 0)

    def test_sections_nonempty(self):
        self.assertGreater(len(self.analysis.sections), 0)

    def test_time_sig_valid(self):
        ts = self.analysis.time_signature
        self.assertEqual(len(ts), 2)
        self.assertIn(ts[0], (2, 3, 4, 6, 12))


class TestDescribe(unittest.TestCase):
    def setUp(self):
        from remixer import analyze, describe
        analysis = analyze(TEST_MIDI)
        self.report = describe(analysis)

    def test_returns_string(self):
        self.assertIsInstance(self.report, str)

    def test_contains_key_info(self):
        self.assertIn("Key / Mode", self.report)

    def test_contains_sections(self):
        self.assertIn("SECTIONS", self.report)

    def test_contains_tempo(self):
        self.assertIn("BPM", self.report)


class TestRemix(unittest.TestCase):
    def _remix(self, *presets, seed=42):
        from remixer import remix
        return remix(TEST_MIDI, *presets, seed=seed)

    def test_baseline_returns_bytes(self):
        result = self._remix()
        self.assertIsInstance(result, bytes)

    def test_baseline_is_valid_midi(self):
        result = self._remix()
        self.assertEqual(result[:4], b"MThd")

    def test_different_seeds_differ(self):
        # Replay mode is deterministic regardless of seed; use a generative
        # preset (JAZZ) which forces generation_mode="generate" so the seed
        # actually varies the output.
        from remixer.presets import JAZZ
        r1 = self._remix(JAZZ, seed=1)
        r2 = self._remix(JAZZ, seed=99)
        self.assertNotEqual(r1, r2)

    def test_same_seed_deterministic(self):
        r1 = self._remix(seed=42)
        r2 = self._remix(seed=42)
        self.assertEqual(r1, r2)


# ── Preset tests ───────────────────────────────────────────────────────────────

class TestPresetsImport(unittest.TestCase):
    def test_presets_dict_complete(self):
        from remixer.presets import PRESETS
        expected = {
            "jazz", "baroque", "classical", "folk", "ambient", "rock",
            "dark_minor", "relative_minor", "epic",
            "mirror_world", "retrograde", "kaleidoscope",
            "fragmented", "reharmonized",
        }
        self.assertEqual(set(PRESETS.keys()), expected)

    def test_each_preset_is_callable(self):
        from remixer.presets import PRESETS
        for name, fn in PRESETS.items():
            with self.subTest(preset=name):
                self.assertTrue(callable(fn))


class TestPresetPipeline(unittest.TestCase):
    """Each preset must produce valid MIDI without raising."""

    def _run(self, preset_name: str):
        from remixer import remix, analyze
        from remixer.presets import PRESETS
        fn = PRESETS[preset_name]
        result = remix(TEST_MIDI, fn, seed=7)
        self.assertIsInstance(result, bytes)
        self.assertEqual(result[:4], b"MThd", f"Preset {preset_name} produced invalid MIDI")
        return result

    def test_jazz(self):             self._run("jazz")
    def test_baroque(self):          self._run("baroque")
    def test_classical(self):        self._run("classical")
    def test_folk(self):             self._run("folk")
    def test_ambient(self):          self._run("ambient")
    def test_rock(self):             self._run("rock")
    def test_dark_minor(self):       self._run("dark_minor")
    def test_relative_minor(self):   self._run("relative_minor")
    def test_epic(self):             self._run("epic")
    def test_mirror_world(self):     self._run("mirror_world")
    def test_retrograde(self):       self._run("retrograde")
    def test_kaleidoscope(self):     self._run("kaleidoscope")
    def test_fragmented(self):       self._run("fragmented")
    def test_reharmonized(self):     self._run("reharmonized")


class TestPresetTransforms(unittest.TestCase):
    """Verify that preset transforms produce the expected IR-level changes."""

    def setUp(self):
        from remixer import analyze
        self.analysis = analyze(TEST_MIDI)

    def test_dark_minor_changes_mode(self):
        from remixer.presets import DARK_MINOR
        remixed = DARK_MINOR(self.analysis)
        self.assertEqual(remixed.mode, "harmonic_minor")

    def test_relative_minor_changes_mode_and_key(self):
        from remixer.presets import RELATIVE_MINOR
        original_pc = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
        remixed = RELATIVE_MINOR(self.analysis)
        self.assertEqual(remixed.mode, "minor")
        # Key should be 3 semitones below original
        from music_generator import ROOT_TO_PC, NOTE_NAMES
        orig_pc = ROOT_TO_PC.get(self.analysis.key, 0)
        new_pc  = ROOT_TO_PC.get(remixed.key, 0)
        self.assertEqual((new_pc - orig_pc) % 12, 9)  # +9 == -3 mod 12

    def test_dark_minor_key_unchanged(self):
        from remixer.presets import DARK_MINOR
        remixed = DARK_MINOR(self.analysis)
        self.assertEqual(remixed.key, self.analysis.key)

    def test_ambient_slows_tempo(self):
        from remixer.presets import AMBIENT
        remixed = AMBIENT(self.analysis)
        self.assertLess(remixed.tempo_bpm, self.analysis.tempo_bpm)

    def test_rock_speeds_tempo(self):
        from remixer.presets import ROCK
        remixed = ROCK(self.analysis)
        self.assertGreater(remixed.tempo_bpm, self.analysis.tempo_bpm)

    def test_jazz_slows_tempo(self):
        from remixer.presets import JAZZ
        remixed = JAZZ(self.analysis)
        self.assertLess(remixed.tempo_bpm, self.analysis.tempo_bpm)

    def test_epic_raises_energy(self):
        from remixer.presets import EPIC
        remixed = EPIC(self.analysis)
        orig_levels  = [s.energy.level for s in self.analysis.sections]
        remix_levels = [s.energy.level for s in remixed.sections]
        self.assertGreater(sum(remix_levels), sum(orig_levels))

    def test_mirror_world_generation_mode(self):
        from remixer.presets import MIRROR_WORLD
        remixed = MIRROR_WORLD(self.analysis)
        for sec in remixed.sections:
            if "soprano" in sec.voice_sequences:
                self.assertEqual(sec.generation_mode, "replay")

    def test_preset_does_not_mutate_input(self):
        """All presets must be pure functions — input is unchanged."""
        from remixer.presets import PRESETS
        import copy
        for name, fn in PRESETS.items():
            with self.subTest(preset=name):
                original_key  = self.analysis.key
                original_mode = self.analysis.mode
                original_bpm  = self.analysis.tempo_bpm
                _ = fn(self.analysis)
                self.assertEqual(self.analysis.key,       original_key)
                self.assertEqual(self.analysis.mode,      original_mode)
                self.assertEqual(self.analysis.tempo_bpm, original_bpm)


class TestChainedPresets(unittest.TestCase):
    """Compound remixes: two presets applied in sequence."""

    def _remix(self, *presets):
        from remixer import remix
        result = remix(TEST_MIDI, *presets, seed=42)
        self.assertEqual(result[:4], b"MThd")
        return result

    def test_dark_jazz(self):
        from remixer.presets import DARK_MINOR, JAZZ
        self._remix(DARK_MINOR, JAZZ)

    def test_epic_rock(self):
        from remixer.presets import EPIC, ROCK
        self._remix(EPIC, ROCK)

    def test_baroque_mirror(self):
        from remixer.presets import MIRROR_WORLD, BAROQUE
        self._remix(MIRROR_WORLD, BAROQUE)

    def test_ambient_minor(self):
        from remixer.presets import RELATIVE_MINOR, AMBIENT
        self._remix(RELATIVE_MINOR, AMBIENT)


class TestRemixToFile(unittest.TestCase):
    def test_writes_file(self):
        import tempfile, os
        from remixer import remix_to_file
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "test_out.mid"
            remix_to_file(TEST_MIDI, out, verbose=False)
            self.assertTrue(out.exists())
            data = out.read_bytes()
            self.assertEqual(data[:4], b"MThd")

    def test_creates_parent_dirs(self):
        import tempfile
        from remixer import remix_to_file
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "nested" / "dir" / "out.mid"
            remix_to_file(TEST_MIDI, out, verbose=False)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
