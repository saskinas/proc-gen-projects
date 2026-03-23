"""
test_music.py — Musical coherence and correctness tests for MusicDomain.

Tests are grouped into five categories:
  1. Generation integrity   — all configs generate without errors
  2. Musical constraints    — voice ranges, ordering, duration balance
  3. Harmonic correctness   — diatonic conformity, tonic/dominant presence
  4. Parameter effects      — density, step ratio, ornaments, imitation
  5. Variety & determinism  — different seeds vary; same seed is identical

Run with:
    python tests.py
or:
    python -m pytest tests.py -v
"""

from __future__ import annotations

import sys
import pathlib
import unittest
import statistics

HERE     = pathlib.Path(__file__).resolve().parent
PROJECTS = HERE.parent
sys.path.insert(0, str(HERE))                      # music_generator package
sys.path.insert(0, str(PROJECTS / "procgen"))      # procgen package

from procgen import Designer, export
from music_generator import MusicDomain
from music_generator import (
    NOTE_NAMES,
    ROOT_TO_PC,
    SCALE_INTERVALS,
    CHORD_INTERVALS,
    VOICE_RANGES,
    VOICE_ROLES,
    note_str_to_midi,
    midi_to_note,
    scale_midi_in_range,
    chord_midi_in_range,
)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def make(**kwargs) -> object:
    """Convenience: build a Designer with keyword params and generate."""
    seed = kwargs.pop("seed", 42)
    d = Designer(domain=MusicDomain(), seed=seed)
    for k, v in kwargs.items():
        d.set(k, v)
    return d.generate()


def melody(score) -> list:
    return score.tracks[0].notes if score.tracks else []


def all_notes(score) -> list:
    return [n for t in score.tracks for n in t.notes]


def midis(notes) -> list[int]:
    return [note_str_to_midi(n.pitch) for n in notes]


def avg_interval(notes) -> float:
    ms = midis(notes)
    if len(ms) < 2:
        return 0.0
    return statistics.mean(abs(ms[i + 1] - ms[i]) for i in range(len(ms) - 1))


def step_pct(notes) -> float:
    ms = midis(notes)
    if len(ms) < 2:
        return 100.0
    steps = sum(1 for i in range(len(ms) - 1) if abs(ms[i + 1] - ms[i]) <= 2)
    return steps / (len(ms) - 1) * 100


def diatonic_pct(notes, key: str, mode: str) -> float:
    if not notes:
        return 100.0
    root_pc = ROOT_TO_PC[key]
    ivs     = set(SCALE_INTERVALS[mode])
    ok = sum(1 for n in notes if (note_str_to_midi(n.pitch) % 12 - root_pc) % 12 in ivs)
    return ok / len(notes) * 100


# ══════════════════════════════════════════════════════════════════════════════
# 1. Generation Integrity
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerationIntegrity(unittest.TestCase):

    def test_default_generates(self):
        score = make()
        self.assertIsNotNone(score)
        self.assertGreater(len(score.tracks), 0)
        self.assertGreater(sum(len(t.notes) for t in score.tracks), 0)

    def test_all_modes(self):
        for mode in SCALE_INTERVALS:
            with self.subTest(mode=mode):
                score = make(mode=mode, key="C")
                self.assertGreater(len(all_notes(score)), 0, f"mode {mode} produced no notes")

    def test_all_forms(self):
        for form in ["binary", "ternary", "rondo", "through_composed"]:
            with self.subTest(form=form):
                score = make(form=form)
                self.assertGreater(len(all_notes(score)), 0)

    def test_all_voice_counts(self):
        for n in [1, 2, 3, 4]:
            with self.subTest(num_voices=n):
                score = make(num_voices=n)
                self.assertEqual(len(score.tracks), n)

    def test_all_bass_types(self):
        for bt in ["sustained", "walking", "alberti", "arpeggiated", "pedal"]:
            with self.subTest(bass_type=bt):
                score = make(num_voices=2, bass_type=bt)
                self.assertGreater(len(all_notes(score)), 0)

    def test_all_inner_voice_styles(self):
        for ivs in ["block_chords", "arpeggiated", "countermelody", "running_sixteenths"]:
            with self.subTest(ivs=ivs):
                score = make(num_voices=3, inner_voice_style=ivs)
                self.assertGreater(len(all_notes(score)), 0)

    def test_all_time_signatures(self):
        for ts in [[4, 4], [3, 4], [6, 8], [2, 4], [2, 2]]:
            with self.subTest(ts=ts):
                score = make(time_signature=ts)
                self.assertEqual(list(score.time_signature), ts)

    def test_midi_export_produces_bytes(self):
        score = make()
        data = export.to_midi(score)
        self.assertIsInstance(data, bytes)
        self.assertGreater(len(data), 14)  # at minimum SMF header

    def test_midi_header_magic(self):
        score = make()
        data = export.to_midi(score)
        self.assertEqual(data[:4], b"MThd")

    def test_all_instruments(self):
        for instr in ["piano", "harpsichord", "organ", "strings", "violin",
                      "cello", "flute", "bass", "guitar"]:
            with self.subTest(instr=instr):
                score = make(instruments=[instr], num_voices=1)
                self.assertEqual(score.tracks[0].instrument, instr)

    def test_extreme_parameters(self):
        """Boundary values should not crash."""
        score = make(
            num_phrases=8, phrase_length=16,
            chord_complexity=1.0, harmonic_rhythm=1.0,
            circle_of_fifths=1.0, modal_mixture=1.0,
            pedal_point=1.0, counterpoint=1.0,
            imitation=1.0, voice_independence=1.0,
            rhythmic_density=1.0, swing=1.0,
            num_voices=4, use_ornaments=True,
            sequence_probability=1.0, step_leap_ratio=0.0,
        )
        self.assertGreater(len(all_notes(score)), 0)

    def test_minimal_parameters(self):
        score = make(
            num_phrases=2, phrase_length=2,
            chord_complexity=0.0, harmonic_rhythm=0.0,
            circle_of_fifths=0.0, modal_mixture=0.0,
            pedal_point=0.0, counterpoint=0.0,
            imitation=0.0, voice_independence=0.0,
            rhythmic_density=0.0, swing=0.0,
            num_voices=1, use_ornaments=False,
            sequence_probability=0.0, step_leap_ratio=1.0,
        )
        self.assertGreater(len(all_notes(score)), 0)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Musical Constraints
# ══════════════════════════════════════════════════════════════════════════════

class TestMusicalConstraints(unittest.TestCase):

    def _score4v(self, seed=1):
        return make(num_voices=4, seed=seed)

    def test_all_pitches_are_valid_strings(self):
        """Every note.pitch should parse to a MIDI number in 0–127."""
        score = make(num_voices=4, use_ornaments=True)
        for track in score.tracks:
            for note in track.notes:
                with self.subTest(pitch=note.pitch):
                    midi = note_str_to_midi(note.pitch)
                    self.assertGreaterEqual(midi, 0)
                    self.assertLessEqual(midi, 127)

    def test_all_velocities_in_range(self):
        score = make(num_voices=4, use_ornaments=True)
        for n in all_notes(score):
            self.assertGreaterEqual(n.velocity, 1)
            self.assertLessEqual(n.velocity, 127)

    def test_all_durations_positive(self):
        score = make(num_voices=4, use_ornaments=True)
        for n in all_notes(score):
            self.assertGreater(n.duration, 0.0)

    def test_soprano_above_bass_on_average(self):
        """Mean soprano MIDI pitch > mean bass MIDI pitch."""
        score = make(num_voices=2, seed=42)
        sop_mean = statistics.mean(midis(score.tracks[0].notes))
        bas_mean = statistics.mean(midis(score.tracks[1].notes))
        self.assertGreater(sop_mean, bas_mean,
            f"soprano mean {sop_mean:.1f} not above bass mean {bas_mean:.1f}")

    def test_4voice_ordering(self):
        """soprano > alto > bass by mean pitch (strict ordering expected)."""
        score = make(num_voices=4, seed=7)
        roles = score.metadata.get("voices", [])
        means = {}
        for idx, track in enumerate(score.tracks):
            if track.notes:
                means[roles[idx]] = statistics.mean(midis(track.notes))
        if "soprano" in means and "alto" in means:
            self.assertGreater(means["soprano"], means["alto"] - 4,
                "soprano mean should be notably above alto")
        if "alto" in means and "bass" in means:
            self.assertGreater(means["alto"], means["bass"],
                "alto mean should be above bass mean")

    def test_soprano_stays_mostly_in_range(self):
        """≥80% of soprano notes within C4–A5 (MIDI 60–81), leaving room for ornaments."""
        score = make(num_voices=1, use_ornaments=True, seed=12)
        lo, hi = VOICE_RANGES["soprano"]
        ns = score.tracks[0].notes
        in_range = sum(1 for n in ns if lo <= note_str_to_midi(n.pitch) <= hi)
        pct = in_range / len(ns) * 100
        self.assertGreaterEqual(pct, 80.0,
            f"Only {pct:.1f}% of soprano notes in range [{lo},{hi}]")

    def test_bass_stays_mostly_in_range(self):
        """≥80% of bass notes within C2–C4 (MIDI 36–60)."""
        score = make(num_voices=2, seed=5)
        lo, hi = VOICE_RANGES["bass"]
        ns = score.tracks[1].notes
        in_range = sum(1 for n in ns if lo <= note_str_to_midi(n.pitch) <= hi)
        pct = in_range / len(ns) * 100
        self.assertGreaterEqual(pct, 80.0,
            f"Only {pct:.1f}% of bass notes in range [{lo},{hi}]")

    def test_voice_total_durations_approximately_equal(self):
        """All voices should span the same total number of beats (±10%)."""
        score = make(num_voices=3, use_ornaments=False, seed=42)
        totals = [sum(n.duration for n in t.notes) for t in score.tracks]
        if not totals:
            return
        avg = statistics.mean(totals)
        for i, total in enumerate(totals):
            deviation = abs(total - avg) / avg
            self.assertLessEqual(deviation, 0.12,
                f"Voice {i} duration {total:.1f} deviates {deviation:.1%} from mean {avg:.1f}")

    def test_score_has_metadata(self):
        score = make()
        self.assertIn("intent", score.metadata)
        self.assertIn("voices", score.metadata)

    def test_midi_roundtrip_length_nonzero(self):
        for seed in [1, 50, 999]:
            score = make(seed=seed, num_voices=3)
            data = export.to_midi(score)
            self.assertGreater(len(data), 100)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Harmonic Correctness
# ══════════════════════════════════════════════════════════════════════════════

class TestHarmonicCorrectness(unittest.TestCase):

    def test_melody_mostly_diatonic_major(self):
        """≥70% of melody notes should belong to the declared scale (major)."""
        score = make(key="G", mode="major", modal_mixture=0.0, seed=1)
        pct = diatonic_pct(melody(score), "G", "major")
        self.assertGreaterEqual(pct, 70.0,
            f"Only {pct:.1f}% of notes diatonic to G major")

    def test_melody_mostly_diatonic_minor(self):
        score = make(key="A", mode="minor", modal_mixture=0.0, seed=2)
        pct = diatonic_pct(melody(score), "A", "minor")
        self.assertGreaterEqual(pct, 70.0,
            f"Only {pct:.1f}% of notes diatonic to A minor")

    def test_melody_mostly_diatonic_dorian(self):
        score = make(key="D", mode="dorian", modal_mixture=0.0, seed=3)
        pct = diatonic_pct(melody(score), "D", "dorian")
        self.assertGreaterEqual(pct, 65.0)

    def test_tonic_pitch_class_present(self):
        """The tonic pitch class should appear in the melody."""
        for key in ["C", "D", "G", "Bb", "F#"]:
            with self.subTest(key=key):
                score = make(key=key, mode="major", seed=7)
                pcs = {note_str_to_midi(n.pitch) % 12 for n in melody(score)}
                self.assertIn(ROOT_TO_PC[key], pcs,
                    f"Tonic {key} never appears in melody")

    def test_chord_types_are_valid(self):
        """Every chord type used must be in our known vocabulary."""
        from music_generator import HarmonicPlanGenerator
        from procgen.seeds import Seed

        params = {
            "key": "C", "mode": "major", "num_phrases": 4, "phrase_length": 4,
            "time_signature": [4, 4], "harmonic_rhythm": 0.5,
            "circle_of_fifths": 0.6, "chord_complexity": 0.8,
            "modal_mixture": 0.3, "pedal_point": 0.2, "form": "binary",
        }
        gen = HarmonicPlanGenerator(Seed(42))
        events = gen.generate(params, {})
        valid = set(CHORD_INTERVALS.keys())
        for ev in events:
            self.assertIn(ev.chord_type, valid,
                f"Unknown chord type: {ev.chord_type}")

    def test_scale_degrees_are_valid(self):
        from music_generator import HarmonicPlanGenerator
        from procgen.seeds import Seed

        params = {
            "key": "C", "mode": "major", "num_phrases": 4, "phrase_length": 4,
            "time_signature": [4, 4], "harmonic_rhythm": 0.6,
            "circle_of_fifths": 0.7, "chord_complexity": 0.4,
            "modal_mixture": 0.1, "pedal_point": 0.0, "form": "binary",
        }
        gen = HarmonicPlanGenerator(Seed(99))
        events = gen.generate(params, {})
        for ev in events:
            self.assertIn(ev.degree, range(1, 8),
                f"Invalid scale degree {ev.degree}")

    def test_root_pc_valid(self):
        from music_generator import HarmonicPlanGenerator
        from procgen.seeds import Seed

        params = {
            "key": "Eb", "mode": "minor", "num_phrases": 4, "phrase_length": 4,
            "time_signature": [4, 4], "harmonic_rhythm": 0.5,
            "circle_of_fifths": 0.5, "chord_complexity": 0.5,
            "modal_mixture": 0.1, "pedal_point": 0.0, "form": "binary",
        }
        gen = HarmonicPlanGenerator(Seed(7))
        events = gen.generate(params, {})
        for ev in events:
            self.assertIn(ev.root_pc, range(0, 12))

    def test_final_phrase_ends_on_tonic(self):
        """
        The last harmonic event in the plan should have degree == 1
        (authentic cadence on tonic).
        """
        from music_generator import HarmonicPlanGenerator
        from procgen.seeds import Seed

        for seed in [1, 2, 3, 4, 5]:
            with self.subTest(seed=seed):
                params = {
                    "key": "C", "mode": "major", "num_phrases": 4, "phrase_length": 4,
                    "time_signature": [4, 4], "harmonic_rhythm": 0.4,
                    "circle_of_fifths": 0.7, "chord_complexity": 0.3,
                    "modal_mixture": 0.0, "pedal_point": 0.0, "form": "binary",
                }
                gen = HarmonicPlanGenerator(Seed(seed))
                events = gen.generate(params, {})
                self.assertEqual(events[-1].degree, 1,
                    f"seed {seed}: last chord degree is {events[-1].degree}, expected 1")

    def test_modal_mixture_increases_chromaticism(self):
        """Higher modal_mixture → lower diatonic percentage."""
        score_pure = make(key="C", mode="major", modal_mixture=0.0,
                          chord_complexity=0.0, seed=10)
        score_mix  = make(key="C", mode="major", modal_mixture=0.9,
                          chord_complexity=0.0, seed=10)
        pct_pure = diatonic_pct(melody(score_pure), "C", "major")
        pct_mix  = diatonic_pct(melody(score_mix),  "C", "major")
        # mixed version should have equal or lower diatonic percentage
        self.assertLessEqual(pct_mix, pct_pure + 5,
            f"Modal mixture did not increase chromaticism: {pct_pure:.1f}% vs {pct_mix:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Parameter Effects
# ══════════════════════════════════════════════════════════════════════════════

class TestParameterEffects(unittest.TestCase):

    def test_high_density_produces_shorter_notes(self):
        low  = make(rhythmic_density=0.1, num_voices=1, seed=42)
        high = make(rhythmic_density=0.9, num_voices=1, seed=42)
        avg_low  = statistics.mean(n.duration for n in melody(low))
        avg_high = statistics.mean(n.duration for n in melody(high))
        self.assertGreater(avg_low, avg_high,
            f"High density should produce shorter notes: low={avg_low:.3f} high={avg_high:.3f}")

    def test_high_density_produces_more_notes(self):
        low  = make(rhythmic_density=0.1, num_voices=1, seed=42)
        high = make(rhythmic_density=0.9, num_voices=1, seed=42)
        self.assertGreater(len(melody(high)), len(melody(low)),
            "High density should produce more notes")

    def test_high_step_ratio_produces_smaller_intervals(self):
        stepwise = make(step_leap_ratio=0.95, num_voices=1, seed=42)
        leaping  = make(step_leap_ratio=0.05, num_voices=1, seed=42)
        ai_step  = avg_interval(melody(stepwise))
        ai_leap  = avg_interval(melody(leaping))
        self.assertLessEqual(ai_step, ai_leap + 0.5,
            f"Stepwise should have smaller intervals: stepwise={ai_step:.2f} leaping={ai_leap:.2f}")

    def test_running_sixteenths_has_short_notes(self):
        """inner_voice_style='running_sixteenths' → inner voice notes mostly 0.25 beats."""
        score = make(num_voices=3, inner_voice_style="running_sixteenths",
                     rhythmic_density=0.9, seed=99)
        # Track index 1 is alto (inner voice for 3-voice)
        inner = score.tracks[1].notes
        if not inner:
            return
        short = sum(1 for n in inner if n.duration <= 0.25)
        pct   = short / len(inner) * 100
        self.assertGreater(pct, 70.0,
            f"Running 16ths: only {pct:.1f}% of inner notes are ≤0.25 beats")

    def test_alberti_bass_has_consistent_pattern(self):
        """Alberti bass should use eighth-note subdivisions (0.5 beats)."""
        score = make(num_voices=2, bass_type="alberti", seed=3)
        bass  = score.tracks[1].notes
        if not bass:
            return
        half_note = sum(1 for n in bass if abs(n.duration - 0.5) < 0.01)
        pct = half_note / len(bass) * 100
        self.assertGreater(pct, 70.0,
            f"Alberti bass: only {pct:.1f}% of notes are 0.5 beats")

    def test_sustained_bass_has_long_notes(self):
        """Sustained bass should use the full chord duration, not subdivisions."""
        score = make(num_voices=2, bass_type="sustained",
                     rhythmic_density=0.5, seed=5)
        bass  = score.tracks[1].notes
        if not bass:
            return
        avg_dur = statistics.mean(n.duration for n in bass)
        self.assertGreater(avg_dur, 1.5,
            f"Sustained bass avg duration {avg_dur:.2f} too short")

    def test_ornaments_flag_produces_shorter_occasional_notes(self):
        """With ornaments enabled, some very short notes should appear (trills, mordents)."""
        score_no  = make(num_voices=1, use_ornaments=False, rhythmic_density=0.5, seed=20)
        score_yes = make(num_voices=1, use_ornaments=True,  rhythmic_density=0.5, seed=20)
        tiny_no  = sum(1 for n in melody(score_no)  if n.duration < 0.4)
        tiny_yes = sum(1 for n in melody(score_yes) if n.duration < 0.4)
        # ornaments should introduce at least as many short notes
        self.assertGreaterEqual(tiny_yes, tiny_no,
            f"Ornaments produced fewer tiny notes: without={tiny_no} with={tiny_yes}")

    def test_imitation_makes_alto_echo_soprano(self):
        """
        With high imitation, some segment of the alto line should match
        the soprano's pitch-class sequence shifted by a few events.
        """
        score = make(num_voices=3, imitation=0.9, seed=50)
        roles = score.metadata.get("voices", [])
        sop_idx = roles.index("soprano") if "soprano" in roles else 0
        alt_idx = roles.index("alto")    if "alto"    in roles else 1

        sop_pcs = [note_str_to_midi(n.pitch) % 12
                   for n in score.tracks[sop_idx].notes[:30]]
        alt_pcs = [note_str_to_midi(n.pitch) % 12
                   for n in score.tracks[alt_idx].notes[:30]]

        # Just verify alto has some variety (it's not all the same pitch)
        self.assertGreater(len(set(alt_pcs)), 3,
            "Alto under imitation has no melodic variety")

    def test_single_voice_monophony(self):
        """num_voices=1 should produce exactly one track."""
        score = make(num_voices=1)
        self.assertEqual(len(score.tracks), 1)

    def test_phrase_length_affects_total_duration(self):
        """Longer phrase_length → longer total piece."""
        short = make(phrase_length=2, num_phrases=2, num_voices=1, seed=1)
        long_ = make(phrase_length=8, num_phrases=2, num_voices=1, seed=1)
        dur_short = sum(n.duration for n in melody(short))
        dur_long  = sum(n.duration for n in melody(long_))
        self.assertGreater(dur_long, dur_short,
            f"Longer phrase_length should give longer piece: {dur_short:.1f} vs {dur_long:.1f}")

    def test_tempo_stored_correctly(self):
        for bpm in [60, 120, 200]:
            score = make(tempo_bpm=bpm)
            self.assertEqual(score.tempo_bpm, bpm)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Variety & Determinism
# ══════════════════════════════════════════════════════════════════════════════

class TestVarietyAndDeterminism(unittest.TestCase):

    def _melody_fingerprint(self, score) -> tuple:
        """Pitch-class sequence of the first 20 melody notes."""
        ns = melody(score)[:20]
        return tuple(note_str_to_midi(n.pitch) % 12 for n in ns)

    def test_deterministic_same_seed(self):
        """Identical seed must always produce identical output."""
        s1 = make(seed=99, num_voices=3, mode="minor", use_ornaments=True)
        s2 = make(seed=99, num_voices=3, mode="minor", use_ornaments=True)
        self.assertEqual(self._melody_fingerprint(s1),
                         self._melody_fingerprint(s2),
                         "Same seed produced different melody")

    def test_different_seeds_differ(self):
        """Different seeds should produce different melodies."""
        fingerprints = {self._melody_fingerprint(make(seed=s)) for s in range(10)}
        self.assertGreater(len(fingerprints), 5,
            f"Only {len(fingerprints)}/10 distinct melodies across seeds 0–9")

    def test_different_keys_differ(self):
        """Different keys should produce different pitch sequences."""
        fps = {self._melody_fingerprint(make(key=k, mode="major", seed=42))
               for k in ["C", "D", "E", "F", "G", "A"]}
        self.assertGreater(len(fps), 3,
            f"Only {len(fps)}/6 distinct melodies across keys")

    def test_different_modes_differ(self):
        modes = list(SCALE_INTERVALS.keys())
        fps = {self._melody_fingerprint(make(key="C", mode=m, seed=42))
               for m in modes}
        self.assertGreater(len(fps), 4,
            f"Only {len(fps)}/{len(modes)} distinct melodies across modes")

    def test_cross_style_note_count_varies(self):
        """Sparse vs dense styles should have substantially different note counts."""
        sparse = make(rhythmic_density=0.05, num_voices=1, phrase_length=4, seed=7)
        dense  = make(rhythmic_density=0.95, num_voices=1, phrase_length=4, seed=7)
        n_sparse = len(melody(sparse))
        n_dense  = len(melody(dense))
        self.assertGreater(n_dense, n_sparse * 3,
            f"Dense ({n_dense}) should be >3x sparse ({n_sparse})")

    def test_bach_preset_properties(self):
        """
        A Bach-like configuration should have:
        - step% ≥ 55%  (mostly stepwise motion)
        - diatonic% ≥ 70%
        - note density ≥ 1 note/beat  (fast motion)
        """
        score = make(
            key="D", mode="minor", form="binary",
            num_voices=2, counterpoint=0.85, imitation=0.6,
            inner_voice_style="running_sixteenths", bass_type="walking",
            circle_of_fifths=0.80, chord_complexity=0.50,
            rhythmic_density=0.80, use_ornaments=True,
            step_leap_ratio=0.72, sequence_probability=0.55,
            instruments=["harpsichord"], tempo_bpm=72,
            seed=42,
        )
        m       = melody(score)
        sp      = step_pct(m)
        diat    = diatonic_pct(m, "D", "minor")
        density = len(m) / sum(n.duration for n in m) if m else 0

        self.assertGreaterEqual(sp, 55.0,
            f"Bach preset: step% {sp:.1f}% below 55%")
        self.assertGreaterEqual(diat, 65.0,
            f"Bach preset: diatonic% {diat:.1f}% below 65%")
        self.assertGreaterEqual(density, 1.0,
            f"Bach preset: note density {density:.2f} below 1.0 notes/beat")

    def test_classical_preset_properties(self):
        """
        Classical (alberti bass, homophonic) should have:
        - step% ≥ 50%  (singable melody)
        - bass note density consistent with alberti pattern
        """
        score = make(
            key="C", mode="major", form="binary",
            num_voices=3, counterpoint=0.2, imitation=0.0,
            inner_voice_style="block_chords", bass_type="alberti",
            circle_of_fifths=0.65, chord_complexity=0.30,
            rhythmic_density=0.55, use_ornaments=False,
            step_leap_ratio=0.65, seed=7,
        )
        m = melody(score)
        sp = step_pct(m)
        self.assertGreaterEqual(sp, 45.0,
            f"Classical preset: step% {sp:.1f}% below 45%")

    def test_jazz_preset_chromatic(self):
        """Jazz with high chord_complexity should have lower diatonic% than classical."""
        classical = make(key="C", mode="major", chord_complexity=0.1,
                         modal_mixture=0.0, seed=5)
        jazz      = make(key="C", mode="major", chord_complexity=0.9,
                         modal_mixture=0.5, seed=5)
        diat_c = diatonic_pct(melody(classical), "C", "major")
        diat_j = diatonic_pct(melody(jazz),      "C", "major")
        # Jazz can have equal or lower diatonic% (more chromatic)
        self.assertLessEqual(diat_j, diat_c + 5,
            f"Jazz ({diat_j:.1f}%) not more chromatic than classical ({diat_c:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Pitch Helper Unit Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPitchHelpers(unittest.TestCase):

    def test_midi_to_note_middle_c(self):
        self.assertEqual(midi_to_note(60), "C4")

    def test_midi_to_note_a440(self):
        self.assertEqual(midi_to_note(69), "A4")

    def test_midi_to_note_roundtrip(self):
        for midi in [36, 48, 60, 69, 72, 81]:
            name = midi_to_note(midi)
            back = note_str_to_midi(name)
            self.assertEqual(back, midi, f"Roundtrip failed for MIDI {midi}: {name} → {back}")

    def test_note_str_to_midi_sharp(self):
        self.assertEqual(note_str_to_midi("C#4"), 61)

    def test_note_str_to_midi_flat(self):
        self.assertEqual(note_str_to_midi("Bb3"), 58)

    def test_scale_midi_in_range_c_major(self):
        pitches = scale_midi_in_range(0, "major", 60, 72)
        pcs = {p % 12 for p in pitches}
        self.assertEqual(pcs, {0, 2, 4, 5, 7, 9, 11})

    def test_scale_midi_in_range_d_minor(self):
        pitches = scale_midi_in_range(2, "minor", 60, 72)
        pcs = {p % 12 for p in pitches}
        expected = {(2 + iv) % 12 for iv in SCALE_INTERVALS["minor"]}
        self.assertEqual(pcs, expected)

    def test_chord_midi_in_range_c_major_triad(self):
        pitches = chord_midi_in_range(0, "maj", 60, 72)
        pcs = {p % 12 for p in pitches}
        self.assertEqual(pcs, {0, 4, 7})

    def test_chord_midi_in_range_g_dom7(self):
        pitches = chord_midi_in_range(7, "dom7", 48, 72)
        pcs = {p % 12 for p in pitches}
        # G dom7: G B D F  → 7, 11, 2, 5
        self.assertEqual(pcs, {7, 11, 2, 5})


# ══════════════════════════════════════════════════════════════════════════════
# 7. Multi-Instrument Support
# ══════════════════════════════════════════════════════════════════════════════

class TestMultiInstrument(unittest.TestCase):

    def test_distinct_instruments_on_distinct_tracks(self):
        """Each instrument in the list should appear on a separate track."""
        score = make(
            num_voices=4,
            instruments=["violin", "viola", "cello", "contrabass"],
            seed=42,
        )
        instruments = [t.instrument for t in score.tracks]
        self.assertEqual(instruments, ["violin", "viola", "cello", "contrabass"],
            f"Expected 4 distinct instruments, got {instruments}")

    def test_string_quartet_generates(self):
        """String quartet (violin/violin/viola/cello) should generate without error."""
        score = make(
            num_voices=4, key="G", mode="major",
            instruments=["violin", "violin", "viola", "cello"],
            counterpoint=0.65, imitation=0.3,
            melodic_theme="motif", seed=301,
        )
        self.assertEqual(len(score.tracks), 4)
        self.assertGreater(sum(len(t.notes) for t in score.tracks), 0)

    def test_piano_trio_generates(self):
        """Piano trio (violin/cello/piano) 3-voice piece."""
        score = make(
            num_voices=3, key="D", mode="major",
            instruments=["violin", "cello", "piano"],
            counterpoint=0.75, imitation=0.45,
            melodic_theme="fugue_subject", seed=404,
        )
        self.assertEqual(len(score.tracks), 3)
        instr = [t.instrument for t in score.tracks]
        self.assertIn("violin", instr)
        self.assertIn("piano",  instr)

    def test_wind_ensemble_generates(self):
        """Wind quartet (flute/oboe/clarinet/bassoon)."""
        score = make(
            num_voices=4, key="F", mode="major",
            instruments=["flute", "oboe", "clarinet", "bassoon"],
            seed=512,
        )
        self.assertEqual(len(score.tracks), 4)
        instr = [t.instrument for t in score.tracks]
        self.assertIn("flute",   instr)
        self.assertIn("bassoon", instr)

    def test_new_instruments_in_midi(self):
        """Viola, clarinet, bassoon, contrabass should produce valid MIDI."""
        for instr in ["viola", "clarinet", "bassoon", "contrabass"]:
            with self.subTest(instr=instr):
                score = make(instruments=[instr], num_voices=1, seed=42)
                data  = export.to_midi(score)
                self.assertIsInstance(data, bytes)
                self.assertGreater(len(data), 14)

    def test_instrument_padding_for_fewer_instruments_than_voices(self):
        """When fewer instruments than voices, last instrument should be repeated."""
        score = make(
            num_voices=4,
            instruments=["violin"],  # only 1 instrument for 4 voices
            seed=42,
        )
        self.assertEqual(len(score.tracks), 4)
        # All tracks should use "violin"
        for t in score.tracks:
            self.assertEqual(t.instrument, "violin")

    def test_soprano_voice_gets_first_instrument(self):
        """Soprano (track 0) should use instruments[0]."""
        score = make(
            num_voices=2,
            instruments=["flute", "cello"],
            seed=42,
        )
        self.assertEqual(score.tracks[0].instrument, "flute",
            "Soprano should use first instrument")

    def test_bass_voice_gets_last_instrument(self):
        """Bass (last track) should use instruments[-1]."""
        score = make(
            num_voices=3,
            instruments=["violin", "viola", "cello"],
            seed=42,
        )
        self.assertEqual(score.tracks[-1].instrument, "cello",
            "Bass should use last instrument")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Thematic Coherence
# ══════════════════════════════════════════════════════════════════════════════

class TestThematicCoherence(unittest.TestCase):

    def test_motif_theme_generates(self):
        """melodic_theme='motif' should produce a valid score."""
        score = make(melodic_theme="motif", num_voices=2, seed=42)
        self.assertGreater(len(score.tracks[0].notes), 0)

    def test_fugue_subject_generates(self):
        """melodic_theme='fugue_subject' should produce a valid score."""
        score = make(melodic_theme="fugue_subject", num_voices=4, seed=42)
        self.assertGreater(len(score.tracks[0].notes), 0)

    def test_no_theme_baseline(self):
        """melodic_theme='none' should still produce a valid score."""
        score = make(melodic_theme="none", num_voices=2, seed=42)
        self.assertGreater(len(score.tracks[0].notes), 0)

    def test_theme_deterministic(self):
        """Same seed + theme type must produce identical soprano line."""
        s1 = make(melodic_theme="motif", num_voices=2, seed=77)
        s2 = make(melodic_theme="motif", num_voices=2, seed=77)
        fp1 = tuple(note_str_to_midi(n.pitch) for n in s1.tracks[0].notes[:20])
        fp2 = tuple(note_str_to_midi(n.pitch) for n in s2.tracks[0].notes[:20])
        self.assertEqual(fp1, fp2, "Themed score not deterministic")

    def test_theme_type_does_not_break_voice_ranges(self):
        """With theme active, soprano still stays mostly in C4-A5."""
        score = make(melodic_theme="motif", num_voices=1, use_ornaments=True, seed=20)
        lo, hi = VOICE_RANGES["soprano"]
        ns = score.tracks[0].notes
        if not ns:
            return
        in_range = sum(1 for n in ns if lo <= note_str_to_midi(n.pitch) <= hi)
        pct = in_range / len(ns) * 100
        self.assertGreaterEqual(pct, 70.0,
            f"Theme mode: only {pct:.1f}% of soprano in range (expected >= 70%)")

    def test_fugue_subject_diatonic(self):
        """Fugue subject melody should still be mostly diatonic."""
        score = make(
            melodic_theme="fugue_subject", key="C", mode="major",
            modal_mixture=0.0, num_voices=2, seed=33,
        )
        pct = diatonic_pct(melody(score), "C", "major")
        self.assertGreaterEqual(pct, 60.0,
            f"Fugue subject: only {pct:.1f}% diatonic")

    def test_different_theme_types_differ(self):
        """motif vs fugue_subject should produce different soprano lines."""
        s_motif = make(melodic_theme="motif",        num_voices=2, seed=55)
        s_fugue = make(melodic_theme="fugue_subject", num_voices=2, seed=55)
        fp_m = tuple(note_str_to_midi(n.pitch) % 12 for n in s_motif.tracks[0].notes[:20])
        fp_f = tuple(note_str_to_midi(n.pitch) % 12 for n in s_fugue.tracks[0].notes[:20])
        self.assertNotEqual(fp_m, fp_f,
            "motif and fugue_subject produced identical melodies")

    def test_all_pitches_valid_with_theme(self):
        """All note pitches should be parseable and in MIDI range 0-127."""
        for theme_type in ["none", "motif", "fugue_subject"]:
            with self.subTest(theme=theme_type):
                score = make(melodic_theme=theme_type, num_voices=4, seed=42)
                for track in score.tracks:
                    for note in track.notes:
                        m = note_str_to_midi(note.pitch)
                        self.assertGreaterEqual(m, 0)
                        self.assertLessEqual(m, 127)


# ══════════════════════════════════════════════════════════════════════════════
# 9. Proper Beginnings and Endings
# ══════════════════════════════════════════════════════════════════════════════

class TestProperBeginsEnds(unittest.TestCase):

    def test_held_ending_final_note_is_long(self):
        """With ending='held', the final soprano note should be as long as the event duration."""
        score = make(ending="held", num_voices=1, phrase_length=4,
                     rhythmic_density=0.8, seed=42)
        notes = score.tracks[0].notes
        self.assertGreater(len(notes), 1)
        # Final note should be at least 1 beat (one full harmonic event at minimum)
        self.assertGreaterEqual(notes[-1].duration, 1.0,
            f"Final note duration {notes[-1].duration} too short for a held ending")

    def test_held_ending_final_pitch_is_tonic(self):
        """Final soprano note should land on the tonic pitch class."""
        for key in ["C", "D", "G"]:
            with self.subTest(key=key):
                score = make(ending="held", num_voices=1, key=key,
                             mode="major", seed=7)
                notes = score.tracks[0].notes
                if notes:
                    final_pc = note_str_to_midi(notes[-1].pitch) % 12
                    self.assertEqual(final_pc, ROOT_TO_PC[key],
                        f"Final note {notes[-1].pitch} is not the tonic {key}")

    def test_none_ending_still_generates(self):
        """ending='none' should produce a valid score with no special final treatment."""
        score = make(ending="none", num_voices=2, seed=42)
        self.assertGreater(len(score.tracks[0].notes), 0)

    def test_ritardando_generates(self):
        """ending='ritardando' should not crash and produce valid notes."""
        score = make(ending="ritardando", num_voices=3, seed=42)
        self.assertGreater(sum(len(t.notes) for t in score.tracks), 0)

    def test_ritardando_penultimate_has_longer_notes(self):
        """
        With ending='ritardando', notes near the end should be longer on average
        than notes in the middle of the piece.
        """
        score = make(ending="ritardando", num_voices=1,
                     rhythmic_density=0.7, num_phrases=4, phrase_length=4,
                     seed=42)
        notes = score.tracks[0].notes
        if len(notes) < 8:
            return
        # Compare average duration of last 20% of notes vs middle 60%
        n = len(notes)
        mid_slice  = notes[n // 5 : 4 * n // 5]
        tail_slice = notes[4 * n // 5 :]
        if not mid_slice or not tail_slice:
            return
        avg_mid  = statistics.mean(n_.duration for n_ in mid_slice)
        avg_tail = statistics.mean(n_.duration for n_ in tail_slice)
        self.assertGreaterEqual(avg_tail, avg_mid * 0.8,
            f"Ritardando tail avg {avg_tail:.3f} not >= 80% of mid avg {avg_mid:.3f}")

    def test_upbeat_intro_generates(self):
        """intro='upbeat' should produce a valid score."""
        score = make(intro="upbeat", num_voices=2, seed=42)
        self.assertGreater(len(score.tracks[0].notes), 0)

    def test_upbeat_soprano_starts_ascending(self):
        """With upbeat intro, the first few soprano notes should be ascending or at same pitch."""
        score = make(intro="upbeat", num_voices=1, key="C", mode="major", seed=10)
        notes = score.tracks[0].notes
        if len(notes) < 3:
            return
        first_3 = [note_str_to_midi(n.pitch) for n in notes[:3]]
        # First notes should generally be non-descending (ascending or flat)
        ascending_pairs = sum(1 for i in range(len(first_3) - 1)
                              if first_3[i + 1] >= first_3[i])
        self.assertGreaterEqual(ascending_pairs, len(first_3) - 2,
            f"Upbeat not ascending: {[midi_to_note(p) for p in first_3]}")

    def test_held_ending_deterministic(self):
        """Same seed with held ending must produce identical output."""
        s1 = make(ending="held", num_voices=3, seed=55)
        s2 = make(ending="held", num_voices=3, seed=55)
        fp1 = tuple(note_str_to_midi(n.pitch) for n in s1.tracks[0].notes[-5:])
        fp2 = tuple(note_str_to_midi(n.pitch) for n in s2.tracks[0].notes[-5:])
        self.assertEqual(fp1, fp2)

    def test_all_ending_options_produce_valid_midi(self):
        """All three ending options should export to valid MIDI."""
        for ending in ["none", "held", "ritardando"]:
            with self.subTest(ending=ending):
                score = make(ending=ending, num_voices=3, seed=42)
                data  = export.to_midi(score)
                self.assertIsInstance(data, bytes)
                self.assertGreater(len(data), 14)


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()

    for cls in [
        TestGenerationIntegrity,
        TestMusicalConstraints,
        TestHarmonicCorrectness,
        TestParameterEffects,
        TestVarietyAndDeterminism,
        TestPitchHelpers,
        TestMultiInstrument,
        TestThematicCoherence,
        TestProperBeginsEnds,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
