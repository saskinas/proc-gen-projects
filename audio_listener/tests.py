"""
Tests for audio_listener — the MIDI analysis pipeline.

Runs with standard unittest (no pytest needed):
    python tests.py
"""

from __future__ import annotations

import pathlib
import struct
import sys
import unittest

HERE     = pathlib.Path(__file__).resolve().parent
PROJECTS = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECTS / "music_generator"))
sys.path.insert(0, str(PROJECTS / "procgen"))


# ── Helpers ──────────────────────────────────────────────────────────────────

def _var_len(v: int) -> bytes:
    result = [v & 0x7F]; v >>= 7
    while v:
        result.insert(0, (v & 0x7F) | 0x80); v >>= 7
    return bytes(result)


def _build_midi(
    tracks_events: list[list[tuple]],  # each track: list of (abs_tick, type, *args)
    time_sig: tuple = (4, 4),
    tempo_bpm: int = 120,
    ticks_per_beat: int = 480,
) -> bytes:
    """
    Build a minimal Format-1 MIDI file from raw event lists.

    Event tuples:
      ("note",   abs_tick, channel, pitch, velocity, duration_ticks)
      ("name",   abs_tick, name_str)      — track name meta
      ("tempo",  abs_tick, bpm)           — set tempo meta
      ("timesig", abs_tick, num, den)     — time signature meta
    """
    uspb = int(60_000_000 / tempo_bpm)
    num, den = time_sig
    log2_den = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}.get(den, 2)

    def make_track(raw_events: list) -> bytes:
        # Expand note events into note_on / note_off
        flat: list[tuple] = []
        for ev in raw_events:
            if ev[0] == "note":
                _, tick, ch, pitch, vel, dur = ev
                flat.append((tick,          "on",  ch, pitch, vel))
                flat.append((tick + dur,    "off", ch, pitch, 0))
            elif ev[0] == "name":
                _, tick, name = ev
                flat.append((tick, "name", name))
            elif ev[0] == "tempo":
                _, tick, bpm = ev
                us = int(60_000_000 / bpm)
                flat.append((tick, "tempo", us))
            elif ev[0] == "timesig":
                _, tick, n, d = ev
                ld = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}.get(d, 2)
                flat.append((tick, "timesig", n, ld))

        flat.sort(key=lambda x: x[0])

        data = bytearray()
        prev = 0
        for ev in flat:
            delta = ev[0] - prev
            prev  = ev[0]
            if ev[1] == "on":
                data += _var_len(delta) + bytes([0x90 | ev[2], ev[3], ev[4]])
            elif ev[1] == "off":
                data += _var_len(delta) + bytes([0x80 | ev[2], ev[3], 0])
            elif ev[1] == "name":
                name_b = ev[2].encode()
                data += _var_len(delta) + bytes([0xFF, 0x03, len(name_b)]) + name_b
            elif ev[1] == "tempo":
                us = ev[2]
                data += _var_len(delta) + b"\xFF\x51\x03"
                data += bytes([(us >> 16) & 0xFF, (us >> 8) & 0xFF, us & 0xFF])
            elif ev[1] == "timesig":
                data += _var_len(delta) + b"\xFF\x58\x04" + bytes([ev[2], ev[3], 24, 8])
        data += _var_len(0) + b"\xFF\x2F\x00"
        return bytes(data)

    header = struct.pack(">4sIHHH", b"MThd", 6, 1, len(tracks_events), ticks_per_beat)
    body   = b""
    for raw in tracks_events:
        td = make_track(raw)
        body += b"MTrk" + struct.pack(">I", len(td)) + td
    return header + body


def _simple_c_major_midi(phrases: int = 4, tpb: int = 480) -> bytes:
    """C major scale melody + root bass, 4/4, 120 BPM, `phrases` × 4 bars."""
    pitches = [60, 62, 64, 65, 67, 69, 67, 65] * phrases
    events  = [("tempo",   0, 120), ("timesig", 0, 4, 4)]
    for i, p in enumerate(pitches):
        tick = i * tpb
        events.append(("note", tick, 0, p, 80, tpb - 1))

    bass_pcs = [48, 43, 45, 48] * phrases
    for i, p in enumerate(bass_pcs):
        tick = i * tpb * 2
        events.append(("note", tick, 1, p, 65, tpb * 2 - 1))

    return _build_midi([events], time_sig=(4, 4), tempo_bpm=120, ticks_per_beat=tpb)


def _waltz_midi(tpb: int = 480) -> bytes:
    """3/4 waltz MIDI, 165 BPM, 8 bars of melody + bass."""
    pitches = [67, 69, 71, 72, 71, 69, 67, 65, 64, 65, 67, 65, 64, 62, 60, 60,
               67, 69, 71, 72, 71, 69, 67, 65, 64, 62, 60, 60]
    events  = [("tempo", 0, 165), ("timesig", 0, 3, 4)]
    for i, p in enumerate(pitches):
        tick = i * tpb
        events.append(("note", tick, 0, p, 80, tpb - 1))

    bass = [48, 43, 48, 43, 40, 43] * 5
    for i, p in enumerate(bass):
        tick = i * tpb * 3
        events.append(("note", tick, 1, p, 65, tpb * 3 - 1))

    return _build_midi([events], time_sig=(3, 4), tempo_bpm=165, ticks_per_beat=tpb)


# ── Test cases ────────────────────────────────────────────────────────────────

class TestMidiParser(unittest.TestCase):
    """audio_listener.midi_parser — basic parse correctness."""

    def test_parse_returns_midadata(self):
        from audio_listener.midi_parser import parse_midi, MidiData
        data = _simple_c_major_midi()
        result = parse_midi(data)
        self.assertIsInstance(result, MidiData)

    def test_channels_present(self):
        from audio_listener.midi_parser import parse_midi
        data = _simple_c_major_midi()
        raw  = parse_midi(data)
        chs  = raw.channels()
        self.assertIn(0, chs)
        self.assertIn(1, chs)

    def test_notes_for_channel_nonempty(self):
        from audio_listener.midi_parser import parse_midi
        data = _simple_c_major_midi()
        raw  = parse_midi(data)
        notes = raw.notes_for_channel(0)
        self.assertGreater(len(notes), 0)

    def test_ticks_per_beat(self):
        from audio_listener.midi_parser import parse_midi
        data = _simple_c_major_midi(tpb=480)
        raw  = parse_midi(data)
        self.assertEqual(raw.ticks_per_beat, 480)

    def test_tempo_changes_present(self):
        from audio_listener.midi_parser import parse_midi
        data = _simple_c_major_midi()
        raw  = parse_midi(data)
        self.assertIsInstance(raw.tempo_changes, list)
        self.assertGreater(len(raw.tempo_changes), 0)

    def test_total_ticks_positive(self):
        from audio_listener.midi_parser import parse_midi
        data = _simple_c_major_midi()
        raw  = parse_midi(data)
        self.assertGreater(raw.total_ticks, 0)

    def test_waltz_timesig(self):
        from audio_listener.midi_parser  import parse_midi
        from audio_listener.tempo_meter  import detect_time_signature
        data  = _waltz_midi()
        raw   = parse_midi(data)
        ts    = detect_time_signature(raw)
        self.assertEqual(ts[0], 3)


class TestTempoMeter(unittest.TestCase):
    """audio_listener.tempo_meter — BPM and time-sig detection."""

    def test_120_bpm(self):
        from audio_listener.midi_parser import parse_midi
        from audio_listener.tempo_meter import detect_tempo
        raw = parse_midi(_simple_c_major_midi())
        self.assertEqual(detect_tempo(raw), 120)

    def test_165_bpm(self):
        from audio_listener.midi_parser import parse_midi
        from audio_listener.tempo_meter import detect_tempo
        raw = parse_midi(_waltz_midi())
        self.assertEqual(detect_tempo(raw), 165)

    def test_time_sig_4_4(self):
        from audio_listener.midi_parser import parse_midi
        from audio_listener.tempo_meter import detect_time_signature
        raw = parse_midi(_simple_c_major_midi())
        ts  = detect_time_signature(raw)
        self.assertEqual(ts[0], 4)
        self.assertEqual(ts[1], 4)

    def test_time_sig_3_4(self):
        from audio_listener.midi_parser import parse_midi
        from audio_listener.tempo_meter import detect_time_signature
        raw = parse_midi(_waltz_midi())
        ts  = detect_time_signature(raw)
        self.assertEqual(ts[0], 3)


class TestPitchTracker(unittest.TestCase):
    """audio_listener.pitch_tracker — note extraction and quantisation."""

    def _extract(self, midi_bytes, channel=0) -> list:
        from audio_listener.midi_parser  import parse_midi
        from audio_listener.pitch_tracker import extract_notes
        raw   = parse_midi(midi_bytes)
        notes = raw.notes_for_channel(channel)
        return extract_notes(notes, raw.ticks_per_beat, raw.tempo_changes)

    def test_note_count(self):
        notes = self._extract(_simple_c_major_midi(phrases=1))
        # 8 pitches in one phrase
        self.assertEqual(len(notes), 8)

    def test_pitches_correct(self):
        notes = self._extract(_simple_c_major_midi(phrases=1))
        pitches = [n.pitch for n in notes]
        self.assertEqual(pitches, [60, 62, 64, 65, 67, 69, 67, 65])

    def test_start_beats_ascending(self):
        notes = self._extract(_simple_c_major_midi(phrases=1))
        starts = [n.start_beat for n in notes]
        self.assertEqual(starts, sorted(starts))

    def test_duration_positive(self):
        notes = self._extract(_simple_c_major_midi(phrases=1))
        for n in notes:
            self.assertGreater(n.duration_beats, 0)

    def test_quantise_does_not_crash(self):
        from audio_listener.midi_parser  import parse_midi
        from audio_listener.pitch_tracker import extract_notes, quantise_beats
        raw   = parse_midi(_simple_c_major_midi())
        notes = raw.notes_for_channel(0)
        tk    = extract_notes(notes, raw.ticks_per_beat, raw.tempo_changes)
        q     = quantise_beats(tk)
        self.assertGreater(len(q), 0)


class TestChannelRouter(unittest.TestCase):
    """audio_listener.channel_router — channel role assignment."""

    def _assign(self, midi_bytes) -> object:
        from audio_listener.midi_parser    import parse_midi
        from audio_listener.channel_router import assign_channels
        return assign_channels(parse_midi(midi_bytes))

    def test_bass_channel_identified(self):
        # Channel 1 is named 'Triangle' → should be bass
        mid = _simple_c_major_midi()
        # Rebuild with explicit triangle track name
        from audio_listener.midi_parser    import parse_midi
        from audio_listener.channel_router import assign_channels
        raw  = parse_midi(mid)
        # Inject track name
        raw.track_names[1] = "Triangle"
        assignment = assign_channels(raw)
        self.assertEqual(assignment.bass, 1)

    def test_melody_is_not_none(self):
        assignment = self._assign(_simple_c_major_midi())
        self.assertIsNotNone(assignment.melody)

    def test_drums_none_when_no_drum_channel(self):
        assignment = self._assign(_simple_c_major_midi())
        self.assertIsNone(assignment.drums)

    def test_channel_9_is_drums(self):
        # Build MIDI with a ch9 track
        events = [("tempo", 0, 120), ("timesig", 0, 4, 4)]
        for i in range(16):
            events.append(("note", i * 480, 9, 36, 100, 100))
        events += [("note", i * 480, 0, 60 + (i % 8), 80, 479) for i in range(16)]
        mid = _build_midi([events])
        from audio_listener.midi_parser    import parse_midi
        from audio_listener.channel_router import assign_channels
        raw  = parse_midi(mid)
        asgn = assign_channels(raw)
        self.assertEqual(asgn.drums, 9)

    def test_sparse_channel_ignored(self):
        # Channel 2 has only 2 notes — should be in ignored list
        events = [("tempo", 0, 120), ("timesig", 0, 4, 4)]
        for i in range(16):
            events.append(("note", i * 480, 0, 60 + (i % 8), 80, 479))
        events.append(("note", 0, 2, 72, 80, 479))
        events.append(("note", 480, 2, 74, 80, 479))
        mid  = _build_midi([events])
        from audio_listener.midi_parser    import parse_midi
        from audio_listener.channel_router import assign_channels
        raw  = parse_midi(mid)
        asgn = assign_channels(raw)
        self.assertIn(2, asgn.ignored)


class TestKeyDetector(unittest.TestCase):
    """audio_listener.key_detector — Krumhansl-Kessler key detection."""

    def _detect(self, pitches: list[int]) -> tuple:
        from audio_listener.key_detector  import detect_key_mode
        from audio_listener.pitch_tracker import TrackedNote
        notes = [TrackedNote(pitch=p, start_beat=i * 1.0, duration_beats=1.0,
                             velocity=80) for i, p in enumerate(pitches)]
        return detect_key_mode(notes, [], [])

    def test_c_major(self):
        # C major scale tones
        pitches = [60, 62, 64, 65, 67, 69, 71] * 4
        key, mode = self._detect(pitches)
        self.assertEqual(key, "C")
        self.assertIn(mode, ("major",))

    def test_g_major(self):
        pitches = [67, 69, 71, 72, 74, 76, 78] * 4
        key, mode = self._detect(pitches)
        self.assertEqual(key, "G")

    def test_a_natural_minor_or_relative(self):
        # A natural minor shares all PCs with C major.
        # Without harmonic context (bass, cadences) K-S may detect either.
        pitches = [69, 71, 72, 74, 76, 77, 79] * 4
        key, mode = self._detect(pitches)
        self.assertIn(key, ("A", "C"))


class TestPhraseSegmenter(unittest.TestCase):
    """audio_listener.phrase_segmenter — phrase and section boundaries."""

    def _notes(self, total_bars: int, bpb: int = 4) -> list:
        from audio_listener.pitch_tracker import TrackedNote
        notes = []
        pitches = [60, 62, 64, 65, 67, 65, 64, 62] * 100
        for bar in range(total_bars):
            for beat in range(bpb):
                t = bar * bpb + beat
                p = pitches[t % len(pitches)]
                notes.append(TrackedNote(pitch=p, start_beat=float(t),
                                         duration_beats=0.9, velocity=80))
        return notes

    def test_segment_returns_phrases(self):
        from audio_listener.phrase_segmenter import segment_phrases
        notes   = self._notes(16, bpb=4)
        phrases = segment_phrases(notes, 4)
        self.assertGreater(len(phrases), 0)

    def test_phrases_cover_all_notes(self):
        from audio_listener.phrase_segmenter import segment_phrases
        notes   = self._notes(16, bpb=4)
        phrases = segment_phrases(notes, 4)
        covered = sum(1 for n in notes
                      if any(ph.start_beat <= n.start_beat < ph.end_beat
                             for ph in phrases))
        self.assertEqual(covered, len(notes))

    def test_4_bar_phrases_for_short_piece(self):
        # Need > 8 bars so total_bars >= 8 threshold triggers (total_beats < 8*bpb
        # when last note duration < 1.0 makes total_bars slightly under 8).
        from audio_listener.phrase_segmenter import segment_phrases
        notes   = self._notes(9, bpb=4)
        phrases = segment_phrases(notes, 4)
        self.assertEqual(phrases[0].bar_count, 4)

    def test_8_bar_phrases_for_long_piece(self):
        from audio_listener.phrase_segmenter import segment_phrases
        notes   = self._notes(32, bpb=4)
        phrases = segment_phrases(notes, 4)
        self.assertEqual(phrases[0].bar_count, 8)

    def test_detect_sections_returns_list(self):
        from audio_listener.phrase_segmenter import segment_phrases, detect_sections
        notes    = self._notes(16, bpb=4)
        phrases  = segment_phrases(notes, 4)
        sections = detect_sections(phrases)
        self.assertIsInstance(sections, list)
        self.assertGreater(len(sections), 0)

    def test_stub_tail_dropped(self):
        """Last stub phrase shorter than 1.5 bars is dropped."""
        from audio_listener.pitch_tracker    import TrackedNote
        from audio_listener.phrase_segmenter import segment_phrases
        # 17 bars at 4/4 — with 8-bar phrases the tail = 1 bar
        notes = self._notes(17, bpb=4)
        phrases = segment_phrases(notes, 4)
        # The last phrase should not be a tiny stub (< 1.5 bars = 6 beats)
        for ph in phrases:
            dur = ph.end_beat - ph.start_beat
            self.assertGreaterEqual(dur, 4 * 1.5 - 0.1)  # at least 1.5 bars


class TestSectionLabeler(unittest.TestCase):
    """audio_listener.section_labeler — role assignment."""

    def _make_sections_and_labels(self, n_bars: int = 32) -> tuple:
        from audio_listener.pitch_tracker    import TrackedNote
        from audio_listener.phrase_segmenter import segment_phrases, detect_sections
        from audio_listener.section_labeler  import label_sections

        pitches = [60, 62, 64, 65, 67, 65, 64, 62] * 100
        notes   = [
            TrackedNote(pitch=pitches[i % len(pitches)],
                        start_beat=float(i), duration_beats=0.9, velocity=80)
            for i in range(n_bars * 4)
        ]
        total_beats = float(n_bars * 4)
        phrases     = segment_phrases(notes, 4)
        sections    = detect_sections(phrases)
        labels      = label_sections(sections, phrases, notes, total_beats)
        return sections, labels

    def test_labels_same_length_as_sections(self):
        sections, labels = self._make_sections_and_labels()
        self.assertEqual(len(sections), len(labels))

    def test_all_roles_are_valid(self):
        valid = {"intro", "outro", "verse", "chorus", "bridge", "generic"}
        _, labels = self._make_sections_and_labels()
        for role, _ in labels:
            self.assertIn(role, valid)

    def test_bridge_count_limited(self):
        _, labels = self._make_sections_and_labels(n_bars=64)
        n_sections = len(labels)
        max_bridges = max(1, n_sections // 5)
        bridge_count = sum(1 for role, _ in labels if role == "bridge")
        self.assertLessEqual(bridge_count, max_bridges)

    def test_not_all_sections_are_bridge(self):
        _, labels = self._make_sections_and_labels(n_bars=64)
        roles = [r for r, _ in labels]
        bridge_fraction = roles.count("bridge") / max(len(roles), 1)
        self.assertLess(bridge_fraction, 0.5)


class TestChordBuilder(unittest.TestCase):
    """audio_listener.chord_builder — harmonic plan construction."""

    def _tracked(self, pitches: list[int], start: float = 0.0, dur: float = 1.0) -> list:
        from audio_listener.pitch_tracker import TrackedNote
        return [TrackedNote(pitch=p, start_beat=start + i * dur,
                            duration_beats=dur, velocity=80)
                for i, p in enumerate(pitches)]

    def test_detect_harmonic_rhythm_uses_bass(self):
        from audio_listener.chord_builder import detect_harmonic_rhythm
        # Bass changes every 4 beats
        bass   = self._tracked([48, 43, 45, 48], dur=4.0)
        melody = self._tracked([60, 62, 64, 65, 67, 65, 64, 62], dur=0.5)
        rhythm = detect_harmonic_rhythm(melody, 4, bass=bass)
        # Should be ≈ 4 beats (snapped), not 0.5
        self.assertGreaterEqual(rhythm, 2.0)

    def test_detect_harmonic_rhythm_fallback(self):
        from audio_listener.chord_builder import detect_harmonic_rhythm
        rhythm = detect_harmonic_rhythm([], 4, bass=None)
        self.assertEqual(rhythm, 4.0)

    def test_build_harmonic_plan_nonempty(self):
        from audio_listener.chord_builder import build_harmonic_plan
        melody = self._tracked([60, 64, 67, 65, 62, 60], dur=2.0)
        bass   = self._tracked([48, 43, 45, 48], dur=3.0)
        plan   = build_harmonic_plan(
            melody=melody, bass=bass, counter=[],
            tonic_pc=0, mode="major",
            harm_rhythm=4.0,
            section_start_beat=0.0, section_end_beat=12.0,
            phrase_map=[(0.0, 12.0, 0)],
        )
        self.assertGreater(len(plan), 0)

    def test_no_sus4_preferred_over_maj(self):
        """maj and sus4 both match root+fifth; maj should win (penalty)."""
        from audio_listener.chord_builder import _identify_chord
        # C major chord intervals dict (minimal)
        CHORD_IV = {"maj": [0, 4, 7], "min": [0, 3, 7], "sus4": [0, 5, 7],
                    "dim": [0, 3, 6], "dom7": [0, 4, 7, 10]}
        # PC set = {0, 7} — root + fifth (ambiguous between maj and sus4)
        result = _identify_chord([0, 7], 0, CHORD_IV)
        self.assertEqual(result, "maj")


class TestMotifFinder(unittest.TestCase):
    """audio_listener.motif_finder — recurring pattern detection."""

    def _make_motif_melody(self) -> tuple:
        """Inject a 5-note motif [2, 2, -1, 2] repeated 4× across phrases."""
        from audio_listener.pitch_tracker    import TrackedNote
        from audio_listener.phrase_segmenter import PhraseSegment

        def phrase_notes(start_beat: float, pitch_offset: int = 0) -> list[TrackedNote]:
            pitches = [60 + pitch_offset, 62 + pitch_offset, 64 + pitch_offset,
                       63 + pitch_offset, 65 + pitch_offset,
                       # filler
                       60 + pitch_offset, 62 + pitch_offset, 64 + pitch_offset]
            return [TrackedNote(pitch=p, start_beat=start_beat + i,
                                duration_beats=1.0, velocity=80)
                    for i, p in enumerate(pitches)]

        phrases  = []
        all_notes: list = []
        for i in range(4):
            start = i * 8.0
            pn    = phrase_notes(start, pitch_offset=i * 2)  # transposed each time
            all_notes.extend(pn)
            phrases.append(PhraseSegment(phrase_idx=i, start_beat=start,
                                         end_beat=start + 8.0, bar_count=2,
                                         melody_fingerprint=()))
        return all_notes, phrases

    def test_find_motifs_nonempty(self):
        from audio_listener.motif_finder import find_motifs
        notes, phrases = self._make_motif_melody()
        motifs = find_motifs(notes, phrases, min_occurrences=3)
        self.assertGreater(len(motifs), 0)

    def test_motif_intervals_consecutive(self):
        from audio_listener.motif_finder import find_motifs
        notes, phrases = self._make_motif_melody()
        motifs = find_motifs(notes, phrases, min_occurrences=3)
        if motifs:
            ivs = motifs[0].intervals
            # Intervals should not all be zero or all be the same huge value
            self.assertTrue(any(iv != 0 for iv in ivs))

    def test_motif_durations_positive(self):
        from audio_listener.motif_finder import find_motifs
        notes, phrases = self._make_motif_melody()
        motifs = find_motifs(notes, phrases, min_occurrences=3)
        if motifs:
            for d in motifs[0].durations:
                self.assertGreater(d, 0.0)

    def test_empty_melody_returns_empty(self):
        from audio_listener.motif_finder    import find_motifs
        from audio_listener.phrase_segmenter import PhraseSegment
        ph = [PhraseSegment(0, 0.0, 8.0, 2, ())]
        self.assertEqual(find_motifs([], ph), [])


class TestListenMidi(unittest.TestCase):
    """Integration — listen_midi() end-to-end."""

    def _analysis(self, midi_bytes: bytes):
        from audio_listener import listen_midi
        return listen_midi(midi_bytes)

    def test_returns_musical_analysis(self):
        from music_generator.ir import MusicalAnalysis
        a = self._analysis(_simple_c_major_midi())
        self.assertIsInstance(a, MusicalAnalysis)

    def test_key_c_major(self):
        a = self._analysis(_simple_c_major_midi())
        self.assertEqual(a.key, "C")

    def test_tempo_120(self):
        a = self._analysis(_simple_c_major_midi())
        self.assertEqual(a.tempo_bpm, 120)

    def test_time_sig_4_4(self):
        a = self._analysis(_simple_c_major_midi())
        self.assertEqual(a.time_signature[0], 4)

    def test_sections_nonempty(self):
        a = self._analysis(_simple_c_major_midi(phrases=8))
        self.assertGreater(len(a.sections), 0)

    def test_sections_have_harmonic_plan(self):
        a = self._analysis(_simple_c_major_midi(phrases=8))
        for sec in a.sections:
            self.assertIsInstance(sec.harmonic_plan, list)

    def test_sections_have_texture(self):
        a = self._analysis(_simple_c_major_midi(phrases=8))
        for sec in a.sections:
            self.assertIsNotNone(sec.texture)

    def test_waltz_time_sig(self):
        a = self._analysis(_waltz_midi())
        self.assertEqual(a.time_signature[0], 3)

    def test_motifs_is_dict(self):
        a = self._analysis(_simple_c_major_midi(phrases=8))
        self.assertIsInstance(a.motifs, dict)

    def test_melodic_direction_valid(self):
        """TextureHints.melodic_direction must be one of the valid values or None."""
        valid = {"ascending", "descending", "arch", "free", None}
        a = self._analysis(_simple_c_major_midi(phrases=8))
        for sec in a.sections:
            self.assertIn(sec.texture.melodic_direction, valid)

    def test_drum_style_valid(self):
        valid = {"none", "rock", "jazz", "waltz", "latin", "brush", "funk", "swing", None}
        a = self._analysis(_simple_c_major_midi(phrases=8))
        for sec in a.sections:
            self.assertIn(sec.texture.drum_style, valid)

    def test_grape_garden_if_present(self):
        """If grape-garden.mid is available, run the full analysis."""
        path = HERE / "sample_input" / "grape-garden.mid"
        if not path.exists():
            self.skipTest("grape-garden.mid not found")
        a = self._analysis(path)
        # Grape Garden is in C#/Db major (the MIDI prominently uses F-natural = E#,
        # the 3rd degree of C# major, not E-natural which B major would require).
        self.assertIn(a.key, ("C#", "B"))
        self.assertEqual(a.time_signature[0], 3)
        self.assertGreater(len(a.sections), 5)

    def test_to_json_roundtrip(self):
        from music_generator.ir import MusicalAnalysis
        import json
        a    = self._analysis(_simple_c_major_midi(phrases=8))
        j    = a.to_json()
        data = json.loads(j)
        a2   = MusicalAnalysis.from_dict(data)
        self.assertEqual(a.key,      a2.key)
        self.assertEqual(a.tempo_bpm, a2.tempo_bpm)
        self.assertEqual(len(a.sections), len(a2.sections))


if __name__ == "__main__":
    unittest.main(verbosity=2)
