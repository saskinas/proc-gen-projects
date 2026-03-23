"""
audio_listener.chord_builder — Build HarmonicEvent sequences from sparse NES voices.

Uses the bass as a root hint and melody/countermelody as chord tones.
Imports HarmonicEvent, CHORD_INTERVALS, SCALE_INTERVALS from music_generator.
"""

from __future__ import annotations

from audio_listener.pitch_tracker import TrackedNote


def _import_music_gen():
    from music_generator import HarmonicEvent, CHORD_INTERVALS, SCALE_INTERVALS, SCALE_INTERVALS as SI
    return HarmonicEvent, CHORD_INTERVALS, SCALE_INTERVALS


def _snap_to_rhythm(ioi: float, beats_per_bar: int) -> float:
    """Snap an IOI to the nearest musically meaningful chord duration."""
    candidates = [0.5, 1.0, 2.0, float(beats_per_bar), float(beats_per_bar * 2)]
    for c in candidates:
        if abs(ioi - c) <= c * 0.4:
            return c
    return float(beats_per_bar)


def detect_harmonic_rhythm(
    melody: list[TrackedNote],
    beats_per_bar: int,
    bass: list[TrackedNote] | None = None,
    counter: list[TrackedNote] | None = None,
) -> float:
    """
    Estimate chord duration in beats. Returns one of: 0.5, 1.0, 2.0, beats_per_bar,
    beats_per_bar*2.

    Strategy (in priority order):
    1. Bass note IOI — most reliable harmonic signal; immune to arpeggio churn.
    2. Melody note IOI — when bass is too sparse (< 4 notes).
    3. Fallback: one chord per bar.

    Counter/arpeggio voices are intentionally excluded because their rapid
    pitch-class cycling would otherwise give 0.5-beat harmonic rhythm.
    """
    def _median_ioi(notes: list[TrackedNote]) -> float | None:
        if len(notes) < 4:
            return None
        sorted_n = sorted(notes, key=lambda n: n.start_beat)
        iois = [sorted_n[i+1].start_beat - sorted_n[i].start_beat
                for i in range(len(sorted_n) - 1)
                if sorted_n[i+1].start_beat - sorted_n[i].start_beat > 0.01]
        if len(iois) < 2:
            return None
        return sorted(iois)[len(iois) // 2]

    # 1. Bass IOI
    if bass:
        m = _median_ioi(bass)
        if m is not None:
            return _snap_to_rhythm(m, beats_per_bar)

    # 2. Melody IOI
    if melody:
        m = _median_ioi(melody)
        if m is not None:
            return _snap_to_rhythm(m, beats_per_bar)

    return float(beats_per_bar)


def _notes_in_window(
    notes: list[TrackedNote],
    start: float,
    end:   float,
    min_overlap: float = 0.25,
) -> list[TrackedNote]:
    """Notes whose active interval overlaps [start, end) by at least min_overlap of window width."""
    width = end - start
    threshold = max(0.0, width * min_overlap)
    out = []
    for n in notes:
        n_start = n.start_beat
        n_end   = n.start_beat + n.duration_beats
        overlap = max(0.0, min(n_end, end) - max(n_start, start))
        if overlap >= threshold or overlap >= n.duration_beats * 0.5:
            out.append(n)
    return out


def _identify_chord(
    pitch_classes: list[int],
    root_pc: int,
    chord_intervals: dict,
) -> str:
    """
    Find the chord_type from CHORD_INTERVALS that best fits observed pitch classes.
    NES: limit to simple types. Returns 'maj' on failure.

    sus4 and aug are penalised so that functional chord types (maj/min/dom7)
    win when evidence is ambiguous (e.g. root + fifth only, or sparse arpeggio).
    """
    # Candidate chord types; list order is tiebreak when scores are equal
    candidates = ["maj", "min", "dom7", "dim", "min7", "maj7", "sus4", "aug"]
    # Non-functional types need clearly better coverage to override functional ones
    _PENALTY   = {"sus4": 0.88, "aug": 0.88, "dim": 0.93}
    pc_set = set(pitch_classes)

    best_type  = "maj"
    best_score = -1.0

    for ctype in candidates:
        ivs       = chord_intervals.get(ctype, [0, 4, 7])
        chord_pcs = {(root_pc + iv) % 12 for iv in ivs}
        if not chord_pcs:
            continue
        matched   = len(pc_set & chord_pcs)
        precision = matched / len(pc_set) if pc_set else 0.0
        recall    = matched / len(chord_pcs)
        score     = (0.5 * precision + 0.5 * recall) * _PENALTY.get(ctype, 1.0)
        if score > best_score:
            best_score = score
            best_type  = ctype

    return best_type


def _pc_to_degree(root_pc: int, tonic_pc: int, mode: str, scale_intervals: dict) -> int:
    """Map a pitch class to a scale degree (1-7) in the given tonic/mode."""
    ivs = scale_intervals.get(mode, scale_intervals["major"])
    interval = (root_pc - tonic_pc) % 12
    # Find the closest scale degree
    best_deg  = 1
    best_dist = 12
    for deg_idx, iv in enumerate(ivs):
        dist = abs(interval - iv)
        if dist < best_dist:
            best_dist = dist
            best_deg  = deg_idx + 1
    return best_deg


def build_harmonic_plan(
    melody:     list[TrackedNote],
    bass:       list[TrackedNote],
    counter:    list[TrackedNote],
    tonic_pc:   int,
    mode:       str,
    harm_rhythm: float,
    section_start_beat: float,
    section_end_beat:   float,
    phrase_map:  list[tuple[float, float, int]],  # (phrase_start, phrase_end, phrase_idx)
) -> list:
    """
    Build a list[HarmonicEvent] for a section.

    Args:
        melody/bass/counter  — note sequences (beats relative to piece start)
        tonic_pc             — pitch class of the key tonic (0-11)
        mode                 — scale mode string
        harm_rhythm          — chord duration in beats
        section_start/end    — beat range of the section
        phrase_map           — (start, end, idx) for each phrase in the section

    Returns list[HarmonicEvent] with consecutive identical chords merged.
    """
    HarmonicEvent, CHORD_INTERVALS, SCALE_INTERVALS = _import_music_gen()

    events: list = []
    cursor = section_start_beat

    while cursor < section_end_beat - 0.01:
        win_end = min(cursor + harm_rhythm, section_end_beat)

        # Collect active pitch classes in window
        active_notes = (
            _notes_in_window(melody,  cursor, win_end) +
            _notes_in_window(bass,    cursor, win_end) +
            _notes_in_window(counter, cursor, win_end)
        )

        if not active_notes:
            # Rest window: use tonic
            root_pc   = tonic_pc
            chord_type = "maj" if "major" in mode else "min"
        else:
            # Root hint: bass note that sounds most in this window
            bass_in_win = _notes_in_window(bass, cursor, win_end)
            if bass_in_win:
                # Use the bass note with greatest overlap as root;
                # tiebreak: prefer lower pitch as bass root
                root_pc = max(
                    bass_in_win,
                    key=lambda n: (
                        min(n.start_beat + n.duration_beats, win_end) - max(n.start_beat, cursor),
                        -n.pitch,
                    ),
                ).pitch % 12
            else:
                # No bass: use lowest active pitch class
                root_pc = min(active_notes, key=lambda n: n.pitch).pitch % 12

            pcs = list({n.pitch % 12 for n in active_notes})
            chord_type = _identify_chord(pcs, root_pc, CHORD_INTERVALS)

        # Determine phrase_idx for this window
        phrase_idx = 0
        for (ps, pe, pi) in phrase_map:
            if ps <= cursor < pe:
                phrase_idx = pi
                break

        duration = win_end - cursor

        # Merge with previous event if same chord
        if (events and
                events[-1].root_pc == root_pc and
                events[-1].chord_type == chord_type and
                events[-1].phrase_idx == phrase_idx):
            # Extend last event
            last = events[-1]
            events[-1] = HarmonicEvent(
                root_pc=last.root_pc,
                chord_type=last.chord_type,
                duration_beats=last.duration_beats + duration,
                degree=last.degree,
                phrase_idx=last.phrase_idx,
            )
        else:
            degree = _pc_to_degree(root_pc, tonic_pc, mode, SCALE_INTERVALS)
            events.append(HarmonicEvent(
                root_pc=root_pc,
                chord_type=chord_type,
                duration_beats=duration,
                degree=degree,
                phrase_idx=phrase_idx,
            ))

        cursor = win_end

    return events
