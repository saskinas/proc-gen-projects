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
    scale_pcs: set[int] | None = None,
) -> str:
    """
    Find the chord_type from CHORD_INTERVALS that best fits observed pitch classes.
    NES: limit to simple types. Returns 'maj' on failure.

    sus4 and aug are penalised so that functional chord types (maj/min/dom7)
    win when evidence is ambiguous (e.g. root + fifth only, or sparse arpeggio).

    When scale_pcs is provided, chord types that introduce non-scale notes are
    penalised further so that passing tones don't flip V→Vm, etc.
    """
    # Candidate chord types; list order is tiebreak when scores are equal
    candidates = ["maj", "min", "dom7", "dim", "min7", "maj7", "sus4", "aug"]
    # Non-functional types need clearly better coverage to override functional ones
    _PENALTY   = {"sus4": 0.82, "aug": 0.75, "dim": 0.90}
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
        # Extra penalty when this chord type introduces non-diatonic notes.
        # Prevents passing tones (C natural over V root in major key) from
        # flipping V→Vm, etc.
        if scale_pcs:
            non_diatonic = chord_pcs - scale_pcs
            if non_diatonic:
                score *= 0.80
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
    scale_ivs = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])
    scale_pcs = {(tonic_pc + iv) % 12 for iv in scale_ivs}

    events: list = []
    cursor = section_start_beat
    last_bass_pc: int | None = None  # carry-forward: last heard bass root

    # Build expected diatonic chord types for each degree index in the scale.
    # Used as a tiebreak when evidence is ambiguous.
    _DIATONIC_TYPES_MAJOR = {0: "maj", 1: "min", 2: "min", 3: "maj",
                              4: "maj", 5: "min", 6: "dim"}
    _DIATONIC_TYPES_MINOR = {0: "min", 1: "dim", 2: "maj", 3: "min",
                              4: "min", 5: "maj", 6: "maj"}
    _diatonic_types = (
        _DIATONIC_TYPES_MAJOR if "major" in mode else _DIATONIC_TYPES_MINOR
    )
    _iv_to_deg_idx = {iv: idx for idx, iv in enumerate(scale_ivs)}

    while cursor < section_end_beat - 0.01:
        win_end = min(cursor + harm_rhythm, section_end_beat)
        win_width = win_end - cursor

        # Collect active pitch classes in window, weighted by overlap duration.
        # Short passing tones (< 5% of chord window) are excluded so that NES
        # arpeggio churn and melodic ornaments don't corrupt chord detection.
        def _weighted_pcs(notes_list: list, min_frac: float = 0.05) -> list[int]:
            from collections import defaultdict
            pc_dur: dict[int, float] = defaultdict(float)
            for n in notes_list:
                n_end   = n.start_beat + n.duration_beats
                overlap = max(0.0, min(n_end, win_end) - max(n.start_beat, cursor))
                if overlap > 0 and win_width > 0:
                    pc_dur[n.pitch % 12] += overlap / win_width
            return [pc for pc, frac in pc_dur.items() if frac >= min_frac]

        # Only use melody + bass for pitch-class collection.
        # The countermelody (counter) often plays fast arpeggios and passing
        # tones that corrupt chord detection (e.g. a C-natural run over an A
        # bass in D major registers as Am instead of A).  Bass is the
        # strongest tonal anchor; melody provides the chord quality.
        mel_notes  = _notes_in_window(melody, cursor, win_end)
        bass_in_win = _notes_in_window(bass, cursor, win_end)
        mel_bass_notes = mel_notes + bass_in_win

        all_active = mel_bass_notes + (
            _notes_in_window(counter, cursor, win_end) if counter else []
        )

        if not mel_bass_notes and not all_active:
            # Rest window: carry forward last bass, or use tonic
            root_pc    = last_bass_pc if last_bass_pc is not None else tonic_pc
            chord_type = _diatonic_types.get(
                _iv_to_deg_idx.get((root_pc - tonic_pc) % 12, 0), "maj"
            )
        else:
            # Root hint: bass note that sounds most in this window.
            if bass_in_win:
                root_pc = max(
                    bass_in_win,
                    key=lambda n: (
                        min(n.start_beat + n.duration_beats, win_end) - max(n.start_beat, cursor),
                        -n.pitch,
                    ),
                ).pitch % 12
                last_bass_pc = root_pc
            elif last_bass_pc is not None:
                # Bass silent: carry forward last heard root rather than
                # guessing from melody (which often lands on non-root notes).
                root_pc = last_bass_pc
            elif mel_bass_notes:
                root_pc = min(mel_bass_notes, key=lambda n: n.pitch).pitch % 12
            else:
                root_pc = min(all_active, key=lambda n: n.pitch).pitch % 12

            # Weight pitch classes by overlap duration (melody + bass only).
            # Exclude brief passing tones (< 5% of chord window).
            notes_for_pcs = mel_bass_notes if mel_bass_notes else all_active
            pcs = _weighted_pcs(notes_for_pcs)
            if not pcs:
                pcs = list({n.pitch % 12 for n in notes_for_pcs})
            chord_type = _identify_chord(pcs, root_pc, CHORD_INTERVALS, scale_pcs)

            # Diatonic preference: when the identified chord type is non-diatonic
            # (e.g. dom7 where we expect maj, or min where we expect maj) and the
            # root IS on a diatonic scale degree, prefer the diatonic type unless
            # the non-diatonic type clearly fits much better (>20% score margin).
            rel_iv = (root_pc - tonic_pc) % 12
            if rel_iv in _iv_to_deg_idx:
                deg_idx      = _iv_to_deg_idx[rel_iv]
                expected_type = _diatonic_types.get(deg_idx, chord_type)
                if chord_type != expected_type:
                    # Only keep non-diatonic if it has a major scale-tone penalty;
                    # otherwise snap back to the expected diatonic type.
                    ivs_id = CHORD_INTERVALS.get(chord_type, [0, 4, 7])
                    non_dia = {(root_pc + iv) % 12 for iv in ivs_id} - scale_pcs
                    if non_dia:
                        chord_type = expected_type

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
