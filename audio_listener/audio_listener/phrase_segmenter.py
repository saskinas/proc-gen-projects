"""
audio_listener.phrase_segmenter — Phrase and section boundary detection.

Divides a piece into fixed-length phrases, then groups similar phrases
into sections using melodic fingerprint comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from audio_listener.pitch_tracker import TrackedNote


@dataclass
class PhraseSegment:
    phrase_idx:          int
    start_beat:          float
    end_beat:            float
    bar_count:           int
    melody_fingerprint:  tuple   # interval sequence (transposition-invariant)


@dataclass
class SectionBoundary:
    section_idx:   int
    start_phrase:  int           # index into phrase list
    end_phrase:    int           # exclusive
    start_beat:    float
    end_beat:      float
    is_repeat_of:  int | None    # section_idx of earlier matching section


def _compute_fingerprint(
    notes: list[TrackedNote],
    start: float,
    end:   float,
    max_pcs: int = 16,
) -> tuple:
    """
    Ordered pitch-class sequence within [start, end), capped at max_pcs.
    Transposition-invariant: normalised so first pc = 0.
    """
    in_window = [n for n in notes if start <= n.start_beat < end]
    in_window.sort(key=lambda n: n.start_beat)
    pcs = [n.pitch % 12 for n in in_window[:max_pcs]]
    if not pcs:
        return ()
    offset = pcs[0]
    return tuple((p - offset) % 12 for p in pcs)


def _fingerprint_distance(a: tuple, b: tuple) -> int:
    """
    Number of differing pitch-class positions (up to min length).
    Returns large number if lengths differ significantly.
    """
    if not a or not b:
        return 99
    min_len = min(len(a), len(b))
    if min_len < 3:
        return 99
    length_penalty = abs(len(a) - len(b))
    mismatches = sum(1 for i in range(min_len) if a[i] != b[i])
    return mismatches + length_penalty


def segment_phrases(
    melody_notes: list[TrackedNote],
    beats_per_bar: int,
    phrase_bars: int = 4,
) -> list[PhraseSegment]:
    """
    Divide the melody into fixed-length phrases of phrase_bars bars.

    phrase_bars is auto-detected: tries 4, 8, 2, then defaults to 4.
    """
    if not melody_notes:
        return []

    total_beats = max(n.start_beat + n.duration_beats for n in melody_notes)
    bar_beats   = float(beats_per_bar)

    # Detect phrase_bars.  Target at least 6 phrases so the section grouper
    # has enough material to identify structure.  Pick the largest candidate
    # where total_bars / candidate >= 6 (i.e. total_bars >= candidate * 6).
    #
    # Examples:
    #   128-bar piece → 16-bar phrases (8 phrases)
    #    32-bar piece →  4-bar phrases (8 phrases)
    #     8-bar piece →  2-bar phrases (4 phrases, else branch)
    total_bars = total_beats / bar_beats
    for candidate in [16, 8, 4, 2]:
        if total_bars >= candidate * 6:
            phrase_bars = candidate
            break
    else:
        phrase_bars = max(1, int(total_bars / 2))

    phrase_beats = bar_beats * phrase_bars
    phrases: list[PhraseSegment] = []
    idx = 0
    pos = 0.0

    while pos < total_beats - bar_beats * 0.5:
        end = min(pos + phrase_beats, total_beats)
        fp  = _compute_fingerprint(melody_notes, pos, end)
        phrases.append(PhraseSegment(
            phrase_idx=idx,
            start_beat=pos,
            end_beat=end,
            bar_count=phrase_bars,
            melody_fingerprint=fp,
        ))
        idx += 1
        pos += phrase_beats

    # Drop a stub tail phrase shorter than 1.5 bars — it is not worth its own
    # section and its chord detection would be unreliable.
    if len(phrases) > 1:
        last = phrases[-1]
        if (last.end_beat - last.start_beat) < bar_beats * 1.5:
            phrases.pop()

    return phrases


def detect_sections(
    phrases: list[PhraseSegment],
    similarity_threshold: int = 3,
    min_section_phrases: int = 1,
) -> list[SectionBoundary]:
    """
    Group phrases into sections using fingerprint similarity.

    Two phrases are 'similar' if their fingerprint distance <= similarity_threshold.
    Consecutive similar phrases are grouped.  Earlier sections are matched for
    repeat detection.
    """
    if not phrases:
        return []

    sections: list[SectionBoundary] = []
    sec_idx   = 0
    i         = 0

    while i < len(phrases):
        # Find the longest run of similar phrases starting at i
        j = i + 1
        while j < len(phrases):
            d = _fingerprint_distance(phrases[i].melody_fingerprint,
                                      phrases[j].melody_fingerprint)
            if d > similarity_threshold:
                break
            j += 1

        if j - i < min_section_phrases:
            j = i + 1   # force at least 1 phrase per section

        # Check if this fingerprint set matches any earlier section
        current_fps = {phrases[k].melody_fingerprint for k in range(i, j)}
        is_repeat   = None
        for prev_sec in sections:
            prev_fps = {
                phrases[k].melody_fingerprint
                for k in range(prev_sec.start_phrase, prev_sec.end_phrase)
                if k < len(phrases)
            }
            shared = sum(
                1 for fp in current_fps
                if any(_fingerprint_distance(fp, pfp) <= similarity_threshold
                       for pfp in prev_fps)
            )
            if shared >= max(1, len(current_fps) // 2):
                is_repeat = prev_sec.section_idx
                break

        sections.append(SectionBoundary(
            section_idx=sec_idx,
            start_phrase=i,
            end_phrase=j,
            start_beat=phrases[i].start_beat,
            end_beat=phrases[j-1].end_beat,
            is_repeat_of=is_repeat,
        ))
        sec_idx += 1
        i = j

    return sections
