"""
audio_listener.motif_finder — Recurring interval-sequence motif detection.
"""

from __future__ import annotations

from audio_listener.pitch_tracker   import TrackedNote
from audio_listener.phrase_segmenter import PhraseSegment


def find_motifs(
    melody:  list[TrackedNote],
    phrases: list[PhraseSegment],
    min_occurrences: int = 3,
    min_len: int = 3,
    max_len: int = 6,
) -> list:
    """
    Find recurring interval-sequence motifs in the melody.

    Uses consecutive semitone intervals as the pattern key — transposition-
    invariant by construction (octave transpositions change the key, same
    melodic shape in any key matches).

    Returns list[MotifDef], ordered by occurrence count descending.
    Falls back to min_occurrences=2 if none found at the given threshold.
    """
    from music_generator.ir import MotifDef

    # Build phrase-indexed note lists
    phrase_notes: list[list[TrackedNote]] = []
    for ph in phrases:
        pn = [n for n in melody if ph.start_beat <= n.start_beat < ph.end_beat]
        pn.sort(key=lambda n: n.start_beat)
        phrase_notes.append(pn)

    def _extract_patterns(min_occ: int):
        # pattern_key (consecutive intervals) → list of (phrase_idx, note_sub_list)
        pattern_map: dict[tuple, list] = {}

        for ph_idx, pnotes in enumerate(phrase_notes):
            for start in range(len(pnotes)):
                for length in range(min_len, min(max_len + 1, len(pnotes) - start + 1)):
                    sub = pnotes[start:start + length]
                    # Consecutive semitone intervals — transposition-invariant
                    pitches = [n.pitch for n in sub]
                    consec  = tuple(pitches[i+1] - pitches[i] for i in range(len(pitches) - 1))
                    pattern_map.setdefault(consec, []).append((ph_idx, sub))

        # Filter by min occurrences across distinct phrases
        results = []
        for pattern, occurrences in pattern_map.items():
            phrase_set = {o[0] for o in occurrences}
            if len(phrase_set) >= min_occ:
                results.append((pattern, occurrences, len(phrase_set)))

        # Remove sub-patterns: if pattern A is a prefix of B with the same occurrence set,
        # keep only B
        results.sort(key=lambda x: -len(x[0]))
        filtered = []
        seen_occ_sets = []
        for pat, occ, cnt in results:
            occ_set = frozenset(o[0] for o in occ)
            is_subsumed = any(occ_set <= prev for prev in seen_occ_sets)
            if not is_subsumed:
                filtered.append((pat, occ, cnt))
                seen_occ_sets.append(occ_set)

        return filtered

    candidates = _extract_patterns(min_occurrences)
    if not candidates and min_occurrences > 2:
        candidates = _extract_patterns(2)
    if not candidates:
        return []

    # Sort by occurrence count descending
    candidates.sort(key=lambda x: -x[2])

    motifs: list = []
    for i, (pattern, occurrences, _cnt) in enumerate(candidates[:5]):
        # pattern is already consecutive intervals; n_notes = len(intervals) + 1
        n_notes = len(pattern) + 1
        dur_lists: list[list[float]] = [[] for _ in range(n_notes)]
        for _ph_idx, sub in occurrences[:20]:   # cap for speed
            for j, note in enumerate(sub):
                dur_lists[j].append(note.duration_beats)

        def _median(vals: list[float]) -> float:
            s = sorted(vals)
            return s[len(s) // 2] if s else 0.25

        durations = [_median(dl) for dl in dur_lists]
        intervals = list(pattern)   # already consecutive semitone intervals

        motifs.append(MotifDef(
            id=f"motif_{i}",
            type="motif",
            intervals=intervals,
            durations=durations,
        ))

    return motifs
