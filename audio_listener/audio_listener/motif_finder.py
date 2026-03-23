"""
audio_listener.motif_finder — Musical pattern recognition.

Detects three classes of recurring musical patterns:

  motif      — Short interval cell (3–7 notes) recurring across ≥ N phrases.
               Transposition-invariant; inverted/retrograde/augmented occurrences
               are tracked as transform labels on each occurrence record.

  sequence   — A motif repeated at a consistent transposed pitch level 2+ times
               within a single phrase (harmonic sequence).  Stored as a MotifDef
               with type="sequence" and sequence_step set.

  cadential  — Short 2–3 note figure appearing at the END of ≥ 2 phrases
               (closing gestures, approach to tonic).
"""

from __future__ import annotations

from audio_listener.pitch_tracker    import TrackedNote
from audio_listener.phrase_segmenter import PhraseSegment


# ── Utility helpers ────────────────────────────────────────────────────────────

def _contour(intervals: list[int]) -> list[int]:
    """Sign-pattern: +1 up, -1 down, 0 same."""
    return [1 if i > 0 else (-1 if i < 0 else 0) for i in intervals]


def _rhythm_class(durations: list[float]) -> str:
    """
    Classify a duration list into one of four rhythm classes.

    even       — all durations within 20 % of each other
    dotted     — adjacent 3:2 ratio (dotted-eighth/sixteenth, etc.)
    syncopated — prominent short-long-short stress
    mixed      — everything else
    """
    if len(durations) < 2:
        return "even"
    ratios = [durations[i] / durations[i + 1]
              for i in range(len(durations) - 1)
              if durations[i + 1] > 1e-9]
    if not ratios:
        return "mixed"
    mn, mx = min(ratios), max(ratios)
    if mx / max(mn, 1e-9) < 1.25:
        return "even"
    if any(abs(r - 1.5) < 0.22 or abs(r - 0.667) < 0.15 for r in ratios):
        return "dotted"
    for i in range(len(durations) - 2):
        if durations[i] < durations[i + 1] > durations[i + 2]:
            return "syncopated"
    return "mixed"


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    return s[len(s) // 2] if s else 0.25


def _is_aug_dim(canon: list[float], other: list[float], tol: float = 0.18) -> float | None:
    """
    Return 2.0 (augmentation) or 0.5 (diminution) if other ≈ factor * canon,
    else None.
    """
    if len(canon) != len(other):
        return None
    for factor in (2.0, 0.5):
        if all(abs(b - factor * a) <= tol * max(a, b, 1e-9)
               for a, b in zip(canon, other)):
            return factor
    return None


# ── Core detection ────────────────────────────────────────────────────────────

def find_motifs(
    melody:          list[TrackedNote],
    phrases:         list[PhraseSegment],
    min_occurrences: int = 3,
    min_len:         int = 3,
    max_len:         int = 7,
) -> list:
    """
    Analyse melody and return a list[MotifDef] covering motifs, sequences,
    and cadential figures found.  Results are ordered by occurrence count
    descending.

    Each MotifDef carries extended metadata:
      contour      — sign-pattern of intervals (+1/0/-1 per interval)
      occurrences  — list of {phrase_idx, beat_offset, transform} records
                     where transform is one of:
                       "original"              — exact match
                       "inverted"              — all intervals negated
                       "retrograde"            — intervals reversed
                       "retrograde_inversion"  — reversed + negated
                       "augmented"             — same intervals, 2× durations
                       "diminuted"             — same intervals, ½ durations
                       "seq(+N)"               — sequential copy offset +N st
      sequence_step — semitone transposition per repetition (sequences only)
      rhythm_class  — "even" | "dotted" | "syncopated" | "mixed"
    """
    from music_generator.ir import MotifDef

    # ── Build phrase-indexed note lists ─────────────────────────────────────
    phrase_notes: list[list[TrackedNote]] = [
        sorted(
            [n for n in melody if ph.start_beat <= n.start_beat < ph.end_beat],
            key=lambda n: n.start_beat,
        )
        for ph in phrases
    ]

    # ── Build pattern map: interval-tuple → [(phrase_idx, notes, durs)] ─────
    # pattern_map[intervals_tuple] = list of (phrase_idx, sub_notes, dur_list)
    pattern_map: dict[tuple, list] = {}

    for ph_idx, pnotes in enumerate(phrase_notes):
        n = len(pnotes)
        for start in range(n):
            for length in range(min_len, min(max_len + 1, n - start + 1)):
                sub    = pnotes[start:start + length]
                pts    = [s.pitch for s in sub]
                consec = tuple(pts[i + 1] - pts[i] for i in range(len(pts) - 1))
                durs   = [s.duration_beats for s in sub]
                pattern_map.setdefault(consec, []).append((ph_idx, sub, durs))

    # ── Occurrence builder: canonical + variants ──────────────────────────
    def _all_occurrences(pattern: tuple, canon_durs: list[float]) -> list[dict]:
        inv     = tuple(-i for i in pattern)
        ret     = tuple(reversed(pattern))
        ret_inv = tuple(-i for i in ret)
        variants = {
            pattern: "original",
            inv:     "inverted",
            ret:     "retrograde",
            ret_inv: "retrograde_inversion",
        }
        out: list[dict] = []
        for vkey, label in variants.items():
            for ph_idx, sub, durs in pattern_map.get(vkey, []):
                aug = _is_aug_dim(canon_durs, durs)
                suffix = ""
                if aug == 2.0:
                    suffix = "(aug)"
                elif aug == 0.5:
                    suffix = "(dim)"
                out.append({
                    "phrase_idx":  ph_idx,
                    "beat_offset": sub[0].start_beat if sub else 0.0,
                    "transform":   label + suffix,
                })
        return out

    # ── Step 1: Exact-match recurring motifs ─────────────────────────────────
    def _motif_candidates(min_occ: int) -> list[tuple]:
        results = []
        for pattern, occs in pattern_map.items():
            phrase_set = {o[0] for o in occs}
            if len(phrase_set) >= min_occ:
                results.append((pattern, occs, len(phrase_set)))

        # Keep longest pattern that subsumes a shorter one's phrase coverage
        results.sort(key=lambda x: -len(x[0]))
        filtered, seen = [], []
        for pat, occ, cnt in results:
            occ_set = frozenset(o[0] for o in occ)
            if not any(occ_set <= prev for prev in seen):
                filtered.append((pat, occ, cnt))
                seen.append(occ_set)
        filtered.sort(key=lambda x: -x[2])
        return filtered

    candidates = _motif_candidates(min_occurrences)
    if not candidates and min_occurrences > 2:
        candidates = _motif_candidates(2)

    # ── Step 2: Harmonic sequence detection ──────────────────────────────────
    def _find_sequences() -> list[dict]:
        """
        Within each phrase, find the same interval pattern appearing 2+ times
        consecutively at a consistent transposition step.
        """
        found = []
        for pattern, occs in pattern_map.items():
            by_phrase: dict[int, list] = {}
            for ph_idx, sub, durs in occs:
                by_phrase.setdefault(ph_idx, []).append((sub[0].start_beat, sub, durs))

            for ph_idx, entries in by_phrase.items():
                entries.sort(key=lambda e: e[0])
                if len(entries) < 2:
                    continue
                steps = [
                    entries[i + 1][1][0].pitch - entries[i][1][0].pitch
                    for i in range(len(entries) - 1)
                ]
                if len(set(steps)) == 1 and steps[0] != 0:
                    found.append({
                        "pattern":       pattern,
                        "phrase_idx":    ph_idx,
                        "step":          steps[0],
                        "n_reps":        len(entries),
                        "occurrences": [
                            {
                                "phrase_idx":  ph_idx,
                                "beat_offset": e[0],
                                "transform":   f"seq(+{steps[0]})",
                            }
                            for e in entries
                        ],
                        "ref_durs": entries[0][2],
                    })
        return found

    sequences = _find_sequences()

    # ── Step 3: Cadential figure detection ───────────────────────────────────
    def _find_cadential() -> list[tuple]:
        cad_map: dict[tuple, list] = {}
        for ph_idx, pnotes in enumerate(phrase_notes):
            if len(pnotes) < 2:
                continue
            for tail_len in (2, 3):
                tail = pnotes[-tail_len:]
                if len(tail) < 2:
                    continue
                pts    = [n.pitch for n in tail]
                consec = tuple(pts[i + 1] - pts[i] for i in range(len(pts) - 1))
                durs   = [n.duration_beats for n in tail]
                cad_map.setdefault(consec, []).append((ph_idx, tail, durs))
        return [
            (pat, occs)
            for pat, occs in cad_map.items()
            if len({o[0] for o in occs}) >= 2
        ]

    cadential = _find_cadential()

    # ── Step 4: Assemble MotifDef list ───────────────────────────────────────
    motifs: list    = []
    used:   set     = set()

    # Primary interval motifs
    for i, (pattern, occ_list, _cnt) in enumerate(candidates[:4]):
        used.add(pattern)
        n_notes   = len(pattern) + 1
        dur_lists = [[] for _ in range(n_notes)]
        for _ph, _sub, durs in occ_list[:20]:
            for j, d in enumerate(durs):
                dur_lists[j].append(d)
        durations  = [_median(dl) for dl in dur_lists]
        intervals  = list(pattern)
        occ_data   = _all_occurrences(pattern, durations)

        motifs.append(MotifDef(
            id           = f"motif_{i}",
            type         = "motif",
            intervals    = intervals,
            durations    = durations,
            contour      = _contour(intervals),
            occurrences  = occ_data,
            rhythm_class = _rhythm_class(durations),
        ))

    # Harmonic sequences (skip if already captured as a motif)
    seq_idx = 0
    for seq in sequences[:2]:
        pat = seq["pattern"]
        if pat in used:
            continue
        used.add(pat)
        n_notes   = len(pat) + 1
        dur_lists = [[] for _ in range(n_notes)]
        for _ph, _sub, durs in pattern_map.get(pat, [])[:10]:
            for j, d in enumerate(durs):
                dur_lists[j].append(d)
        durations = [_median(dl) for dl in dur_lists]

        motifs.append(MotifDef(
            id            = f"seq_{seq_idx}",
            type          = "sequence",
            intervals     = list(pat),
            durations     = durations,
            contour       = _contour(list(pat)),
            occurrences   = seq["occurrences"],
            sequence_step = seq["step"],
            rhythm_class  = _rhythm_class(durations),
        ))
        seq_idx += 1

    # Cadential figures (skip already-used patterns and very common short steps)
    cad_idx = 0
    for pat, occ_list in cadential[:2]:
        if pat in used or len(pat) > 3:
            continue
        used.add(pat)
        n_notes   = len(pat) + 1
        dur_lists = [[] for _ in range(n_notes)]
        for _ph, sub, durs in occ_list[:10]:
            for j, d in enumerate(durs):
                dur_lists[j].append(d)
        durations = [_median(dl) for dl in dur_lists]

        motifs.append(MotifDef(
            id           = f"cadence_{cad_idx}",
            type         = "cadential",
            intervals    = list(pat),
            durations    = durations,
            contour      = _contour(list(pat)),
            occurrences  = [
                {
                    "phrase_idx":  o[0],
                    "beat_offset": o[1][-1].start_beat if o[1] else 0.0,
                    "transform":   "cadential",
                }
                for o in occ_list
            ],
            rhythm_class = _rhythm_class(durations),
        ))
        cad_idx += 1

    return motifs
