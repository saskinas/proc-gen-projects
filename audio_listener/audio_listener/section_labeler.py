"""
audio_listener.section_labeler — Assign role and label to each section.
"""

from __future__ import annotations

from audio_listener.phrase_segmenter import PhraseSegment, SectionBoundary
from audio_listener.pitch_tracker   import TrackedNote


def _section_density(
    section: SectionBoundary,
    melody:  list[TrackedNote],
) -> float:
    """Notes per beat in the section's beat range."""
    notes = [n for n in melody
             if section.start_beat <= n.start_beat < section.end_beat]
    dur = section.end_beat - section.start_beat
    return len(notes) / max(dur, 1.0)


def label_sections(
    sections:    list[SectionBoundary],
    phrases:     list[PhraseSegment],
    melody:      list[TrackedNote],
    total_beats: float,
) -> list[tuple[str, str]]:
    """
    Returns list of (role, label) for each section.

    NES heuristics:
    - First section + short + non-repeat → 'intro'
    - Last section + density drops → 'outro'
    - Most-repeated fingerprint group → 'chorus'
    - Sections between choruses → 'verse'
    - Single-occurrence high-energy section → 'bridge'
    - Repeats of an earlier section → inherit the original's role
    - Default: 'generic'
    """
    n = len(sections)
    if n == 0:
        return []

    densities = [_section_density(s, melody) for s in sections]
    max_dens  = max(densities) if densities else 1.0
    if max_dens < 1e-9:
        max_dens = 1.0   # avoid division-by-zero / degenerate threshold when melody is silent

    # Count how many times each section_idx is referenced as is_repeat_of
    repeat_counts: dict[int | None, int] = {}
    for s in sections:
        repeat_counts[s.is_repeat_of] = repeat_counts.get(s.is_repeat_of, 0) + 1

    # The original section with most repeats is the 'chorus'
    chorus_original: int | None = None
    max_repeats = 0
    for orig_idx, cnt in repeat_counts.items():
        if orig_idx is not None and cnt > max_repeats:
            max_repeats   = cnt
            chorus_original = orig_idx

    roles:  list[str] = ["generic"] * n
    labels: list[str] = []

    # Mark chorus and its repeats
    for i, s in enumerate(sections):
        if s.section_idx == chorus_original:
            roles[i] = "chorus"
        elif s.is_repeat_of == chorus_original:
            roles[i] = "chorus"

    # Determine if this is a highly repetitive piece (most sections share
    # the same repeat fingerprint, as happens with looping game music).
    chorus_repeat_count = sum(
        1 for s in sections
        if s.is_repeat_of == chorus_original or s.section_idx == chorus_original
    ) if chorus_original is not None else 0
    is_highly_repetitive = chorus_repeat_count >= n * 0.6

    if is_highly_repetitive:
        # Every section is a "chorus repeat" — give the first pass an intro
        # label and the last pass an outro label when density drops, so the
        # generator can provide a distinct opening and closing.
        roles[0] = "intro"
        if n > 1 and densities[n-1] <= max_dens * 0.75:
            roles[n-1] = "outro"
    else:
        # Standard heuristics for pieces with genuine section variety.
        # Intro: first section, not chorus, short, early in piece.
        if roles[0] != "chorus":
            n_phrases_0 = sections[0].end_phrase - sections[0].start_phrase
            if n_phrases_0 <= 2 and sections[0].end_beat <= total_beats * 0.25:
                roles[0] = "intro"
        # Outro: last section, not chorus, density significantly drops.
        if roles[n-1] not in ("chorus",) and n > 1:
            if densities[n-1] <= max_dens * 0.6:
                roles[n-1] = "outro"

    # Bridge: unique section whose density is a clear outlier above the average.
    # Require it to sit in the middle 15–90 % of the piece (not intro/outro territory)
    # and limit total bridges to avoid labelling every section as bridge.
    avg_dens   = sum(densities) / len(densities) if densities else 0.0
    variance   = sum((d - avg_dens) ** 2 for d in densities) / len(densities) if densities else 0.0
    stdev_dens = variance ** 0.5
    # Must be above average + half a standard deviation, and at least 70 % of max
    bridge_threshold = max(avg_dens + stdev_dens * 0.5, max_dens * 0.70)
    max_bridges  = max(1, n // 5)   # e.g. ≤ 2 bridges for a 10-section piece
    bridge_count = 0

    for i, s in enumerate(sections):
        if roles[i] == "generic" and s.is_repeat_of is None and bridge_count < max_bridges:
            position_frac = s.start_beat / max(total_beats, 1.0)
            if densities[i] >= bridge_threshold and 0.15 <= position_frac <= 0.90:
                roles[i] = "bridge"
                bridge_count += 1

    # Verse: everything else that is not a repeat of something already labelled
    for i, s in enumerate(sections):
        if roles[i] == "generic":
            if s.is_repeat_of is not None:
                # Inherit parent role
                parent_role = roles[s.is_repeat_of] if s.is_repeat_of < n else "verse"
                roles[i] = parent_role
            else:
                roles[i] = "verse"

    # Structural diversity: when most sections share the same role (highly
    # repetitive music like game loops where every section is a "chorus repeat"),
    # cap "chorus" to at most 40 % of the piece and reassign the rest as "verse".
    # This gives the generator meaningful structural variety to work with.
    chorus_idxs = [i for i, r in enumerate(roles) if r == "chorus"]
    max_chorus  = max(1, n * 2 // 5)   # 40 % cap
    if len(chorus_idxs) > max_chorus:
        # Pick max_chorus evenly-spaced entries from chorus_idxs to keep as chorus.
        n_ch = len(chorus_idxs)
        keep = set()
        for k in range(max_chorus):
            if max_chorus == 1:
                pos = n_ch // 2          # keep the middle occurrence
            else:
                pos = round(k * (n_ch - 1) / (max_chorus - 1))
            keep.add(chorus_idxs[pos])
        for ci in chorus_idxs:
            if ci not in keep:
                roles[ci] = "verse"

    # Assign letter labels A, B, C... per unique fingerprint group
    seen_orig: dict[int, str] = {}
    label_counter = 0
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for i, s in enumerate(sections):
        role = roles[i]
        if role == "intro":
            lbl = "Intro"
        elif role == "outro":
            lbl = "Outro"
        else:
            orig = s.is_repeat_of if s.is_repeat_of is not None else s.section_idx
            if orig not in seen_orig:
                seen_orig[orig] = letters[label_counter % len(letters)]
                label_counter  += 1
            lbl = seen_orig[orig]
        labels.append(lbl)

    return list(zip(roles, labels))
