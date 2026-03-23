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

    # Intro: first section, not chorus, short, early in piece
    if roles[0] != "chorus":
        n_phrases = sections[0].end_phrase - sections[0].start_phrase
        if n_phrases <= 2 and sections[0].end_beat <= total_beats * 0.25:
            roles[0] = "intro"

    # Outro: last section, not chorus, density significantly drops
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
