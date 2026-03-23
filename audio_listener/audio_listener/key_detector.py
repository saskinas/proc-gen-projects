"""
audio_listener.key_detector — Krumhansl-Kessler key and mode detection.
"""

from __future__ import annotations

from audio_listener.pitch_tracker import TrackedNote

# Krumhansl-Kessler (1990) pitch-class profiles
_KK_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KK_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _build_histogram(notes: list[TrackedNote], weight: float = 1.0) -> list[float]:
    """Sum duration_beats per pitch class (0-11), scaled by weight."""
    hist = [0.0] * 12
    for n in notes:
        hist[n.pitch % 12] += n.duration_beats * weight
    return hist


def _pearson(x: list[float], y: list[float]) -> float:
    n    = len(x)
    mx   = sum(x) / n
    my   = sum(y) / n
    num  = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    d_x  = (sum((v - mx) ** 2 for v in x)) ** 0.5
    d_y  = (sum((v - my) ** 2 for v in y)) ** 0.5
    if d_x < 1e-9 or d_y < 1e-9:
        return 0.0
    return num / (d_x * d_y)


def detect_key_mode(
    melody:  list[TrackedNote],
    bass:    list[TrackedNote],
    counter: list[TrackedNote],
) -> tuple[str, str]:
    """
    Run K-S over all tonal voices combined.

    Bass notes are weighted 1.5× (bass as root hint is stronger evidence).
    Returns (key_name, mode_string) where mode_string is one of:
        'major', 'minor', 'harmonic_minor', 'dorian'
    """
    hist = [0.0] * 12
    for n in melody:
        hist[n.pitch % 12] += n.duration_beats
    for n in bass:
        hist[n.pitch % 12] += n.duration_beats * 1.5
    for n in counter:
        hist[n.pitch % 12] += n.duration_beats

    total = sum(hist)
    if total < 1e-9:
        return "C", "major"

    hist = [v / total for v in hist]

    best_score = -2.0
    best_root  = 0
    best_mode  = "major"

    for root in range(12):
        for mode, profile in [("major", _KK_MAJOR), ("minor", _KK_MINOR)]:
            # Rotate profile so rotated[i] = expected weight for pitch class i in this key.
            # rotated[i] = profile[(i - root) % 12]
            # Equivalent to: profile[(12-root)%12:] + profile[:(12-root)%12]
            shift   = (12 - root) % 12
            rotated = profile[shift:] + profile[:shift]
            score   = _pearson(hist, rotated)
            if score > best_score:
                best_score = score
                best_root  = root
                best_mode  = mode

    key_name = _NOTE_NAMES[best_root]

    # Refine minor: distinguish natural_minor / harmonic_minor / dorian
    if best_mode == "minor":
        best_mode = _refine_minor_mode(best_root, melody + counter)

    return key_name, best_mode


def _refine_minor_mode(root_pc: int, melody: list[TrackedNote]) -> str:
    """
    Check scale-degree 6 and 7 occurrences to pick the right minor mode.

    natural_minor   : b6 and b7  (most common for NES)
    harmonic_minor  : b6 and #7
    dorian          : #6 and b7
    """
    # Scale degree intervals from root for natural minor: 0,2,3,5,7,8,10
    # Raised 7th (leading tone) = root + 11
    # Raised 6th (dorian)       = root + 9
    raised_7  = (root_pc + 11) % 12
    natural_7 = (root_pc + 10) % 12
    raised_6  = (root_pc + 9)  % 12
    natural_6 = (root_pc + 8)  % 12

    counts = {}
    for n in melody:
        pc = n.pitch % 12
        counts[pc] = counts.get(pc, 0.0) + n.duration_beats

    r7  = counts.get(raised_7,  0.0)
    n7  = counts.get(natural_7, 0.0)
    r6  = counts.get(raised_6,  0.0)
    _n6 = counts.get(natural_6, 0.0)

    if r7 > n7 * 1.5:
        return "harmonic_minor"
    if r6 > _n6 * 1.5:
        return "dorian"
    return "minor"
