"""
Melodic variation transforms — imported into transforms.py.

These transforms create musically interesting melodic changes that preserve
the recognisability of the original while making it sound different.
"""

from __future__ import annotations

import random as _rng_mod
from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from music_generator.ir import MusicalAnalysis


def _get_helpers():
    """Lazy import to avoid circular deps."""
    from music_generator.transforms import (
        _decode_ev, _encode_ev, _section_tonal,
    )
    return _decode_ev, _encode_ev, _section_tonal


def vary_intervals(
    analysis: "MusicalAnalysis",
    section_id: str,
    voice: str = "soprano",
    factor: float = 1.3,
    seed: int = 0,
) -> "MusicalAnalysis":
    """
    Scale melodic intervals while preserving contour direction.

    factor > 1.0 : widen leaps (more dramatic, operatic feel).
    factor < 1.0 : narrow leaps (smoother, more stepwise).

    Each interval is scaled by factor then snapped to the nearest semitone.
    Direction (up/down/unison) is always preserved.  A small seeded jitter
    (+-1 semitone) is added so the result sounds musical rather than mechanical.
    """
    _decode_ev, _encode_ev, _section_tonal = _get_helpers()
    rng = _rng_mod.Random(seed)
    result = deepcopy(analysis)
    for section in result.sections:
        if section.id != section_id:
            continue
        if voice not in section.voice_sequences:
            continue
        tpc, sivs = _section_tonal(section, result)
        ev_list = section.voice_sequences[voice]

        pitches = [_decode_ev(ev, tpc, sivs) for ev in ev_list]

        new_evs = []
        prev_midi = None
        for i, ev in enumerate(ev_list):
            midi = pitches[i]
            if ev.get("degree") == "rest" or "pitch" in ev or midi is None:
                new_evs.append(dict(ev))
                if midi is not None:
                    prev_midi = midi
                continue

            if prev_midi is None:
                new_evs.append(dict(ev))
                prev_midi = midi
                continue

            orig_interval = midi - prev_midi
            if orig_interval == 0:
                new_midi = prev_midi
            else:
                direction = 1 if orig_interval > 0 else -1
                scaled = abs(orig_interval) * factor
                jitter = rng.choice([-1, 0, 0, 0, 1])
                new_interval = max(1, round(scaled) + jitter) * direction
                new_midi = prev_midi + new_interval

            new_midi = max(24, min(108, new_midi))
            new_ev = _encode_ev(new_midi, tpc, sivs)
            new_ev["duration"] = ev["duration"]
            new_ev["velocity"] = ev.get("velocity", 80)
            new_evs.append(new_ev)
            prev_midi = new_midi

        section.voice_sequences[voice] = new_evs
        section.generation_mode = "replay"
    return result


def embellish_melody(
    analysis: "MusicalAnalysis",
    section_id: str,
    voice: str = "soprano",
    density: float = 0.3,
    seed: int = 0,
) -> "MusicalAnalysis":
    """
    Add passing tones, neighbor tones, and turns between existing melody notes.

    density : probability of inserting an ornament between any two consecutive
              pitched notes (0.0 = none, 1.0 = every gap).

    Ornamental notes borrow time from the preceding note (splitting its duration)
    so the piece length stays unchanged.
    """
    _decode_ev, _encode_ev, _section_tonal = _get_helpers()
    rng = _rng_mod.Random(seed)
    result = deepcopy(analysis)
    for section in result.sections:
        if section.id != section_id:
            continue
        if voice not in section.voice_sequences:
            continue
        tpc, sivs = _section_tonal(section, result)
        ev_list = section.voice_sequences[voice]

        pitches = [_decode_ev(ev, tpc, sivs) for ev in ev_list]
        new_evs: list[dict] = []

        for i, ev in enumerate(ev_list):
            midi_cur = pitches[i]

            # Look ahead for the next pitched note
            midi_next = None
            for j in range(i + 1, len(ev_list)):
                midi_next = pitches[j]
                if midi_next is not None:
                    break

            if (ev.get("degree") == "rest" or "pitch" in ev
                    or midi_cur is None or midi_next is None):
                new_evs.append(dict(ev))
                continue

            dur = ev["duration"]
            vel = ev.get("velocity", 80)

            if rng.random() >= density or dur < 0.20:
                new_evs.append(dict(ev))
                continue

            interval = midi_next - midi_cur
            abs_interval = abs(interval)

            if abs_interval >= 5 and dur >= 0.30:
                # Passing tone(s) through a leap
                if abs_interval >= 7:
                    step = interval // 3
                    p1, p2 = midi_cur + step, midi_cur + step * 2
                    t = round(dur / 3, 4)
                    new_evs.append(dict(ev, duration=t))
                    for p in (p1, p2):
                        pe = _encode_ev(max(24, min(108, p)), tpc, sivs)
                        pe["duration"] = t
                        pe["velocity"] = max(30, vel - 15)
                        new_evs.append(pe)
                else:
                    mid_pitch = midi_cur + interval // 2
                    t = round(dur / 2, 4)
                    new_evs.append(dict(ev, duration=t))
                    pe = _encode_ev(max(24, min(108, mid_pitch)), tpc, sivs)
                    pe["duration"] = t
                    pe["velocity"] = max(30, vel - 15)
                    new_evs.append(pe)
            elif dur >= 0.40:
                # Neighbor tone (upper or lower)
                direction = rng.choice([1, -1])
                step = rng.choice([1, 2]) * direction
                neighbor = midi_cur + step
                t = round(dur / 2, 4)
                new_evs.append(dict(ev, duration=t))
                ne = _encode_ev(max(24, min(108, neighbor)), tpc, sivs)
                ne["duration"] = t
                ne["velocity"] = max(30, vel - 10)
                new_evs.append(ne)
            else:
                new_evs.append(dict(ev))

        section.voice_sequences[voice] = new_evs
        section.generation_mode = "replay"
    return result


def rhythmic_displace(
    analysis: "MusicalAnalysis",
    section_id: str,
    voice: str = "soprano",
    displacement: float = 0.25,
    probability: float = 0.4,
    seed: int = 0,
) -> "MusicalAnalysis":
    """
    Shift selected notes earlier or later by a fraction of a beat (syncopation).

    displacement : amount in beats to shift (e.g. 0.25 = sixteenth-note push).
    probability  : chance that each note gets displaced (0.0-1.0).

    Works by stealing time from the preceding note and giving it to the
    current note (or vice versa), so total section duration stays constant.
    """
    rng = _rng_mod.Random(seed)
    result = deepcopy(analysis)
    for section in result.sections:
        if section.id != section_id:
            continue
        if voice not in section.voice_sequences:
            continue
        ev_list = list(section.voice_sequences[voice])
        if len(ev_list) < 3:
            continue

        new_evs = [dict(ev) for ev in ev_list]

        for i in range(1, len(new_evs) - 1):
            ev = new_evs[i]
            if ev.get("degree") == "rest":
                continue
            if rng.random() >= probability:
                continue

            dur_cur = ev["duration"]
            dur_prev = new_evs[i - 1]["duration"]

            direction = rng.choice([-1, 1])
            shift = min(displacement, dur_prev * 0.4, dur_cur * 0.4)
            if shift < 0.05:
                continue

            if direction == -1:
                new_evs[i - 1]["duration"] = round(dur_prev - shift, 4)
                new_evs[i]["duration"] = round(dur_cur + shift, 4)
            else:
                new_evs[i - 1]["duration"] = round(dur_prev + shift, 4)
                new_evs[i]["duration"] = round(dur_cur - shift, 4)

        section.voice_sequences[voice] = new_evs
        section.generation_mode = "replay"
    return result


def contour_remap(
    analysis: "MusicalAnalysis",
    section_id: str,
    voice: str = "soprano",
    step_size: int = 2,
    seed: int = 0,
) -> "MusicalAnalysis":
    """
    Preserve the melodic contour (up/down/same) but replace interval sizes.

    step_size=2 : all motion by whole steps (smooth, Debussy-like).
    step_size=1 : chromatic crawl (same shape, minimal motion).
    step_size=5 : quartal leaps (angular, Bartok-like).
    """
    _decode_ev, _encode_ev, _section_tonal = _get_helpers()
    rng = _rng_mod.Random(seed)
    result = deepcopy(analysis)
    for section in result.sections:
        if section.id != section_id:
            continue
        if voice not in section.voice_sequences:
            continue
        tpc, sivs = _section_tonal(section, result)
        ev_list = section.voice_sequences[voice]

        pitches = [_decode_ev(ev, tpc, sivs) for ev in ev_list]

        new_evs = []
        prev_midi = None
        for i, ev in enumerate(ev_list):
            midi = pitches[i]
            if ev.get("degree") == "rest" or "pitch" in ev or midi is None:
                new_evs.append(dict(ev))
                if midi is not None:
                    prev_midi = midi
                continue

            if prev_midi is None:
                new_evs.append(dict(ev))
                prev_midi = midi
                continue

            orig_interval = midi - prev_midi
            if orig_interval == 0:
                new_midi = prev_midi
            else:
                direction = 1 if orig_interval > 0 else -1
                jitter = rng.choice([0, 0, 0, -1, 1])
                new_midi = prev_midi + direction * (step_size + jitter)

            new_midi = max(24, min(108, new_midi))
            new_ev = _encode_ev(new_midi, tpc, sivs)
            new_ev["duration"] = ev["duration"]
            new_ev["velocity"] = ev.get("velocity", 80)
            new_evs.append(new_ev)
            prev_midi = new_midi

        section.voice_sequences[voice] = new_evs
        section.generation_mode = "replay"
    return result


def motif_substitute(
    analysis: "MusicalAnalysis",
    section_id: str,
    voice: str = "soprano",
    seed: int = 0,
) -> "MusicalAnalysis":
    """
    Replace motif occurrences in a voice with related variants from the
    same contour family.

    Builds a 'family' of all motifs sharing the same contour (sign pattern),
    then at each occurrence substitutes a different family member's intervals
    while preserving rhythm.  The listener hears the same shape but with
    fresh interval content.
    """
    _decode_ev, _encode_ev, _section_tonal = _get_helpers()
    rng = _rng_mod.Random(seed)
    result = deepcopy(analysis)
    if not result.motifs:
        return result

    # Build contour families: contour_tuple -> [MotifDef, ...]
    families: dict[tuple, list] = {}
    for m in result.motifs.values():
        if m.type != "motif" or len(m.intervals) < 2:
            continue
        key = tuple(m.contour)
        families.setdefault(key, []).append(m)

    rich_families = {k: v for k, v in families.items() if len(v) >= 2}
    if not rich_families:
        return result

    for section in result.sections:
        if section.id != section_id:
            continue
        if voice not in section.voice_sequences:
            continue

        tpc, sivs = _section_tonal(section, result)
        ev_list = section.voice_sequences[voice]
        pitches = [_decode_ev(ev, tpc, sivs) for ev in ev_list]

        pattern_to_family: dict[tuple, list] = {}
        for family_members in rich_families.values():
            for m in family_members:
                pattern_to_family[tuple(m.intervals)] = family_members

        pitched_indices = [i for i, p in enumerate(pitches) if p is not None]
        if len(pitched_indices) < 3:
            continue

        pitched_midis = [pitches[i] for i in pitched_indices]
        intervals_stream = [
            pitched_midis[i + 1] - pitched_midis[i]
            for i in range(len(pitched_midis) - 1)
        ]

        replaced: set = set()
        new_evs = [dict(ev) for ev in ev_list]

        for motif_len in range(7, 2, -1):
            for start in range(len(intervals_stream) - motif_len + 1):
                if any(i in replaced for i in range(start, start + motif_len)):
                    continue
                window = tuple(intervals_stream[start:start + motif_len])
                if window not in pattern_to_family:
                    continue

                family = pattern_to_family[window]
                alternatives = [m for m in family if tuple(m.intervals) != window]
                if not alternatives:
                    continue
                substitute = rng.choice(alternatives)

                base_midi = pitched_midis[start]
                for j, new_iv in enumerate(substitute.intervals):
                    ev_idx = pitched_indices[start + j + 1]
                    new_midi = max(24, min(108, base_midi + new_iv))
                    new_ev = _encode_ev(new_midi, tpc, sivs)
                    new_ev["duration"] = new_evs[ev_idx]["duration"]
                    new_ev["velocity"] = new_evs[ev_idx].get("velocity", 80)
                    new_evs[ev_idx] = new_ev
                    base_midi = new_midi

                replaced.update(range(start, start + motif_len))

        section.voice_sequences[voice] = new_evs
        section.generation_mode = "replay"
    return result


def melodic_development(
    analysis: "MusicalAnalysis",
    section_id: str,
    voice: str = "soprano",
    technique: str = "expand",
    seed: int = 0,
) -> "MusicalAnalysis":
    """
    Apply a classical melodic development technique to a voice.

    technique choices:
      "expand"         widen intervals ~30 % (more dramatic)
      "contract"       narrow intervals ~30 % (smoother)
      "sequence_up"    transpose second half up a whole step
      "sequence_down"  transpose second half down a whole step
      "ornament"       add neighbor/passing tones at ~35 % density
      "displace"       syncopate ~40 % of notes by a sixteenth
    """
    if technique == "expand":
        return vary_intervals(analysis, section_id, voice, factor=1.3, seed=seed)
    elif technique == "contract":
        return vary_intervals(analysis, section_id, voice, factor=0.7, seed=seed)
    elif technique == "ornament":
        return embellish_melody(analysis, section_id, voice, density=0.35, seed=seed)
    elif technique == "displace":
        return rhythmic_displace(analysis, section_id, voice,
                                 displacement=0.25, probability=0.4, seed=seed)
    elif technique == "sequence_up":
        return _split_transpose(analysis, section_id, voice, semitones=2)
    elif technique == "sequence_down":
        return _split_transpose(analysis, section_id, voice, semitones=-2)
    else:
        raise ValueError(
            f"Unknown technique '{technique}'. "
            f"Available: expand, contract, sequence_up, sequence_down, ornament, displace"
        )


def _split_transpose(
    analysis: "MusicalAnalysis",
    section_id: str,
    voice: str,
    semitones: int,
) -> "MusicalAnalysis":
    """Transpose the second half of pitched events in a voice by semitones."""
    _decode_ev, _encode_ev, _section_tonal = _get_helpers()
    result = deepcopy(analysis)
    for section in result.sections:
        if section.id != section_id:
            continue
        if voice not in section.voice_sequences:
            continue
        tpc, sivs = _section_tonal(section, result)
        ev_list = section.voice_sequences[voice]

        pitched_indices = [
            i for i, ev in enumerate(ev_list)
            if ev.get("degree") != "rest" and "pitch" not in ev
        ]
        if len(pitched_indices) < 4:
            continue

        midpoint = len(pitched_indices) // 2
        new_evs = [dict(ev) for ev in ev_list]

        for idx in pitched_indices[midpoint:]:
            ev = new_evs[idx]
            midi = _decode_ev(ev, tpc, sivs)
            if midi is None:
                continue
            new_midi = max(24, min(108, midi + semitones))
            new_ev = _encode_ev(new_midi, tpc, sivs)
            new_ev["duration"] = ev["duration"]
            new_ev["velocity"] = ev.get("velocity", 80)
            new_evs[idx] = new_ev

        section.voice_sequences[voice] = new_evs
        section.generation_mode = "replay"
    return result
