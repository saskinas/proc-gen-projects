"""
music_generator._voice_styler — Voice-sequence-level style transforms.

These operate on individual voice sequences (list[dict]) rather than on
full MusicalAnalysis objects.  They modify rhythm, velocity, and note
patterns in place to impart a musical style while preserving the original
pitches and melodic contour.

Voice-sequence event format (pitched):
    {"degree": int, "alter": int, "octave": int, "duration": float, "velocity": int}
    degree="rest" for rests.

Voice-sequence event format (drums):
    {"pitch": str, "duration": float, "velocity": int}
"""

from __future__ import annotations

from copy import deepcopy


# ── Helpers ──────────────────────────────────────────────────────────────────

def _is_rest(ev: dict) -> bool:
    return ev.get("degree") == "rest"


def _is_pitched(ev: dict) -> bool:
    return ev.get("degree") not in ("rest", None) and "degree" in ev


def _total_duration(seq: list[dict]) -> float:
    return sum(ev.get("duration", 0.0) for ev in seq)


def _get_helpers():
    """Lazy import to avoid circular deps with transforms.py."""
    from music_generator.transforms import _decode_ev, _encode_ev
    return _decode_ev, _encode_ev


# ── Swing ────────────────────────────────────────────────────────────────────

def swing_rhythm(seq: list[dict], amount: float = 0.6) -> list[dict]:
    """
    Apply swing feel by redistributing duration within consecutive note pairs.

    amount: 0.5 = straight, 0.67 = standard swing, 0.75 = heavy shuffle.
    Only swings pairs where both notes are pitched (skips rests).
    """
    if amount <= 0.5:
        return seq
    result = deepcopy(seq)
    i = 0
    while i < len(result) - 1:
        a, b = result[i], result[i + 1]
        if _is_rest(a) or _is_rest(b):
            i += 1
            continue
        pair_dur = a["duration"] + b["duration"]
        # Only swing pairs of roughly equal duration (within 2x of each other)
        ratio = a["duration"] / b["duration"] if b["duration"] > 0 else 999
        if 0.3 < ratio < 3.0:
            a["duration"] = round(pair_dur * amount, 4)
            b["duration"] = round(pair_dur * (1.0 - amount), 4)
        i += 2
    return result


# ── Walking bass fill ────────────────────────────────────────────────────────

def walking_bass_fill(
    seq: list[dict],
    tonic_pc: int,
    scale_ivs: list[int],
    step_dur: float = 0.5,
) -> list[dict]:
    """
    Fill rests between bass notes with stepwise passing tones.

    Walks diatonically from the previous note toward the next pitched note.
    step_dur controls the duration of each passing tone (default = eighth note).
    """
    _decode_ev, _encode_ev = _get_helpers()
    result = deepcopy(seq)
    out: list[dict] = []

    for idx, ev in enumerate(result):
        if not _is_rest(ev) or ev["duration"] < step_dur * 1.8:
            out.append(ev)
            continue

        # Find the preceding and following pitched notes
        prev_pitch = None
        for j in range(len(out) - 1, -1, -1):
            if _is_pitched(out[j]):
                prev_pitch = _decode_ev(out[j], tonic_pc, scale_ivs)
                prev_vel = out[j].get("velocity", 64)
                break

        next_pitch = None
        for j in range(idx + 1, len(result)):
            if _is_pitched(result[j]):
                next_pitch = _decode_ev(result[j], tonic_pc, scale_ivs)
                break

        if prev_pitch is None or next_pitch is None:
            out.append(ev)
            continue

        # Generate stepwise walk from prev toward next
        rest_dur = ev["duration"]
        n_steps = max(1, int(rest_dur / step_dur))
        actual_step_dur = round(rest_dur / n_steps, 4)

        direction = 1 if next_pitch > prev_pitch else -1
        # Build scale-tone pitches between prev and next
        walk_pitches = []
        p = prev_pitch
        for _ in range(n_steps):
            # Step by one scale degree
            p_pc = p % 12
            p_oct = p // 12
            rel = (p_pc - tonic_pc) % 12
            # Find current scale degree index
            best_idx = 0
            best_dist = 99
            for si, iv in enumerate(scale_ivs):
                d = abs(rel - iv)
                if d < best_dist:
                    best_dist = d
                    best_idx = si
            # Move one degree
            next_idx = best_idx + direction
            if 0 <= next_idx < len(scale_ivs):
                new_pc = (tonic_pc + scale_ivs[next_idx]) % 12
                new_pitch = p_oct * 12 + new_pc
                # Adjust octave if we wrapped
                if direction > 0 and new_pitch <= p:
                    new_pitch += 12
                elif direction < 0 and new_pitch >= p:
                    new_pitch -= 12
            elif next_idx >= len(scale_ivs):
                # Wrap to next octave
                new_pc = tonic_pc
                new_pitch = (p_oct + 1) * 12 + new_pc
            else:
                # Wrap to previous octave
                new_pc = (tonic_pc + scale_ivs[-1]) % 12
                new_pitch = (p_oct - 1) * 12 + new_pc

            walk_pitches.append(new_pitch)
            p = new_pitch

        # Convert pitches to degree events
        vel = int(prev_vel * 0.75)  # passing tones slightly softer
        for wp in walk_pitches:
            deg_ev = _encode_ev(wp, tonic_pc, scale_ivs)
            deg_ev["duration"] = actual_step_dur
            deg_ev["velocity"] = vel
            out.append(deg_ev)

        continue  # rest replaced with walk

    return out


# ── Arpeggiate sustained notes ───────────────────────────────────────────────

def arpeggiate_bass(
    seq: list[dict],
    tonic_pc: int,
    scale_ivs: list[int],
    pattern: str = "alberti",
    subdivision: float = 0.5,
) -> list[dict]:
    """
    Break sustained bass notes into arpeggiated patterns.

    pattern:
        "alberti"  — root-fifth-third-fifth (classical Alberti bass)
        "simple"   — root-third-fifth (folk broken chord)
        "octave"   — root-octave alternation (minimal)

    Only breaks notes longer than 2 * subdivision.
    """
    _decode_ev, _encode_ev = _get_helpers()
    result = []

    for ev in seq:
        if _is_rest(ev) or ev["duration"] < subdivision * 1.8:
            result.append(deepcopy(ev))
            continue

        pitch = _decode_ev(ev, tonic_pc, scale_ivs)
        if pitch is None:
            result.append(deepcopy(ev))
            continue

        dur = ev["duration"]
        vel = ev.get("velocity", 64)

        # Build the arpeggio pitch intervals (in scale degrees above root)
        if pattern == "alberti":
            # root, 5th, 3rd, 5th (repeating)
            degree_offsets = [0, 4, 2, 4]
        elif pattern == "simple":
            degree_offsets = [0, 2, 4]
        elif pattern == "octave":
            degree_offsets = [0, 7]  # semitone offset for octave
        else:
            degree_offsets = [0, 2, 4]

        n_notes = max(2, int(dur / subdivision))
        note_dur = round(dur / n_notes, 4)

        for i in range(n_notes):
            offset_idx = degree_offsets[i % len(degree_offsets)]
            if pattern == "octave":
                # Direct semitone offset
                arp_pitch = pitch + offset_idx
            else:
                # Scale-degree offset
                p_pc = pitch % 12
                p_oct = pitch // 12
                rel = (p_pc - tonic_pc) % 12
                # Find current scale degree
                base_idx = 0
                best_dist = 99
                for si, iv in enumerate(scale_ivs):
                    if abs(rel - iv) < best_dist:
                        best_dist = abs(rel - iv)
                        base_idx = si
                target_idx = base_idx + offset_idx
                target_oct = p_oct + target_idx // len(scale_ivs)
                target_idx = target_idx % len(scale_ivs)
                arp_pitch = target_oct * 12 + (tonic_pc + scale_ivs[target_idx]) % 12

            arp_ev = _encode_ev(arp_pitch, tonic_pc, scale_ivs)
            arp_ev["duration"] = note_dur
            # Slight velocity variation for musicality
            v_mod = 1.0 if i % len(degree_offsets) == 0 else 0.8
            arp_ev["velocity"] = max(1, min(127, int(vel * v_mod)))
            result.append(arp_ev)

    return result


# ── Driving eighths ──────────────────────────────────────────────────────────

def driving_eighths(seq: list[dict], subdivision: float = 0.5) -> list[dict]:
    """
    Convert sustained notes into repeated eighth-note pulses (rock/punk feel).

    Only breaks notes longer than 2 * subdivision.
    """
    result = []
    for ev in seq:
        if _is_rest(ev) or ev["duration"] < subdivision * 1.8:
            result.append(deepcopy(ev))
            continue

        dur = ev["duration"]
        n_reps = max(2, int(dur / subdivision))
        rep_dur = round(dur / n_reps, 4)
        vel = ev.get("velocity", 64)

        for i in range(n_reps):
            rep = deepcopy(ev)
            rep["duration"] = rep_dur
            # Accent on downbeats (every other note)
            rep["velocity"] = vel if i % 2 == 0 else max(1, int(vel * 0.7))
            result.append(rep)

    return result


# ── Ornaments ────────────────────────────────────────────────────────────────

def add_ornaments(
    seq: list[dict],
    tonic_pc: int,
    scale_ivs: list[int],
    density: float = 0.3,
    seed: int = 0,
) -> list[dict]:
    """
    Add baroque-style ornaments (mordents and turns) to suitable notes.

    density: 0.0-1.0 fraction of eligible notes that get ornamented.
    Only ornaments notes with duration >= 0.4 beats (enough room for the ornament).
    """
    import random
    _decode_ev, _encode_ev = _get_helpers()

    rng = random.Random(seed)
    result = []

    for ev in seq:
        if _is_rest(ev) or ev["duration"] < 0.4:
            result.append(deepcopy(ev))
            continue

        if rng.random() > density:
            result.append(deepcopy(ev))
            continue

        pitch = _decode_ev(ev, tonic_pc, scale_ivs)
        if pitch is None:
            result.append(deepcopy(ev))
            continue

        vel = ev.get("velocity", 64)
        dur = ev["duration"]

        # Choose mordent (upper or lower neighbor)
        neighbor_dir = rng.choice([1, -1])

        # Find the neighbor scale tone
        p_pc = pitch % 12
        p_oct = pitch // 12
        rel = (p_pc - tonic_pc) % 12
        base_idx = 0
        best_dist = 99
        for si, iv in enumerate(scale_ivs):
            if abs(rel - iv) < best_dist:
                best_dist = abs(rel - iv)
                base_idx = si
        nbr_idx = base_idx + neighbor_dir
        if 0 <= nbr_idx < len(scale_ivs):
            nbr_pitch = p_oct * 12 + (tonic_pc + scale_ivs[nbr_idx]) % 12
            if neighbor_dir > 0 and nbr_pitch <= pitch:
                nbr_pitch += 12
            elif neighbor_dir < 0 and nbr_pitch >= pitch:
                nbr_pitch -= 12
        else:
            # Skip ornament if at scale boundary
            result.append(deepcopy(ev))
            continue

        # Mordent: main → neighbor → main, each taking ornament_dur
        ornament_dur = round(min(0.125, dur * 0.2), 4)
        main_dur = round(dur - ornament_dur * 2, 4)

        if main_dur < 0.1:
            result.append(deepcopy(ev))
            continue

        # Main note (shortened)
        main = deepcopy(ev)
        main["duration"] = ornament_dur
        result.append(main)

        # Neighbor
        nbr_ev = _encode_ev(nbr_pitch, tonic_pc, scale_ivs)
        nbr_ev["duration"] = ornament_dur
        nbr_ev["velocity"] = max(1, int(vel * 0.85))
        result.append(nbr_ev)

        # Return to main
        tail = deepcopy(ev)
        tail["duration"] = main_dur
        result.append(tail)

    return result


# ── Velocity accents ─────────────────────────────────────────────────────────

def accent_velocity(
    seq: list[dict],
    beats_per_bar: int = 4,
    boost: int = 20,
    pattern: str = "downbeat",
) -> list[dict]:
    """
    Apply repeating velocity accent pattern.

    pattern:
        "downbeat"  — accent beat 1 of each bar
        "backbeat"  — accent beats 2 and 4 (rock/pop)
        "all"       — accent every beat (driving)
    """
    result = deepcopy(seq)
    cursor = 0.0

    for ev in result:
        dur = ev.get("duration", 0.5)
        if not _is_rest(ev):
            beat_in_bar = cursor % beats_per_bar
            should_accent = False
            if pattern == "downbeat":
                should_accent = beat_in_bar < 0.1
            elif pattern == "backbeat":
                should_accent = (abs(beat_in_bar - 1) < 0.1 or
                                 abs(beat_in_bar - 3) < 0.1)
            elif pattern == "all":
                should_accent = abs(beat_in_bar - round(beat_in_bar)) < 0.1

            if should_accent:
                ev["velocity"] = min(127, ev.get("velocity", 64) + boost)
            else:
                ev["velocity"] = max(1, ev.get("velocity", 64) - boost // 3)
        cursor += dur

    return result


# ── Thin to downbeats ────────────────────────────────────────────────────────

def thin_to_downbeats(
    seq: list[dict],
    beats_per_bar: int = 4,
    keep_fraction: float = 0.5,
) -> list[dict]:
    """
    Reduce density by converting off-beat notes to rests.

    Keeps notes near beat boundaries; converts others to rests that extend
    the previous event.  keep_fraction controls how many off-beat notes survive.
    """
    import random
    rng = random.Random(42)
    result = []
    cursor = 0.0

    for ev in deepcopy(seq):
        dur = ev.get("duration", 0.5)
        if _is_rest(ev):
            result.append(ev)
            cursor += dur
            continue

        beat_in_bar = cursor % beats_per_bar
        on_beat = abs(beat_in_bar - round(beat_in_bar)) < 0.15

        if on_beat or rng.random() < keep_fraction:
            result.append(ev)
        else:
            # Replace with rest
            rest = {"degree": "rest", "alter": 0, "octave": 0,
                    "duration": dur, "velocity": 0}
            # Merge with previous rest if possible
            if result and _is_rest(result[-1]):
                result[-1]["duration"] = round(result[-1]["duration"] + dur, 4)
            else:
                result.append(rest)
        cursor += dur

    return result


# ── Humanize timing ──────────────────────────────────────────────────────────

def humanize(seq: list[dict], amount: float = 0.02, seed: int = 0) -> list[dict]:
    """
    Add tiny random duration fluctuations for a less mechanical feel.

    amount: maximum deviation in beats (default 0.02 = subtle).
    Total duration is preserved by compensating in the next event.
    """
    import random
    rng = random.Random(seed)
    result = deepcopy(seq)

    for i in range(len(result) - 1):
        if _is_rest(result[i]) and _is_rest(result[i + 1]):
            continue
        delta = rng.uniform(-amount, amount)
        new_dur = result[i]["duration"] + delta
        if new_dur < 0.03:
            continue
        next_dur = result[i + 1]["duration"] - delta
        if next_dur < 0.03:
            continue
        result[i]["duration"] = round(new_dur, 4)
        result[i + 1]["duration"] = round(next_dur, 4)

    return result


# ── Velocity scale ───────────────────────────────────────────────────────────

def velocity_scale(seq: list[dict], factor: float = 1.0) -> list[dict]:
    """Scale all velocities by a factor, clamped to 1-127."""
    result = deepcopy(seq)
    for ev in result:
        if not _is_rest(ev) and "velocity" in ev:
            ev["velocity"] = max(1, min(127, int(ev["velocity"] * factor)))
    return result


# ── Style application helpers ────────────────────────────────────────────────

def style_voices(
    section,
    tonic_pc: int,
    scale_ivs: list[int],
    style: str,
    lead_voice: str = "soprano",
) -> None:
    """
    Apply style-specific transforms to all voice_sequences in a section.

    Modifies section.voice_sequences in place. The lead voice gets lighter
    treatment (preserving the recognisable melody); accompaniment voices
    get heavier stylistic modification.

    style: "jazz" | "baroque" | "classical" | "folk" | "rock" | "lofi" |
           "ambient" | "minimalist" | "epic"
    """
    vs = section.voice_sequences
    if not vs:
        return

    for role, seq in list(vs.items()):
        if role == "drums":
            # Drum sequences have a different format — apply velocity only
            if style == "jazz":
                vs[role] = swing_rhythm(seq, 0.6)
                vs[role] = velocity_scale(vs[role], 0.85)
            elif style == "rock":
                vs[role] = accent_velocity(seq, pattern="backbeat", boost=25)
            elif style in ("ambient", "minimalist"):
                vs[role] = velocity_scale(seq, 0.5)
                vs[role] = thin_to_downbeats(vs[role], keep_fraction=0.3)
            elif style == "lofi":
                vs[role] = swing_rhythm(seq, 0.58)
                vs[role] = velocity_scale(vs[role], 0.7)
            continue

        is_lead = (role == lead_voice)
        is_bass = (role == "bass")

        if style == "jazz":
            if is_lead:
                vs[role] = swing_rhythm(seq, 0.6)
                vs[role] = humanize(vs[role], 0.02)
            elif is_bass:
                vs[role] = swing_rhythm(seq, 0.6)
                vs[role] = walking_bass_fill(vs[role], tonic_pc, scale_ivs, step_dur=0.5)
            else:
                vs[role] = swing_rhythm(seq, 0.55)
                vs[role] = humanize(vs[role], 0.015)

        elif style == "baroque":
            if is_lead:
                vs[role] = add_ornaments(seq, tonic_pc, scale_ivs, density=0.25)
            elif is_bass:
                vs[role] = walking_bass_fill(seq, tonic_pc, scale_ivs, step_dur=0.5)
            else:
                vs[role] = add_ornaments(seq, tonic_pc, scale_ivs, density=0.15)

        elif style == "classical":
            if is_lead:
                pass  # Keep melody untouched
            elif is_bass:
                vs[role] = arpeggiate_bass(seq, tonic_pc, scale_ivs,
                                           pattern="alberti", subdivision=0.5)
            else:
                # Thin inner voices slightly for clarity
                vs[role] = thin_to_downbeats(seq, keep_fraction=0.7)

        elif style == "folk":
            if is_lead:
                vs[role] = humanize(seq, 0.025)
            elif is_bass:
                vs[role] = arpeggiate_bass(seq, tonic_pc, scale_ivs,
                                           pattern="simple", subdivision=0.5)
            else:
                vs[role] = humanize(seq, 0.02)

        elif style == "rock":
            if is_lead:
                vs[role] = accent_velocity(seq, pattern="all", boost=15)
            elif is_bass:
                vs[role] = driving_eighths(seq, subdivision=0.5)
                vs[role] = accent_velocity(vs[role], pattern="downbeat", boost=20)
            else:
                vs[role] = accent_velocity(seq, pattern="backbeat", boost=15)

        elif style == "lofi":
            if is_lead:
                vs[role] = swing_rhythm(seq, 0.58)
                vs[role] = humanize(vs[role], 0.03)
                vs[role] = velocity_scale(vs[role], 0.8)
            elif is_bass:
                vs[role] = swing_rhythm(seq, 0.58)
                vs[role] = velocity_scale(vs[role], 0.75)
            else:
                vs[role] = swing_rhythm(seq, 0.55)
                vs[role] = thin_to_downbeats(vs[role], keep_fraction=0.6)
                vs[role] = velocity_scale(vs[role], 0.7)

        elif style == "ambient":
            if is_lead:
                vs[role] = velocity_scale(seq, 0.7)
                vs[role] = humanize(vs[role], 0.04)
            elif is_bass:
                vs[role] = thin_to_downbeats(seq, keep_fraction=0.3)
                vs[role] = velocity_scale(vs[role], 0.6)
            else:
                vs[role] = thin_to_downbeats(seq, keep_fraction=0.25)
                vs[role] = velocity_scale(vs[role], 0.5)

        elif style == "minimalist":
            if is_lead:
                vs[role] = velocity_scale(seq, 0.75)
            elif is_bass:
                vs[role] = thin_to_downbeats(seq, keep_fraction=0.3)
                vs[role] = velocity_scale(vs[role], 0.65)
            else:
                vs[role] = thin_to_downbeats(seq, keep_fraction=0.2)
                vs[role] = velocity_scale(vs[role], 0.5)

        elif style == "epic":
            if is_lead:
                vs[role] = accent_velocity(seq, pattern="downbeat", boost=20)
            elif is_bass:
                vs[role] = driving_eighths(seq, subdivision=0.5)
                vs[role] = accent_velocity(vs[role], pattern="downbeat", boost=25)
                vs[role] = velocity_scale(vs[role], 1.15)
            else:
                vs[role] = accent_velocity(seq, pattern="downbeat", boost=15)
                vs[role] = velocity_scale(vs[role], 1.1)
