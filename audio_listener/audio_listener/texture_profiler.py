"""
audio_listener.texture_profiler — Measure TextureHints and EnergyProfile per section.
"""

from __future__ import annotations

from audio_listener.phrase_segmenter import SectionBoundary, PhraseSegment
from audio_listener.pitch_tracker   import TrackedNote


def _notes_in(notes: list[TrackedNote], start: float, end: float) -> list[TrackedNote]:
    return [n for n in notes if start <= n.start_beat < end]


def _mean_velocity(notes: list[TrackedNote]) -> float:
    vels = [n.velocity for n in notes if n.velocity > 0]
    return sum(vels) / len(vels) if vels else 64.0


def _density(notes: list[TrackedNote], start: float, end: float) -> float:
    """Notes per beat."""
    dur = end - start
    n   = _notes_in(notes, start, end)
    return len(n) / max(dur, 1.0)


def _rhythmic_density_param(melody: list[TrackedNote], start: float, end: float) -> float:
    """
    Map notes/beat to rhythmic_density param [0, 1].
    0.25 notes/beat → 0.2   (quarter notes at 120 BPM = whole-note feel)
    2.0  notes/beat → 0.85  (eighth notes at 120 BPM = running sixteenths feel)
    """
    npb = _density(melody, start, end)
    # Logistic-ish mapping
    raw = (npb - 0.25) / 1.75   # 0 at 0.25 npb, 1 at 2.0 npb
    return max(0.1, min(0.9, raw))


def _rhythmic_regularity(melody: list[TrackedNote], start: float, end: float) -> float:
    """
    Coefficient of variation of note durations (inverted).
    All same duration → 1.0. High variance → 0.0.
    """
    notes = _notes_in(melody, start, end)
    durs  = [n.duration_beats for n in notes]
    if len(durs) < 2:
        return 0.8
    mean = sum(durs) / len(durs)
    if mean < 1e-9:
        return 0.8
    var  = sum((d - mean) ** 2 for d in durs) / len(durs)
    cv   = var ** 0.5 / mean
    return max(0.0, min(1.0, 1.0 - cv))


def _step_leap_ratio(melody: list[TrackedNote], start: float, end: float) -> float:
    """Fraction of consecutive intervals that are steps (<=2 semitones)."""
    notes = sorted(_notes_in(melody, start, end), key=lambda n: n.start_beat)
    if len(notes) < 2:
        return 0.7
    intervals = [abs(notes[i+1].pitch - notes[i].pitch) for i in range(len(notes)-1)]
    steps = sum(1 for iv in intervals if iv <= 2)
    return steps / len(intervals)


def _detect_bass_type(bass: list[TrackedNote], start: float, end: float,
                      beats_per_bar: int,
                      extra: list[TrackedNote] | None = None) -> str:
    """
    Classify bass pattern within the section.

    extra  — optional low-register notes from other channels (e.g. counter)
             used to supplement a sparse assigned-bass channel.
    """
    notes = sorted(_notes_in(bass, start, end), key=lambda n: n.start_beat)
    if extra:
        # Merge in low-register notes not already present
        low_extra = sorted(
            [n for n in _notes_in(extra, start, end) if n.pitch < 52],
            key=lambda n: n.start_beat,
        )
        # De-duplicate by start_beat (keep bass note if same onset)
        bass_onsets = {n.start_beat for n in notes}
        notes = sorted(notes + [n for n in low_extra if n.start_beat not in bass_onsets],
                       key=lambda n: n.start_beat)

    if len(notes) < 2:
        return "sustained"

    unique_pcs     = len({n.pitch % 12 for n in notes})
    total_dur      = end - start
    notes_per_beat = len(notes) / max(total_dur, 1.0)

    # NES bass often plays once per bar (~0.25 npb in 4/4) — that's still
    # harmonic motion, not a drone.  Only treat as "sustained" when the bass
    # barely moves at all (< one note every 4 bars).
    if notes_per_beat < 0.08:
        return "sustained"
    elif notes_per_beat < 0.5 and unique_pcs <= 2:
        return "pedal"
    elif notes_per_beat < 0.5:
        return "walking"          # sparse but harmonically active
    elif notes_per_beat < 1.2 and unique_pcs <= 2:
        return "pedal"
    elif notes_per_beat >= 1.0 and unique_pcs >= 3:
        return "walking" if unique_pcs <= 4 else "arpeggiated"
    else:
        return "walking"


def _detect_drum_style(
    drum_notes: list[TrackedNote] | None,
    beats_per_bar: int,
    start: float,
    end: float,
) -> tuple[str, float]:
    """Return (drum_style, drum_intensity) for the section."""
    if not drum_notes:
        return "none", 0.0

    in_sec = [n for n in drum_notes if start <= n.start_beat < end]
    if not in_sec:
        return "none", 0.0

    dur = end - start
    density = len(in_sec) / max(dur, 1.0)  # hits/beat
    intensity = min(1.0, density / 4.0)    # normalize: 4 hits/beat = full intensity

    if beats_per_bar == 3:
        return "waltz", max(0.3, intensity)
    if beats_per_bar == 6:
        return "latin", max(0.3, intensity)

    # 4/4: check kick/snare pattern for genre
    # Count hits on common positions
    bar_beats = float(beats_per_bar)
    beat2_hits = sum(1 for n in in_sec if (n.start_beat - start) % bar_beats == 1.0)
    beat4_hits = sum(1 for n in in_sec if (n.start_beat - start) % bar_beats == 3.0)

    if beat2_hits + beat4_hits > len(in_sec) * 0.3:
        return "rock", max(0.3, intensity)
    if density > 2.0:
        return "jazz", max(0.3, intensity)
    return "rock", max(0.3, intensity)


def _melodic_arc(melody: list[TrackedNote], start: float, end: float) -> str:
    """Pitch trajectory: ascending / descending / arch / flat."""
    notes = sorted(_notes_in(melody, start, end), key=lambda n: n.start_beat)
    if len(notes) < 4:
        return "arch"
    mid  = (start + end) / 2.0
    h1   = [n.pitch for n in notes if n.start_beat < mid]
    h2   = [n.pitch for n in notes if n.start_beat >= mid]
    if not h1 or not h2:
        return "arch"
    mean1, mean2 = sum(h1) / len(h1), sum(h2) / len(h2)
    # Quarters
    q1 = notes[:len(notes)//4]
    q4 = notes[3*len(notes)//4:]
    m_q1 = sum(n.pitch for n in q1) / max(len(q1), 1)
    m_q4 = sum(n.pitch for n in q4) / max(len(q4), 1)
    diff  = m_q4 - m_q1
    peak_mid = max((n.pitch for n in notes if abs(n.start_beat - (start+end)/2) < (end-start)/4),
                   default=m_q1)
    if abs(diff) < 1.5:
        return "flat"
    if peak_mid > max(m_q1, m_q4) + 0.5:
        return "arch"
    if diff > 0:
        return "ascending"
    return "descending"


def profile_section(
    section:       SectionBoundary,
    melody:        list[TrackedNote],
    bass:          list[TrackedNote],
    counter:       list[TrackedNote],
    drum_notes:    list[TrackedNote] | None,
    beats_per_bar: int,
    all_sections:  list[SectionBoundary],
) -> tuple:
    """
    Returns (TextureHints, EnergyProfile) for the given section.
    """
    from music_generator.ir import TextureHints, EnergyProfile

    s, e = section.start_beat, section.end_beat

    # Num voices
    num_voices = 1
    if bass and _notes_in(bass, s, e):
        num_voices += 1
    if counter and _notes_in(counter, s, e):
        num_voices += 1
    num_voices = min(4, num_voices)

    # Supplement bass only when there are NO bass notes in this section at all.
    # Merging a dense counter channel (e.g. an arpeggiated accompaniment) into
    # the bass pool corrupts the type detection — only do it as a last resort.
    bass_in_sec = _notes_in(bass or [], s, e)
    if bass_in_sec:
        bass_extra = None
    else:
        bass_extra = counter if counter else None
    bass_type  = _detect_bass_type(bass or [], s, e, beats_per_bar, extra=bass_extra)
    rhy_dens   = _rhythmic_density_param(melody, s, e)
    rhy_reg    = _rhythmic_regularity(melody, s, e)
    step_leap  = _step_leap_ratio(melody, s, e)
    arc        = _melodic_arc(melody, s, e)

    # Inner voice style
    inner_style = None
    if counter and _notes_in(counter, s, e):
        inner_style = "countermelody"

    drum_style, drum_intensity = _detect_drum_style(drum_notes, beats_per_bar, s, e)

    texture = TextureHints(
        bass_type=bass_type,
        inner_voice_style=inner_style,
        rhythmic_density=round(rhy_dens, 3),
        rhythmic_regularity=round(rhy_reg, 3),
        step_leap_ratio=round(step_leap, 3),
        melodic_direction=arc if arc in ("ascending", "descending", "arch") else None,
        num_voices=num_voices,
        swing=0.0,
        use_ornaments=False,
        drum_style=drum_style,
        drum_intensity=round(drum_intensity, 3),
    )

    # Energy
    mel_notes = _notes_in(melody, s, e)
    all_notes_in_sec = mel_notes + _notes_in(bass, s, e) + _notes_in(counter, s, e)

    # Density score: normalise against all sections
    all_dens = [
        _density(melody, sec.start_beat, sec.end_beat)
        for sec in all_sections
    ]
    max_dens = max(all_dens) if all_dens else 1.0
    dens_score = _density(melody, s, e) / max(max_dens, 1e-9)

    vel_score  = _mean_velocity(all_notes_in_sec) / 127.0
    energy_level = min(1.0, max(0.05, 0.6 * dens_score + 0.4 * vel_score))

    energy = EnergyProfile(level=round(energy_level, 3), arc=arc)

    return texture, energy
