"""
music_generator.transforms — Pure IR transform functions

All functions accept a MusicalAnalysis (or subcomponent) and return a new
object without mutating the input.  Chain them to build remix pipelines:

    from music_generator.transforms import mode_swap, transpose, apply_style_preset

    remixed = (
        analysis
        |> transpose(-3)            # down a minor third
        |> mode_swap("minor")       # reinterpret in minor
        |> apply_style_preset("jazz")   # jazz texture on every section
    )

(Python doesn't have |>, so chain with assignment or compose():
    remixed = compose(analysis, transpose(-3), mode_swap("minor"), apply_style_preset("jazz"))
)
"""

from __future__ import annotations

import math

from copy import deepcopy
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from music_generator.ir import MusicalAnalysis, SectionSpec, TextureHints

from music_generator import NOTE_NAMES, ROOT_TO_PC, DIATONIC_TRIADS, SCALE_INTERVALS, degree_root_pc


# ── Voice-sequence codec helpers ──────────────────────────────────────────────
# Used internally by transforms that operate on stored note events.

def _decode_ev(ev: dict, tonic_pc: int, scale_ivs: list) -> int | None:
    """Degree-event → MIDI pitch, or None for rests."""
    degree = ev.get("degree")
    if degree == "rest":
        return None
    octave = ev.get("octave", 4)
    alter  = ev.get("alter", 0)
    if degree == 0:
        raw_pc = (tonic_pc + alter) % 12
    else:
        try:
            iv = scale_ivs[int(degree) - 1]
        except (IndexError, ValueError):
            iv = 0
        raw_pc = (tonic_pc + iv + alter) % 12
    return (octave + 1) * 12 + raw_pc


def _encode_ev(pitch: int, tonic_pc: int, scale_ivs: list) -> dict:
    """MIDI pitch → {degree, alter, octave}."""
    relative_pc = (pitch % 12 - tonic_pc) % 12
    octave      = pitch // 12 - 1
    best_degree, best_alter, best_abs = 0, relative_pc, relative_pc
    for d_idx, iv in enumerate(scale_ivs):
        alter = relative_pc - iv
        if abs(alter) < best_abs:
            best_abs, best_degree, best_alter = abs(alter), d_idx + 1, alter
        elif abs(alter) == best_abs and alter < best_alter:
            best_degree, best_alter = d_idx + 1, alter
    return {"degree": best_degree, "alter": best_alter, "octave": octave}


def _section_tonal(section, analysis) -> tuple:
    """Return (tonic_pc, scale_ivs) for a section, honouring modulation."""
    key  = section.extra_params.get("key",  analysis.key)
    mode = section.extra_params.get("mode", analysis.mode)
    tpc  = ROOT_TO_PC.get(key, ROOT_TO_PC.get(analysis.key, 0))
    sivs = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])
    return tpc, sivs


# ── Harmonic transforms ───────────────────────────────────────────────────────

def transpose(analysis: "MusicalAnalysis", semitones: int) -> "MusicalAnalysis":
    """
    Transpose the entire analysis by the given number of semitones.

    Updates:
    - analysis.key (re-spelt using sharps)
    - Every HarmonicEvent.root_pc in every section's harmonic_plan

    voice_sequences do NOT need updating: they are stored as scale degrees
    relative to the tonic, so changing analysis.key (and thus tonic_pc at
    decode time) automatically shifts every decoded pitch by the right amount.
    """
    result = deepcopy(analysis)
    old_pc = ROOT_TO_PC[result.key]
    new_pc = (old_pc + semitones) % 12
    result.key = NOTE_NAMES[new_pc]

    for section in result.sections:
        for ev in section.harmonic_plan:
            ev.root_pc = (ev.root_pc + semitones) % 12

    return result


def mode_swap(analysis: "MusicalAnalysis", new_mode: str) -> "MusicalAnalysis":
    """
    Reinterpret the harmonic plan in a new mode (parallel mode substitution).

    Keeps each HarmonicEvent's degree and phrase_idx intact; recalculates
    root_pc and chord_type from the scale degree in new_mode.

    voice_sequences do NOT need updating: because pitches are stored as
    (degree, alter, octave), decode time uses SCALE_INTERVALS[new_mode] to
    look up diatonic offsets. Degree 3 in major → major third (4 st above
    tonic); the same degree 3 in minor → minor third (3 st). Parallel-mode
    melody adaptation is automatic and musically correct.
    """
    result = deepcopy(analysis)
    result.mode = new_mode
    tonic_pc = ROOT_TO_PC[result.key]

    for section in result.sections:
        for ev in section.harmonic_plan:
            ev.root_pc   = degree_root_pc(tonic_pc, new_mode, ev.degree)
            try:
                ev.chord_type = DIATONIC_TRIADS[new_mode][ev.degree - 1]
            except (KeyError, IndexError):
                pass  # non-diatonic mode: leave chord_type as-is

    return result


def reharmonize(
    analysis: "MusicalAnalysis",
    section_id: str,
    complexity: float = 0.7,
    seed: int = 0,
) -> "MusicalAnalysis":
    """
    Re-run harmonic generation for one section, discarding its existing plan.

    Sets harmonic_plan=[] on the target section so generate_from_analysis()
    will call HarmonicPlanGenerator with the section's extra_params (which
    should include num_phrases, phrase_length, and any harmony knobs).

    complexity : chord_complexity override for this section's re-generation.
    """
    result = deepcopy(analysis)
    for section in result.sections:
        if section.id == section_id:
            section.harmonic_plan = []
            section.extra_params.setdefault("chord_complexity", complexity)
            section.generation_mode = "generate"
    return result


# ── Structural transforms ─────────────────────────────────────────────────────

def reorder_sections(
    analysis: "MusicalAnalysis",
    order: list[str],
) -> "MusicalAnalysis":
    """
    Rearrange sections according to the given list of section ids.

    Sections whose ids are not in order are dropped.
    Sections can be repeated by including an id more than once — each
    repetition uses the same SectionSpec (deep-copied independently).
    """
    result = deepcopy(analysis)
    section_map = {s.id: s for s in result.sections}
    result.sections = [
        deepcopy(section_map[sid])
        for sid in order
        if sid in section_map
    ]
    return result


def drop_section(analysis: "MusicalAnalysis", section_id: str) -> "MusicalAnalysis":
    """Remove the section with the given id."""
    result = deepcopy(analysis)
    result.sections = [s for s in result.sections if s.id != section_id]
    return result


def repeat_section(
    analysis: "MusicalAnalysis",
    section_id: str,
    n: int = 2,
) -> "MusicalAnalysis":
    """
    Replace the section with n copies of it placed consecutively.

    Each copy gets a unique id suffix (_r1, _r2, …) so they remain distinct.
    """
    result = deepcopy(analysis)
    new_sections: list = []
    for section in result.sections:
        if section.id == section_id:
            for i in range(n):
                copy = deepcopy(section)
                copy.id = f"{section_id}_r{i + 1}" if i > 0 else section_id
                copy.repeat_of = section_id if i > 0 else None
                new_sections.append(copy)
        else:
            new_sections.append(section)
    result.sections = new_sections
    return result


# ── Texture transforms ────────────────────────────────────────────────────────

def change_texture(
    analysis: "MusicalAnalysis",
    section_id: str,
    **hints,
) -> "MusicalAnalysis":
    """
    Update TextureHints fields for a specific section.

    Keyword args must be valid TextureHints field names:
        change_texture(analysis, "chorus", bass_type="rock", rhythmic_density=0.8)

    Texture changes switch the section from replay mode to generative mode so the
    new style is actually heard.  Voice sequences are retained as a melodic template
    (phrase motif) for the generator.
    """
    result = deepcopy(analysis)
    for section in result.sections:
        if section.id == section_id:
            for k, v in hints.items():
                if hasattr(section.texture, k):
                    setattr(section.texture, k, v)
            section.generation_mode = "arrange"
    return result


def change_energy(
    analysis: "MusicalAnalysis",
    section_id: str,
    level: float | None = None,
    arc: str | None = None,
) -> "MusicalAnalysis":
    """
    Set energy level and/or arc for a specific section.

    Energy changes switch the section from replay mode to generative mode so
    the new density / arc is actually applied.
    """
    result = deepcopy(analysis)
    for section in result.sections:
        if section.id == section_id:
            if level is not None:
                section.energy.level = max(0.0, min(1.0, level))
            if arc is not None:
                section.energy.arc = arc
            section.generation_mode = "arrange"
    return result


def apply_style_preset(
    analysis: "MusicalAnalysis",
    style: str,
    section_ids: list[str] | None = None,
) -> "MusicalAnalysis":
    """
    Apply a named style preset to sections' TextureHints.

    section_ids : list of section ids to apply to; None = all sections.

    Style presets switch affected sections from replay mode to generative mode
    so the new texture is heard.  Voice sequences are retained as a melodic
    template (phrase motif) guiding the generator.

    Presets
    -------
    baroque   : walking bass, running sixteenths, ornaments, high counterpoint
    jazz      : walking bass, countermelody, strong swing
    rock      : rock bass, block chords, metronomic, no ornaments
    classical : alberti bass, block chords, moderate counterpoint
    folk      : arpeggiated bass, countermelody, stepwise melody, ornaments
    ambient   : sustained bass, block chords, low density, sparse
    """
    PRESETS: dict[str, dict] = {
        "baroque": dict(
            bass_type="walking", inner_voice_style="running_sixteenths",
            swing=0.0, use_ornaments=True, counterpoint=0.85,
            step_leap_ratio=0.72, rhythmic_density=0.78,
        ),
        "jazz": dict(
            bass_type="walking", inner_voice_style="countermelody",
            swing=0.65, use_ornaments=False, counterpoint=0.45,
            step_leap_ratio=0.50,
        ),
        "rock": dict(
            bass_type="rock", inner_voice_style="block_chords",
            swing=0.0, use_ornaments=False, rhythmic_regularity=0.75,
            counterpoint=0.20,
        ),
        "classical": dict(
            bass_type="alberti", inner_voice_style="block_chords",
            swing=0.0, use_ornaments=False, counterpoint=0.25,
            step_leap_ratio=0.65,
        ),
        "folk": dict(
            bass_type="arpeggiated", inner_voice_style="countermelody",
            swing=0.0, use_ornaments=True, step_leap_ratio=0.80,
            counterpoint=0.55,
        ),
        "ambient": dict(
            bass_type="sustained", inner_voice_style="block_chords",
            swing=0.0, use_ornaments=False, rhythmic_density=0.15,
            counterpoint=0.1,
        ),
    }

    if style not in PRESETS:
        raise ValueError(
            f"Unknown style preset '{style}'. "
            f"Available: {sorted(PRESETS)}"
        )

    result = deepcopy(analysis)
    target_ids = set(section_ids) if section_ids else None

    for section in result.sections:
        if target_ids is not None and section.id not in target_ids:
            continue
        for k, v in PRESETS[style].items():
            if hasattr(section.texture, k):
                setattr(section.texture, k, v)
        section.generation_mode = "arrange"

    return result


# ── Motif transforms ──────────────────────────────────────────────────────────

def assign_motif(
    analysis: "MusicalAnalysis",
    motif_id: str,
    section_ids: list[str] | None = None,
) -> "MusicalAnalysis":
    """
    Assign a motif (by id) to one or more sections.

    section_ids : list of ids to update; None = all sections.
    The motif must already exist in analysis.motifs.
    """
    if motif_id not in analysis.motifs:
        raise KeyError(f"Motif '{motif_id}' not found in analysis.motifs")

    result = deepcopy(analysis)
    target_ids = set(section_ids) if section_ids else None

    for section in result.sections:
        if target_ids is None or section.id in target_ids:
            section.motif_id = motif_id

    return result


def augment_motif(
    analysis: "MusicalAnalysis",
    motif_id: str,
    factor: float = 2.0,
) -> "MusicalAnalysis":
    """
    Stretch all durations of a motif by factor (augmentation).

    A factor > 1 makes it slower; < 1 makes it faster (diminution).
    Transformations (inverted, retrograde, …) are unchanged since they
    share the same durations list.
    """
    result = deepcopy(analysis)
    if motif_id in result.motifs:
        m = result.motifs[motif_id]
        m.durations = [d * factor for d in m.durations]
    return result


# ── Harmonic development ──────────────────────────────────────────────────────

def develop_harmony(
    analysis: "MusicalAnalysis",
    complexity_arc: str = "flat",
    rhythm_arc: str = "flat",
) -> "MusicalAnalysis":
    """
    Scale chord_complexity and harmonic_rhythm progressively across sections.

    Only sections whose harmonic_plan is empty are affected — sections with a
    pre-built plan are left untouched.  The arc functions map the section index
    (0 = first, 1 = last) to a value in the given range.

    complexity_arc / rhythm_arc:
        "flat"       — no override applied
        "ascending"  — rises from low to high across sections
        "descending" — falls from high to low
        "arch"       — peaks at the middle section

    Example — build tension to a climax then release::

        analysis = develop_harmony(
            analysis,
            complexity_arc="arch",   # simple -> chromatic -> simple
            rhythm_arc="ascending",  # slow harmonic rhythm -> fast at climax
        )
    """
    result = deepcopy(analysis)
    n = len(result.sections)
    if n < 2:
        return result

    def _arc(idx: int, arc: str, lo: float, hi: float):
        t = idx / (n - 1)
        if arc == "ascending":  return lo + t * (hi - lo)
        if arc == "descending": return hi - t * (hi - lo)
        if arc == "arch":       return lo + (1 - abs(2 * t - 1)) * (hi - lo)
        return None  # flat -> no override

    for i, section in enumerate(result.sections):
        if section.harmonic_plan:   # pre-built — leave untouched
            continue
        c_val = _arc(i, complexity_arc, 0.20, 0.80)
        r_val = _arc(i, rhythm_arc,     0.25, 0.80)
        if c_val is not None:
            section.extra_params.setdefault("chord_complexity", round(c_val, 3))
        if r_val is not None:
            section.extra_params.setdefault("harmonic_rhythm",  round(r_val, 3))

    return result


def modulate_section(
    analysis: "MusicalAnalysis",
    section_id: str,
    key: str,
    mode: str | None = None,
) -> "MusicalAnalysis":
    """
    Override the tonal centre for a specific section.

    Sets ``key`` (and optionally ``mode``) in the section's ``extra_params``,
    which ``generate_section()`` applies after the global analysis key/mode.
    Any pre-built harmonic_plan is cleared so HarmonicPlanGenerator rebuilds
    it in the new key.

    Example — bridge in the relative major::

        analysis = modulate_section(analysis, "bridge", key="C", mode="major")
    """
    result = deepcopy(analysis)
    for section in result.sections:
        if section.id == section_id:
            section.extra_params["key"] = key
            if mode is not None:
                section.extra_params["mode"] = mode
            section.harmonic_plan = []   # regenerate in new key
            section.voice_sequences = {}  # degree encoding is key-relative; different key → wrong pitches
    return result


# ── Voice-sequence transforms ─────────────────────────────────────────────────

def voice_exchange(
    analysis: "MusicalAnalysis",
    section_id: str,
    voice1: str = "soprano",
    voice2: str = "alto",
) -> "MusicalAnalysis":
    """
    Swap the voice_sequences of two voices in a section.

    Classic invertible-counterpoint operation: what soprano sang, alto now
    sings and vice versa.  Works on any pair of named voices (soprano, alto,
    bass, inner_N, etc.).
    """
    result = deepcopy(analysis)
    for section in result.sections:
        if section.id != section_id:
            continue
        seqs = section.voice_sequences
        if voice1 in seqs and voice2 in seqs:
            seqs[voice1], seqs[voice2] = deepcopy(seqs[voice2]), deepcopy(seqs[voice1])
            section.generation_mode = "replay"
    return result


def invert_melody(
    analysis: "MusicalAnalysis",
    section_id: str,
    voice: str = "soprano",
) -> "MusicalAnalysis":
    """
    Mirror the melodic contour of a voice: every ascending interval becomes
    descending by the same number of semitones (melodic inversion).

    The pivot is the first pitched note of the voice in that section.  All
    subsequent notes are reflected: 2*pivot_midi − original_midi.

    Operates on voice_sequences via decode → invert → re-encode so that
    subsequent transpose() and mode_swap() continue to work correctly.
    """
    result = deepcopy(analysis)
    for section in result.sections:
        if section.id != section_id:
            continue
        if voice not in section.voice_sequences:
            continue
        tpc, sivs = _section_tonal(section, result)
        ev_list   = section.voice_sequences[voice]

        # Find pivot (first pitched note)
        pivot = None
        for ev in ev_list:
            m = _decode_ev(ev, tpc, sivs)
            if m is not None:
                pivot = m
                break
        if pivot is None:
            continue

        new_evs = []
        for ev in ev_list:
            if ev.get("degree") == "rest" or "pitch" in ev:
                new_evs.append(dict(ev))
                continue
            midi = _decode_ev(ev, tpc, sivs)
            if midi is None:
                new_evs.append(dict(ev))
                continue
            inv_midi = max(0, min(127, 2 * pivot - midi))
            new_ev   = _encode_ev(inv_midi, tpc, sivs)
            new_ev["duration"] = ev["duration"]
            new_ev["velocity"] = ev.get("velocity", 80)
            new_evs.append(new_ev)

        section.voice_sequences[voice] = new_evs
        section.generation_mode = "replay"
    return result


def retrograde_melody(
    analysis: "MusicalAnalysis",
    section_id: str,
    voice: str = "soprano",
) -> "MusicalAnalysis":
    """
    Retrograde the pitched events of a voice: the pitch order is reversed
    while durations remain in their original positions.

    Rest events are left in place; only the degree/alter/octave of pitched
    notes is reversed so the rhythm and section length are preserved.
    """
    result = deepcopy(analysis)
    for section in result.sections:
        if section.id != section_id:
            continue
        if voice not in section.voice_sequences:
            continue
        ev_list = section.voice_sequences[voice]

        pitched_pitch_info = [
            {k: ev[k] for k in ("degree", "alter", "octave") if k in ev}
            for ev in ev_list
            if ev.get("degree") not in ("rest",) and "pitch" not in ev
        ]
        if len(pitched_pitch_info) < 2:
            continue

        reversed_info = list(reversed(pitched_pitch_info))
        info_iter     = iter(reversed_info)
        new_evs = []
        for ev in ev_list:
            if ev.get("degree") == "rest" or "pitch" in ev:
                new_evs.append(dict(ev))
            else:
                new_ev = dict(ev)
                new_ev.update(next(info_iter, {}))
                new_evs.append(new_ev)

        section.voice_sequences[voice] = new_evs
        section.generation_mode = "replay"
    return result


def octave_shift(
    analysis: "MusicalAnalysis",
    section_id: str,
    voice: str,
    delta: int = 1,
) -> "MusicalAnalysis":
    """
    Shift all pitched notes in a voice up or down by delta octaves.

    delta=+1 : up one octave.
    delta=-1 : down one octave.

    Only the ``octave`` field of each pitched event is changed; degree and
    alter are unchanged, so subsequent harmonic transforms still work.
    """
    result = deepcopy(analysis)
    for section in result.sections:
        if section.id != section_id:
            continue
        if voice not in section.voice_sequences:
            continue
        new_evs = []
        for ev in section.voice_sequences[voice]:
            new_ev = dict(ev)
            if ev.get("degree") != "rest" and "pitch" not in ev:
                new_ev["octave"] = ev.get("octave", 4) + delta
            new_evs.append(new_ev)
        section.voice_sequences[voice] = new_evs
    return result


def harmonize_in_thirds(
    analysis: "MusicalAnalysis",
    section_id: str,
    voice: str = "soprano",
    interval_degrees: int = -2,
    target_voice: str | None = None,
) -> "MusicalAnalysis":
    """
    Add a parallel harmonization of a voice at a scale-degree interval.

    interval_degrees=-2 : a diatonic third below (default).
    interval_degrees=+2 : a diatonic third above.

    The harmony voice is inserted into voice_sequences so it replays
    alongside the original.  Each harmony note is 10–15 velocity units softer
    than the source note.

    target_voice : name for the new voice (default: "harmony_{voice}").
    """
    result = deepcopy(analysis)
    for section in result.sections:
        if section.id != section_id:
            continue
        if voice not in section.voice_sequences:
            continue
        ev_list    = section.voice_sequences[voice]
        new_name   = target_voice or f"harmony_{voice}"
        new_evs    = []
        for ev in ev_list:
            new_ev = dict(ev)
            deg = ev.get("degree")
            if deg != "rest" and "pitch" not in ev and isinstance(deg, int) and deg != 0:
                new_deg = deg + interval_degrees
                oct_delta = 0
                while new_deg < 1:
                    new_deg  += 7
                    oct_delta -= 1
                while new_deg > 7:
                    new_deg  -= 7
                    oct_delta += 1
                new_ev = {
                    "degree":   new_deg,
                    "alter":    ev.get("alter", 0),
                    "octave":   ev.get("octave", 4) + oct_delta,
                    "duration": ev["duration"],
                    "velocity": max(30, ev.get("velocity", 80) - 15),
                }
            new_evs.append(new_ev)
        section.voice_sequences[new_name] = new_evs
        section.generation_mode = "replay"
    return result


def rhythmic_augmentation(
    analysis: "MusicalAnalysis",
    section_id: str,
    voice: str = "soprano",
    factor: float = 2.0,
) -> "MusicalAnalysis":
    """
    Scale all note and rest durations of a voice by factor.

    factor=2.0 : augmentation — notes twice as long (classic development).
    factor=0.5 : diminution  — notes twice as fast.

    Both pitched events and rests are scaled, so the voice remains
    internally consistent, though it will no longer be the same total
    length as other unmodified voices.
    """
    result = deepcopy(analysis)
    for section in result.sections:
        if section.id != section_id:
            continue
        if voice not in section.voice_sequences:
            continue
        new_evs = [
            dict(ev, duration=round(ev["duration"] * factor, 4))
            for ev in section.voice_sequences[voice]
        ]
        section.voice_sequences[voice] = new_evs
        section.generation_mode = "replay"
    return result


def tempo_scale(analysis: "MusicalAnalysis", factor: float) -> "MusicalAnalysis":
    """
    Scale the global tempo by factor.

    factor=2.0 → twice as fast.  factor=0.5 → half speed.

    voice_sequences store durations in beats (tempo-independent), so they
    need no update — they automatically play correctly at the new tempo.
    """
    result = deepcopy(analysis)
    result.tempo_bpm = max(20, min(300, int(result.tempo_bpm * factor)))
    return result


# ── Motivic development transforms ───────────────────────────────────────────

def sequence_motif(
    analysis: "MusicalAnalysis",
    motif_id: str,
    n_steps: int = 3,
    step_semitones: int = 2,
) -> "MusicalAnalysis":
    """
    Build a sequential development of a motif and register it as a new motif.

    The new motif (id = "{motif_id}_seq") consists of the original motif
    repeated n_steps times, with each copy transposed step_semitones higher
    (or lower if negative) than the previous.  A bridge interval connects
    consecutive copies.

    Use assign_motif() to deploy the sequenced motif to sections::

        analysis = sequence_motif(analysis, "motif_0", n_steps=3, step_semitones=2)
        analysis = assign_motif(analysis, "motif_0_seq", ["chorus_1"])
        analysis = change_texture(analysis, "chorus_1", ...)  # optional

    step_semitones > 0 → ascending sequence (e.g., +2 = Rosalia sequence).
    step_semitones < 0 → descending sequence (common in development sections).
    """
    from music_generator.ir import MotifDef

    result = deepcopy(analysis)
    if motif_id not in result.motifs:
        raise KeyError(f"Motif '{motif_id}' not found in analysis.motifs")

    m = result.motifs[motif_id]
    total_rise = sum(m.intervals)   # net semitone displacement of one copy

    seq_intervals: list[int]   = []
    seq_durations: list[float] = []

    for step in range(n_steps):
        seq_intervals.extend(m.intervals)
        seq_durations.extend(m.durations[:-1])   # all notes but last of each copy
        if step < n_steps - 1:
            # Bridge: from end of this copy to start of next copy
            bridge = step_semitones - total_rise
            seq_intervals.append(bridge)
            # Bridge note takes the last note's duration of this copy
            seq_durations.append(m.durations[-1])

    seq_durations.append(m.durations[-1])   # final note of last copy

    new_id = f"{motif_id}_seq"
    result.motifs[new_id] = MotifDef(
        id            = new_id,
        type          = "sequence",
        intervals     = seq_intervals,
        durations     = seq_durations,
        sequence_step = step_semitones,
    )
    return result


def fragment_motif(
    analysis: "MusicalAnalysis",
    motif_id: str,
    start: int = 0,
    length: int = 2,
) -> "MusicalAnalysis":
    """
    Extract a sub-fragment from a motif and register it as a new motif.

    start  : 0-based index of the first interval to include.
    length : number of intervals in the fragment (fragment has length+1 notes).

    The new motif id is "{motif_id}_frag{start}".  Use assign_motif() to
    deploy it, or develop_harmony() to use it as a generative seed.

    Example — take just the opening leap of motif_0::

        analysis = fragment_motif(analysis, "motif_0", start=0, length=2)
        analysis = assign_motif(analysis, "motif_0_frag0", ["verse_2"])
    """
    from music_generator.ir import MotifDef

    result = deepcopy(analysis)
    if motif_id not in result.motifs:
        raise KeyError(f"Motif '{motif_id}' not found")

    m = result.motifs[motif_id]
    n  = len(m.intervals)
    start  = max(0, min(start, n - 1))
    length = max(1, min(length, n - start))

    new_id = f"{motif_id}_frag{start}"
    result.motifs[new_id] = MotifDef(
        id        = new_id,
        type      = "motif",
        intervals = m.intervals[start:start + length],
        durations = m.durations[start:start + length + 1],
    )
    return result


def derive_countermelody(
    analysis: "MusicalAnalysis",
    section_id: str,
    source_voice: str = "soprano",
    target_voice: str = "countermelody",
    offset_beats: float = 0.0,
) -> "MusicalAnalysis":
    """
    Derive a countermelody from a source voice using contrary motion.

    The countermelody is built by:
    1. Inverting the melodic contour (ascending → descending).
    2. Placing each note a diatonic third below the source (harmonically
       complementary range).
    3. Optionally offsetting the start by offset_beats (canon-like delay).

    The result is inserted into voice_sequences as target_voice and replays
    automatically alongside the original voices.
    """
    result = deepcopy(analysis)
    for section in result.sections:
        if section.id != section_id:
            continue
        if source_voice not in section.voice_sequences:
            continue
        tpc, sivs = _section_tonal(section, result)
        src_evs   = section.voice_sequences[source_voice]

        # Step 1: decode to MIDI pitches
        midi_pitches = [_decode_ev(ev, tpc, sivs) for ev in src_evs]

        # Step 2: find pivot for inversion (first pitched note)
        pivot = next((m for m in midi_pitches if m is not None), None)
        if pivot is None:
            continue

        # Step 3: build countermelody events
        counter_evs: list[dict] = []

        # Add offset rest if requested
        if offset_beats > 0:
            counter_evs.append({
                "degree": "rest", "alter": 0, "octave": 0,
                "duration": round(offset_beats, 4), "velocity": 0,
            })

        for ev, midi in zip(src_evs, midi_pitches):
            if ev.get("degree") == "rest" or "pitch" in ev or midi is None:
                counter_evs.append(dict(ev))
                continue
            # Invert around pivot, then drop a diatonic third (−2 degrees ≈ −3..−4 st)
            inv_midi   = max(0, min(127, 2 * pivot - midi - 3))
            new_pitch  = _encode_ev(inv_midi, tpc, sivs)
            counter_ev = {
                **new_pitch,
                "duration": ev["duration"],
                "velocity": max(30, ev.get("velocity", 80) - 10),
            }
            counter_evs.append(counter_ev)

        # Trim trailing offset rest if any (so total duration stays sensible)
        section.voice_sequences[target_voice] = counter_evs
        section.generation_mode = "replay"
    return result


# ── Composition helper ────────────────────────────────────────────────────────

def compose(
    analysis: "MusicalAnalysis",
    *transforms: Callable,
) -> "MusicalAnalysis":
    """
    Apply a sequence of transform functions left-to-right.

    Each transform must accept a MusicalAnalysis and return a new one.

    Example::

        result = compose(
            analysis,
            lambda a: transpose(a, -3),
            lambda a: mode_swap(a, "minor"),
            lambda a: apply_style_preset(a, "jazz"),
        )
    """
    result = analysis
    for fn in transforms:
        result = fn(result)
    return result


# ── Melodic variation transforms (delegated to _melodic_variation module) ─────
# Re-exported here so callers use a single import path.

from music_generator._melodic_variation import (   # noqa: E402
    vary_intervals,
    embellish_melody,
    rhythmic_displace,
    contour_remap,
    motif_substitute,
    melodic_development,
)
