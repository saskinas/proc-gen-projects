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

from copy import deepcopy
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from music_generator.ir import MusicalAnalysis, SectionSpec, TextureHints

from music_generator import NOTE_NAMES, ROOT_TO_PC, DIATONIC_TRIADS, degree_root_pc


# ── Harmonic transforms ───────────────────────────────────────────────────────

def transpose(analysis: "MusicalAnalysis", semitones: int) -> "MusicalAnalysis":
    """
    Transpose the entire analysis by the given number of semitones.

    Updates:
    - analysis.key (re-spelt using sharps)
    - Every HarmonicEvent.root_pc in every section's harmonic_plan
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
    Reinterpret the harmonic plan in a new mode.

    Keeps each HarmonicEvent's degree and phrase_idx intact; recalculates
    root_pc and chord_type from the scale degree in new_mode.

    This is the canonical "parallel mode" transform: e.g. a ii-V-I in C major
    becomes a ii°-V-i in C minor when mode_swap(analysis, "minor") is called.
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
    """
    result = deepcopy(analysis)
    for section in result.sections:
        if section.id == section_id:
            for k, v in hints.items():
                if hasattr(section.texture, k):
                    setattr(section.texture, k, v)
    return result


def change_energy(
    analysis: "MusicalAnalysis",
    section_id: str,
    level: float | None = None,
    arc: str | None = None,
) -> "MusicalAnalysis":
    """Set energy level and/or arc for a specific section."""
    result = deepcopy(analysis)
    for section in result.sections:
        if section.id == section_id:
            if level is not None:
                section.energy.level = max(0.0, min(1.0, level))
            if arc is not None:
                section.energy.arc = arc
    return result


def apply_style_preset(
    analysis: "MusicalAnalysis",
    style: str,
    section_ids: list[str] | None = None,
) -> "MusicalAnalysis":
    """
    Apply a named style preset to sections' TextureHints.

    section_ids : list of section ids to apply to; None = all sections.

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
