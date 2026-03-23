"""
music_generator.compose — Multi-section composition API

generate_from_analysis() is the primary entry point.  It takes a MusicalAnalysis
IR, generates each section independently, and concatenates them into a single
MusicScore ready for MIDI export.

How injection works
-------------------
When SectionSpec.harmonic_plan is non-empty, InjectedHarmonicPlanGenerator
returns it directly (bypassing HarmonicPlanGenerator).  When SectionSpec.motif_id
is set, InjectedThemeGenerator returns the corresponding MotifDef's theme dict
(bypassing ThemeGenerator).  Both use Designer.override() so the rest of the
pipeline — voice leading, elaboration, score assembly — remains unchanged.

Example
-------
    from music_generator.ir import MusicalAnalysis, SectionSpec, TextureHints, EnergyProfile
    from music_generator.compose import generate_from_analysis
    from procgen import export

    analysis = MusicalAnalysis(
        key="A", mode="minor", tempo_bpm=100, time_signature=[4, 4],
        sections=[
            SectionSpec("intro",  "Intro",  "intro",  harmonic_plan=[],
                        texture=TextureHints(bass_type="sustained", rhythmic_density=0.3),
                        energy=EnergyProfile(level=0.3, arc="ascending"),
                        extra_params={"num_phrases": 1, "phrase_length": 4}),
            SectionSpec("verse",  "Verse",  "verse",  harmonic_plan=[],
                        texture=TextureHints(bass_type="walking", rhythmic_density=0.6),
                        energy=EnergyProfile(level=0.6, arc="arch"),
                        extra_params={"num_phrases": 2, "phrase_length": 4}),
            SectionSpec("chorus", "Chorus", "chorus", harmonic_plan=[],
                        texture=TextureHints(bass_type="rock", rhythmic_density=0.75),
                        energy=EnergyProfile(level=0.85, arc="arch"),
                        extra_params={"num_phrases": 2, "phrase_length": 4}),
        ],
    )

    base = dict(num_voices=3, instruments=["piano"], circle_of_fifths=0.65,
                chord_complexity=0.4, step_leap_ratio=0.7, use_ornaments=False,
                sequence_probability=0.3, melodic_direction="arch",
                rhythmic_regularity=0.6, swing=0.0, counterpoint=0.5,
                imitation=0.0, voice_independence=0.5, modal_mixture=0.1,
                pedal_point=0.0, melodic_range=2, melodic_theme="none",
                power_chords=False, ending="held", intro="none",
                form="binary", texture="polyphonic",
                num_phrases=2, phrase_length=4)

    score = generate_from_analysis(analysis, base_params=base, seed=42)
    midi  = export.to_midi(score)
"""

from __future__ import annotations

import sys
import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from music_generator.ir import MusicalAnalysis, SectionSpec

# ── Injector generator factories ─────────────────────────────────────────────
#
# build_plan() in MusicDomain only passes known schema keys into the params
# dict, so private "_injected_*" keys set via d.set() are silently dropped.
# The solution: factory functions that return a fresh generator *class* whose
# data is captured in a closure, so the generator never needs params at all.

def _make_plan_injector(harmonic_plan: list):
    """
    Return a generator class (compatible with Designer.override) that returns
    the given harmonic_plan verbatim, bypassing HarmonicPlanGenerator.
    """
    class _PlanInjector:
        def __init__(self, seed):
            self.seed = seed

        def generate(self, params: dict, context: dict) -> list:
            return harmonic_plan

    return _PlanInjector


def _make_theme_injector(theme_dict: dict):
    """
    Return a generator class (compatible with Designer.override) that returns
    the given theme dict verbatim, bypassing ThemeGenerator.
    """
    class _ThemeInjector:
        def __init__(self, seed):
            self.seed = seed

        def generate(self, params: dict, context: dict) -> dict:
            return theme_dict

    return _ThemeInjector


# ── Section generation ────────────────────────────────────────────────────────

def generate_section(
    section: "SectionSpec",
    analysis: "MusicalAnalysis",
    base_params: dict,
    seed: int = 0,
):
    """
    Generate one section and return its MusicScore.

    Parameter resolution order (later entries win):
        base_params
        → global context (key, mode, tempo, time_sig from analysis)
        → energy adjustments (rhythmic_density boost, arc → melodic_direction)
        → section.texture (TextureHints overrides)
        → section.extra_params (any remaining per-section intent values)

    If section.harmonic_plan is non-empty, HarmonicPlanGenerator is bypassed.
    If section.motif_id is set and found in analysis.motifs, ThemeGenerator
    is bypassed and the referenced MotifDef is injected as the theme.
    """
    # Resolve path to the procgen package so compose.py works standalone
    _ensure_paths()

    from procgen import Designer
    from music_generator import MusicDomain

    # ── 1. Start from base params ──
    params = dict(base_params)

    # ── 2. Global context always wins over base_params ──
    params["key"]            = analysis.key
    params["mode"]           = analysis.mode
    params["tempo_bpm"]      = analysis.tempo_bpm
    params["time_signature"] = list(analysis.time_signature)

    # ── 3. Energy adjustments ──
    energy = section.energy
    base_density = params.get("rhythmic_density", 0.5)
    boosted = base_density + energy.density_boost()
    params["rhythmic_density"] = max(0.05, min(0.95, boosted))
    if energy.arc in ("ascending", "descending", "arch"):
        params["melodic_direction"] = energy.arc

    # ── 4. Texture overrides ──
    params = section.texture.apply_to(params)

    # ── 5. Section-level extra params (highest priority) ──
    params.update(section.extra_params)

    # ── 6. Build designer and set all non-private params ──
    d = Designer(domain=MusicDomain(), seed=seed)
    for k, v in params.items():
        if not k.startswith("_"):
            d.set(k, v)

    # ── 7. Inject pre-built harmonic plan if present ──
    if section.harmonic_plan:
        d.override("harmonic_plan", _make_plan_injector(section.harmonic_plan))

    # ── 8. Inject motif if referenced ──
    motif_theme = analysis.get_motif_for_section(section)
    if motif_theme:
        d.override("theme", _make_theme_injector(motif_theme))

    return d.generate()


# ── Score concatenation ───────────────────────────────────────────────────────

def concatenate_scores(scores) -> object:
    """
    Merge a list of MusicScore objects into a single score.

    - Uses tempo and time_signature from the first score.
    - Aligns voices by role name (soprano, alto, tenor, bass).
    - Voices absent from a given section are padded with a silent rest
      spanning the section's duration (velocity-0 note).

    Returns a MusicScore.
    """
    from music_generator import MusicScore, Track, Note

    if not scores:
        raise ValueError("concatenate_scores: received empty list")
    if len(scores) == 1:
        return scores[0]

    first = scores[0]

    # Collect all voice roles that appear, preserving first-seen order
    all_roles: list[str] = []
    seen: set[str] = set()
    for score in scores:
        roles = score.metadata.get("voices", [t.instrument for t in score.tracks])
        for r in roles:
            if r not in seen:
                all_roles.append(r)
                seen.add(r)

    merged:      dict[str, list[Note]] = {r: [] for r in all_roles}
    instruments: dict[str, str]        = {}

    for score in scores:
        roles     = score.metadata.get("voices", [t.instrument for t in score.tracks])
        track_map = {
            role: score.tracks[i]
            for i, role in enumerate(roles)
            if i < len(score.tracks)
        }

        # Section duration = longest voice in this section
        section_beats = max(
            (sum(n.duration for n in t.notes) for t in track_map.values()),
            default=0.0,
        )

        for role in all_roles:
            if role in track_map:
                track = track_map[role]
                merged[role].extend(track.notes)
                instruments[role] = track.instrument
            elif section_beats > 0:
                # Silent rest to maintain alignment
                merged[role].append(Note("C4", section_beats, 0))

    tracks = [
        Track(instrument=instruments.get(r, "piano"), notes=merged[r])
        for r in all_roles
    ]

    return MusicScore(
        tempo_bpm      = first.tempo_bpm,
        time_signature = first.time_signature,
        tracks         = tracks,
        metadata={
            "intent":  first.metadata.get("intent", {}),
            "voices":  all_roles,
            "form":    "multi_section",
        },
    )


# ── Top-level API ─────────────────────────────────────────────────────────────

def generate_from_analysis(
    analysis: "MusicalAnalysis",
    base_params: dict,
    seed: int = 0,
):
    """
    Generate a full multi-section piece from a MusicalAnalysis IR.

    Each section in analysis.sections is generated independently via
    generate_section(), then all section scores are concatenated into one
    MusicScore by concatenate_scores().

    Parameters
    ----------
    analysis    MusicalAnalysis describing the full piece structure.
    base_params Intent parameters shared by all sections.  Individual sections
                can override these through their TextureHints and extra_params.
    seed        Base seed; each section receives seed + i*1000 for determinism.

    Returns
    -------
    MusicScore ready for export.to_midi().
    """
    scores = []
    for i, section in enumerate(analysis.sections):
        section_seed = seed + i * 1000
        score = generate_section(section, analysis, base_params, seed=section_seed)
        scores.append(score)
    return concatenate_scores(scores)


# ── Path helper ───────────────────────────────────────────────────────────────

def _ensure_paths() -> None:
    """
    Add the music_generator and procgen package roots to sys.path if needed.
    This lets compose.py be imported from any working directory.
    """
    here     = pathlib.Path(__file__).resolve().parent.parent   # .../music_generator/
    projects = here.parent                                        # .../proc gen projects/

    for p in [str(here), str(projects / "procgen")]:
        if p not in sys.path:
            sys.path.insert(0, p)
