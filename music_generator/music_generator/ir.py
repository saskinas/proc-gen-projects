"""
music_generator.ir — Symbolic Intermediate Representation

MusicalAnalysis is the bridge between external sources (hand-authored specs,
future audio listener output, or remix transforms) and the generator pipeline.

Hierarchy
---------
MusicalAnalysis
    global context  : key, mode, tempo, time_signature
    motifs          : dict[id → MotifDef]
    sections        : list[SectionSpec]
        SectionSpec : label, role, harmonic_plan, motif_id, TextureHints, EnergyProfile

Compatibility
-------------
- SectionSpec.harmonic_plan holds list[HarmonicEvent] from music_generator.
  The import is deferred inside from_dict() to avoid a circular import at
  module load time; runtime usage is fully typed.
- MotifDef.to_theme_dict() returns the exact dict format that
  ElaborationGenerator expects, so no translation layer is needed.

Usage
-----
    from music_generator.ir import MusicalAnalysis, SectionSpec, MotifDef, TextureHints, EnergyProfile
    from music_generator.compose import generate_from_analysis
    from music_generator.transforms import mode_swap, transpose

    analysis = MusicalAnalysis(
        key="D", mode="minor", tempo_bpm=96, time_signature=[4, 4],
        sections=[
            SectionSpec(id="verse", label="Verse", role="verse",
                        harmonic_plan=[],          # let generator build it
                        texture=TextureHints(bass_type="walking"),
                        extra_params={"num_phrases": 2, "phrase_length": 4}),
            ...
        ],
    )
    score = generate_from_analysis(analysis, base_params={...}, seed=42)
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


# ── MotifDef ──────────────────────────────────────────────────────────────────

@dataclass
class MotifDef:
    """
    A melodic motif definition with its four canonical transformations.

    Stored as semitone intervals (not absolute pitches) so it can be applied
    at any transposition. Compatible with ThemeGenerator's output dict.

    intervals   : semitone offsets between consecutive notes  (len = notes − 1)
    durations   : note durations in beats                     (len = notes)
    """

    id: str
    type: str            # "motif" | "fugue_subject" | "custom"
    intervals: list[int]
    durations: list[float]

    # Transformations — auto-computed by __post_init__ if left empty
    inverted: list[int]             = field(default_factory=list)
    retrograde: list[int]           = field(default_factory=list)
    retrograde_inversion: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.inverted:
            self.inverted = [-i for i in self.intervals]
        if not self.retrograde:
            self.retrograde = list(reversed(self.intervals))
        if not self.retrograde_inversion:
            self.retrograde_inversion = list(reversed(self.inverted))

    # ── Compatibility ──────────────────────────────────────────────────────

    def to_theme_dict(self) -> dict:
        """
        Return the dict format consumed by ElaborationGenerator.
        Compatible drop-in for ThemeGenerator's output.
        """
        return {
            "type":      self.type,
            "intervals": self.intervals,
            "durations": self.durations,
            "inverted":  self.inverted,
            "retrograde": self.retrograde,
        }

    # ── Serialization ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "id":                   self.id,
            "type":                 self.type,
            "intervals":            self.intervals,
            "durations":            self.durations,
            "inverted":             self.inverted,
            "retrograde":           self.retrograde,
            "retrograde_inversion": self.retrograde_inversion,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MotifDef":
        return cls(
            id=d["id"],
            type=d["type"],
            intervals=d["intervals"],
            durations=d["durations"],
            inverted=d.get("inverted", []),
            retrograde=d.get("retrograde", []),
            retrograde_inversion=d.get("retrograde_inversion", []),
        )

    @classmethod
    def from_intervals(
        cls,
        id: str,
        type: str,
        intervals: list[int],
        durations: list[float],
    ) -> "MotifDef":
        """Convenience constructor — transformations are auto-computed."""
        return cls(id=id, type=type, intervals=intervals, durations=durations)


# ── TextureHints ──────────────────────────────────────────────────────────────

@dataclass
class TextureHints:
    """
    Per-section texture overrides applied on top of the base generation params.

    Every field defaults to None, meaning "inherit from base_params".
    Only non-None fields are written into the params dict for that section.
    """

    bass_type:          str | None   = None   # walking|alberti|pedal|arpeggiated|rock|sustained
    inner_voice_style:  str | None   = None   # block_chords|arpeggiated|countermelody|running_sixteenths
    rhythmic_density:   float | None = None   # 0.0 – 1.0
    swing:              float | None = None   # 0.0 – 1.0
    num_voices:         int | None   = None   # 1 – 4
    rhythmic_regularity: float | None = None  # 0.0 syncopated → 1.0 metronomic
    melodic_direction:  str | None   = None   # ascending|descending|arch|free
    use_ornaments:      bool | None  = None
    counterpoint:       float | None = None   # 0.0 – 1.0
    step_leap_ratio:    float | None = None   # 0.0 all leaps → 1.0 all steps
    instruments:        list | None  = None

    _FIELDS = (
        "bass_type", "inner_voice_style", "rhythmic_density", "swing",
        "num_voices", "rhythmic_regularity", "melodic_direction",
        "use_ornaments", "counterpoint", "step_leap_ratio", "instruments",
    )

    def apply_to(self, params: dict) -> dict:
        """Return a copy of params with this section's non-None fields applied."""
        p = dict(params)
        for f in self._FIELDS:
            val = getattr(self, f)
            if val is not None:
                p[f] = val
        return p

    def to_dict(self) -> dict:
        return {f: getattr(self, f) for f in self._FIELDS if getattr(self, f) is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "TextureHints":
        valid = {f for f in cls._FIELDS}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ── EnergyProfile ─────────────────────────────────────────────────────────────

@dataclass
class EnergyProfile:
    """
    Per-section energy shape.

    level : 0.0 = very sparse/quiet, 1.0 = dense/loud
    arc   : melodic contour direction for this section
    """

    level: float = 0.5
    arc:   str   = "arch"   # ascending | descending | arch | flat

    def density_boost(self) -> float:
        """Additive offset for rhythmic_density based on energy level (±0.3)."""
        return (self.level - 0.5) * 0.6

    def to_dict(self) -> dict:
        return {"level": self.level, "arc": self.arc}

    @classmethod
    def from_dict(cls, d: dict) -> "EnergyProfile":
        return cls(level=d.get("level", 0.5), arc=d.get("arc", "arch"))


# ── SectionSpec ───────────────────────────────────────────────────────────────

@dataclass
class SectionSpec:
    """
    One musical section: harmonic plan + texture + energy + optional motif.

    harmonic_plan
        list[HarmonicEvent] — if non-empty, bypasses HarmonicPlanGenerator.
        Pass an empty list to let the generator build harmony from key/mode/form.

    motif_id
        Key into MusicalAnalysis.motifs.  If set, bypasses ThemeGenerator and
        injects the referenced MotifDef as the melodic theme for this section.

    extra_params
        Any additional intent parameters specific to this section (e.g.,
        num_phrases=2, phrase_length=4, circle_of_fifths=0.8).
        These override base_params after TextureHints are applied.

    repeat_of
        Optional id of another SectionSpec this is a structural repeat of.
        Informational only — does not change generation behaviour.
    """

    id:           str
    label:        str           # "A", "verse", "chorus", "bridge", …
    role:         str           # intro | verse | chorus | bridge | outro | generic
    harmonic_plan: list         # list[HarmonicEvent]; loosely typed to avoid circular import

    motif_id:    str | None       = None
    texture:     TextureHints     = field(default_factory=TextureHints)
    energy:      EnergyProfile    = field(default_factory=EnergyProfile)
    repeat_of:   str | None       = None
    extra_params: dict            = field(default_factory=dict)

    # ── Serialization ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "id":          self.id,
            "label":       self.label,
            "role":        self.role,
            "harmonic_plan": [
                {
                    "root_pc":       ev.root_pc,
                    "chord_type":    ev.chord_type,
                    "duration_beats": ev.duration_beats,
                    "degree":        ev.degree,
                    "phrase_idx":    ev.phrase_idx,
                }
                for ev in self.harmonic_plan
            ],
            "motif_id":    self.motif_id,
            "texture":     self.texture.to_dict(),
            "energy":      self.energy.to_dict(),
            "repeat_of":   self.repeat_of,
            "extra_params": self.extra_params,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SectionSpec":
        # Deferred import avoids circular dependency at module load time
        from music_generator import HarmonicEvent  # noqa: PLC0415
        harmonic_plan = [
            HarmonicEvent(
                root_pc=ev["root_pc"],
                chord_type=ev["chord_type"],
                duration_beats=ev["duration_beats"],
                degree=ev["degree"],
                phrase_idx=ev.get("phrase_idx", 0),
            )
            for ev in d.get("harmonic_plan", [])
        ]
        return cls(
            id=d["id"],
            label=d["label"],
            role=d.get("role", "generic"),
            harmonic_plan=harmonic_plan,
            motif_id=d.get("motif_id"),
            texture=TextureHints.from_dict(d.get("texture", {})),
            energy=EnergyProfile.from_dict(d.get("energy", {})),
            repeat_of=d.get("repeat_of"),
            extra_params=d.get("extra_params", {}),
        )


# ── MusicalAnalysis ───────────────────────────────────────────────────────────

@dataclass
class MusicalAnalysis:
    """
    Top-level symbolic IR for a piece of music.

    Can represent:
    - A hand-authored multi-section composition spec
    - Output from a future audio listener/analyser
    - A procedurally transformed or remixed version of another analysis

    Feed into generate_from_analysis() (compose.py) to produce a MusicScore.
    """

    key:            str
    mode:           str
    tempo_bpm:      int
    time_signature: list[int]        # [numerator, denominator]
    sections:       list[SectionSpec]

    motifs:         dict[str, MotifDef] = field(default_factory=dict)
    form_hint:      str                 = "through_composed"

    # ── Convenience ────────────────────────────────────────────────────────

    def add_motif(self, motif: MotifDef) -> "MusicalAnalysis":
        """Register a motif and return self for chaining."""
        self.motifs[motif.id] = motif
        return self

    def get_motif_for_section(self, section: SectionSpec) -> dict | None:
        """
        Return the theme dict for ElaborationGenerator for this section,
        or None if the section has no motif_id or the id is not in motifs.
        """
        if section.motif_id and section.motif_id in self.motifs:
            return self.motifs[section.motif_id].to_theme_dict()
        return None

    def section_by_id(self, section_id: str) -> SectionSpec | None:
        for s in self.sections:
            if s.id == section_id:
                return s
        return None

    # ── Serialization ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "key":            self.key,
            "mode":           self.mode,
            "tempo_bpm":      self.tempo_bpm,
            "time_signature": self.time_signature,
            "form_hint":      self.form_hint,
            "motifs":         {k: v.to_dict() for k, v in self.motifs.items()},
            "sections":       [s.to_dict() for s in self.sections],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "MusicalAnalysis":
        motifs   = {k: MotifDef.from_dict(v) for k, v in d.get("motifs", {}).items()}
        sections = [SectionSpec.from_dict(s) for s in d.get("sections", [])]
        return cls(
            key=d["key"],
            mode=d["mode"],
            tempo_bpm=d["tempo_bpm"],
            time_signature=d["time_signature"],
            sections=sections,
            motifs=motifs,
            form_hint=d.get("form_hint", "through_composed"),
        )

    @classmethod
    def from_json(cls, s: str) -> "MusicalAnalysis":
        return cls.from_dict(json.loads(s))

    def __repr__(self) -> str:
        return (
            f"MusicalAnalysis({self.key} {self.mode}, {self.tempo_bpm} BPM, "
            f"{len(self.sections)} sections, {len(self.motifs)} motifs)"
        )
