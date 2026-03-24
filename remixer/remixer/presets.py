"""
remixer.presets — Named remix presets.

Each preset is a callable  (MusicalAnalysis) → MusicalAnalysis  that can be
passed directly to remix() or chained with other presets/transforms:

    from remixer import remix
    from remixer.presets import JAZZ, DARK_MINOR

    # Single preset
    midi = remix("grape-garden.mid", JAZZ, seed=42)

    # Chained — applied left-to-right
    midi = remix("grape-garden.mid", DARK_MINOR, JAZZ, seed=42)

    # Access by name
    from remixer.presets import PRESETS
    fn = PRESETS["epic"]
    midi = remix("final-boss.mid", fn, seed=0)

Available presets
-----------------
  jazz          Walking bass, strong swing, countermelody inner voices.
  baroque       Ornaments, running sixteenths, high counterpoint, no swing.
  classical     Alberti bass, block chords, moderate counterpoint.
  folk          Arpeggiated bass, stepwise melody, light ornaments.
  ambient       Sustained bass, very low density, slow feel.
  rock          Driving rock bass, block chords, metronomic.
  dark_minor    Parallel mode swap to harmonic minor (same key, darker colour).
  relative_minor Transpose to the relative minor key (−3 semitones + minor).
  epic          High energy across all sections, develop main motif as sequence.
  mirror_world  Invert the soprano melody in every section (melodic mirror).
  retrograde    Retrograde soprano pitch order per section (time reversal).
  kaleidoscope  Alternates invert/retrograde per section for a fractured effect.
  fragmented    Fragment the first motif to its opening cell + sequence it.
  reharmonized  Re-run harmonic generation on all sections with high complexity.
"""

from __future__ import annotations

from copy import deepcopy
from music_generator.transforms import (
    apply_style_preset,
    tempo_scale,
    mode_swap,
    transpose,
    change_energy,
    develop_harmony,
    reharmonize,
    sequence_motif,
    fragment_motif,
    invert_melody,
    retrograde_melody,
    octave_shift,
)


# ── Style presets ─────────────────────────────────────────────────────────────

def jazz(analysis):
    """
    Swing feel, walking bass, jazzy chord extensions, slightly relaxed tempo.

    At high source densities (> 2 n/beat), swing and countermelody are
    dialled back so the jazz style stays legible against a busy melody.
    """
    a = deepcopy(analysis)

    for sec in a.sections:
        density = _soprano_density(sec)

        if density > 2.0:
            hints = dict(
                bass_type="walking",        inner_voice_style="block_chords",
                swing=0.35,                 use_ornaments=False,
                counterpoint=0.25,          step_leap_ratio=0.55,
            )
        else:
            hints = dict(
                bass_type="walking",        inner_voice_style="countermelody",
                swing=0.65,                 use_ornaments=False,
                counterpoint=0.45,          step_leap_ratio=0.50,
            )

        for k, v in hints.items():
            if hasattr(sec.texture, k):
                setattr(sec.texture, k, v)
        sec.generation_mode = "arrange"

    a = tempo_scale(a, 0.90)
    return a


def _soprano_density(section) -> float:
    """
    Return the note density of a section's soprano voice in notes-per-beat.

    Used to decide how much contrapuntal activity to add: a dense source
    melody (e.g. Final Boss at 3-4 n/beat) cannot absorb running sixteenth
    counterpoint without becoming noise.  A sparse melody (< 0.8 n/beat)
    positively benefits from it.
    """
    sop_seq = section.voice_sequences.get("soprano", [])
    pitched  = [ev for ev in sop_seq if ev.get("degree") != "rest"]
    total_dur = sum(ev.get("duration", 0.5) for ev in sop_seq)
    return len(pitched) / total_dur if total_dur > 0.5 else 0.0


def baroque(analysis):
    """
    Baroque style adapted to source density.

    Rather than always adding running sixteenth counterpoint (which clashes
    with dense game-music melodies already at 3-4 notes/beat), the texture
    scales down as the source gets busier:

    Soprano density   Baroque texture chosen
    ─────────────────────────────────────────
    < 0.8  n/beat     Full baroque: running sixteenths + ornaments + high
                      counterpoint (Vivaldi-like; soprano is slow enough to
                      absorb it)
    0.8–2  n/beat     Moderate: countermelody + walking bass + light ornaments
                      (more Handel suite; soprano is already moving, so we
                      complement rather than compete)
    > 2    n/beat     Minimal: walking bass + block chords only
                      (Dense source like Final Boss / Rainbow Resort — any
                      fast counterpoint would create rhythmic noise)

    Tempo is capped at 140 BPM regardless of tier — baroque running
    sixteenths at 165 BPM are still too fast even when appropriate.
    """
    a = deepcopy(analysis)

    for sec in a.sections:
        density = _soprano_density(sec)

        if density > 2.0:
            # Dense soprano — just provide harmonic bass support
            hints = dict(
                bass_type="walking",        inner_voice_style="block_chords",
                swing=0.0,                  use_ornaments=False,
                counterpoint=0.15,          step_leap_ratio=0.65,
                rhythmic_density=0.35,
            )
        elif density > 0.8:
            # Moderate soprano — countermelody instead of running sixteenths
            hints = dict(
                bass_type="walking",        inner_voice_style="countermelody",
                swing=0.0,                  use_ornaments=True,
                counterpoint=0.55,          step_leap_ratio=0.72,
                rhythmic_density=0.55,
            )
        else:
            # Sparse soprano — full baroque texture (including 0 = fallback-generated)
            hints = dict(
                bass_type="walking",        inner_voice_style="running_sixteenths",
                swing=0.0,                  use_ornaments=True,
                counterpoint=0.85,          step_leap_ratio=0.72,
                rhythmic_density=0.78,
            )

        for k, v in hints.items():
            if hasattr(sec.texture, k):
                setattr(sec.texture, k, v)
        sec.generation_mode = "arrange"

    # Always cap tempo — prevents frantic sixteenths on fast source material
    _MAX_BPM = 140
    if a.tempo_bpm > _MAX_BPM:
        a = tempo_scale(a, _MAX_BPM / a.tempo_bpm)
    return a


def classical(analysis):
    """
    Classical style adapted to source density.

    Alberti bass against a melody playing at 3 notes/beat is
    indistinguishable from random noise.  Density tiers:

    Soprano density   Classical texture chosen
    ──────────────────────────────────────────
    < 0.8  n/beat     Full classical: alberti bass, block-chord inner voices,
                      moderate counterpoint
    0.8–2  n/beat     Simplified: walking bass, block chords, low counterpoint
    > 2    n/beat     Minimal: sustained bass, very sparse — let the melody
                      carry the piece

    Tempo capped at 130 BPM.
    """
    a = deepcopy(analysis)

    for sec in a.sections:
        density = _soprano_density(sec)

        if density > 2.0:
            hints = dict(
                bass_type="sustained",      inner_voice_style="block_chords",
                swing=0.0,                  use_ornaments=False,
                counterpoint=0.10,          step_leap_ratio=0.65,
                rhythmic_density=0.25,
            )
        elif density > 0.8:
            hints = dict(
                bass_type="walking",        inner_voice_style="block_chords",
                swing=0.0,                  use_ornaments=False,
                counterpoint=0.25,          step_leap_ratio=0.65,
                rhythmic_density=0.45,
            )
        else:
            hints = dict(
                bass_type="alberti",        inner_voice_style="block_chords",
                swing=0.0,                  use_ornaments=False,
                counterpoint=0.25,          step_leap_ratio=0.65,
                rhythmic_density=0.55,
            )

        for k, v in hints.items():
            if hasattr(sec.texture, k):
                setattr(sec.texture, k, v)
        sec.generation_mode = "arrange"

    _MAX_BPM = 130
    if a.tempo_bpm > _MAX_BPM:
        a = tempo_scale(a, _MAX_BPM / a.tempo_bpm)
    return a


def folk(analysis):
    """
    Arpeggiated bass, stepwise melody, light ornamentation.

    At high source densities the arpeggiated bass is replaced with a simpler
    walking bass so the fast arpeggios don't stack on top of an already busy
    melody.
    """
    a = deepcopy(analysis)

    for sec in a.sections:
        density = _soprano_density(sec)

        if density > 1.5:
            hints = dict(
                bass_type="walking",        inner_voice_style="countermelody",
                swing=0.0,                  use_ornaments=False,
                step_leap_ratio=0.75,       counterpoint=0.35,
            )
        else:
            hints = dict(
                bass_type="arpeggiated",    inner_voice_style="countermelody",
                swing=0.0,                  use_ornaments=True,
                step_leap_ratio=0.80,       counterpoint=0.55,
            )

        for k, v in hints.items():
            if hasattr(sec.texture, k):
                setattr(sec.texture, k, v)
        sec.generation_mode = "arrange"

    return a


def ambient(analysis):
    """Slow, sparse, sustained — half tempo, very low density."""
    a = apply_style_preset(analysis, "ambient")
    a = tempo_scale(a, 0.50)
    return a


def rock(analysis):
    """
    Driving rock bass, block chords, no swing, metronomic feel.

    At high source densities, block chords are removed (bass+drums only) so
    the rock drive doesn't compete with an already-dense melody.
    """
    a = deepcopy(analysis)

    for sec in a.sections:
        density = _soprano_density(sec)

        if density > 2.0:
            hints = dict(
                bass_type="rock",           inner_voice_style="block_chords",
                swing=0.0,                  use_ornaments=False,
                rhythmic_regularity=0.75,   counterpoint=0.10,
                rhythmic_density=0.40,
            )
        else:
            hints = dict(
                bass_type="rock",           inner_voice_style="block_chords",
                swing=0.0,                  use_ornaments=False,
                rhythmic_regularity=0.75,   counterpoint=0.20,
            )

        for k, v in hints.items():
            if hasattr(sec.texture, k):
                setattr(sec.texture, k, v)
        sec.generation_mode = "arrange"

    a = tempo_scale(a, 1.10)
    return a


def lofi(analysis):
    """
    Lo-fi chill: original melody + lazy swing bass, sparse accompaniment.

    Keeps the Kirby melody intact via arrange mode; replaces accompaniment
    with a slow-swing feel at 85 % of original tempo.
    """
    a = apply_style_preset(analysis, "jazz")   # sets arrange mode + jazz texture
    # Dial back the jazz energy to a lazy groove
    for sec in a.sections:
        sec.texture.swing           = 0.40
        sec.texture.rhythmic_density = 0.30
        sec.texture.use_ornaments   = False
    a = tempo_scale(a, 0.85)
    return a


def minimalist(analysis):
    """
    Minimalist: original melody + bare sustained bass pulses, no drums.

    Strips accompaniment to a single sustained bass note per chord change.
    """
    a = apply_style_preset(analysis, "ambient")  # sets arrange mode + ambient texture
    for sec in a.sections:
        sec.texture.rhythmic_density = 0.08
        sec.texture.counterpoint     = 0.0
    a = tempo_scale(a, 0.75)
    return a


# ── Tonal presets ─────────────────────────────────────────────────────────────

def dark_minor(analysis):
    """
    Parallel mode swap to harmonic minor.

    Keeps the same tonic (e.g. F major → F harmonic minor).  The raised 7th
    degree of harmonic minor gives the melody an exotic, tense quality.
    Voice sequences adapt automatically at decode time — no pitch data changes.
    """
    return mode_swap(analysis, "harmonic_minor")


def relative_minor(analysis):
    """
    Shift to the relative minor key (−3 semitones, natural minor mode).

    E.g. F major → D minor.  The tonal centre changes; voice sequence degrees
    are re-interpreted in the new tonic context.
    """
    a = transpose(analysis, -3)
    a = mode_swap(a, "minor")
    return a


# ── Energy / development presets ──────────────────────────────────────────────

def epic(analysis):
    """
    High-energy reworking.

    - All sections pushed to energy level 0.85+ with ascending arc.
    - Main motif sequenced upward in thirds (+4 semitones × 3 steps) to
      build climactic development material.
    - Harmonic complexity arcs upward across the form.
    """
    a = deepcopy(analysis)

    # Push energy up on every section
    for sec in a.sections:
        sec.energy.level = max(sec.energy.level, 0.82) + 0.05 * (
            ["intro", "verse", "chorus", "bridge", "outro"].index(sec.role)
            if sec.role in ("intro", "verse", "chorus", "bridge", "outro") else 0
        )
        sec.energy.level = min(sec.energy.level, 0.97)
        sec.energy.arc   = "ascending"
        sec.generation_mode = "generate"

    # Develop main motif as ascending sequence
    if a.motifs:
        first_motif_id = next(iter(a.motifs))
        a = sequence_motif(a, first_motif_id, n_steps=3, step_semitones=4)

    a = develop_harmony(a, complexity_arc="ascending", rhythm_arc="ascending")
    return a


# ── Voice-level presets ───────────────────────────────────────────────────────

def mirror_world(analysis):
    """
    Invert the soprano melody in every section that has one.

    Every ascending interval becomes descending by the same distance,
    mirrored around the opening note.  Harmony and rhythm unchanged.
    """
    a = deepcopy(analysis)
    for sec in a.sections:
        if "soprano" in sec.voice_sequences:
            a = invert_melody(a, sec.id, "soprano")
    return a


def retrograde(analysis):
    """
    Retrograde the soprano pitch order in every section.

    The sequence of pitches is reversed while note durations stay in their
    original positions, preserving rhythm while reversing the melodic arc.
    """
    a = deepcopy(analysis)
    for sec in a.sections:
        if "soprano" in sec.voice_sequences:
            a = retrograde_melody(a, sec.id, "soprano")
    return a


def kaleidoscope(analysis):
    """
    Alternates inversion and retrograde across sections for a fractured effect.

    Odd sections (0-indexed) get melodic inversion; even get retrograde.
    """
    a = deepcopy(analysis)
    for i, sec in enumerate(a.sections):
        if "soprano" not in sec.voice_sequences:
            continue
        if i % 2 == 0:
            a = invert_melody(a, sec.id, "soprano")
        else:
            a = retrograde_melody(a, sec.id, "soprano")
    return a


# ── Motivic development presets ───────────────────────────────────────────────

def fragmented(analysis):
    """
    Fragment the first detected motif to its opening 3-note cell, then
    sequence it in ascending steps of a perfect fourth (+5 semitones).

    Good for creating a sense of obsessive development from a small idea.
    """
    a = deepcopy(analysis)
    if not a.motifs:
        return a
    mid = next(iter(a.motifs))
    m   = a.motifs[mid]
    if len(m.intervals) >= 3:
        a = fragment_motif(a, mid, start=0, length=3)
        frag_id = f"{mid}_frag0"
        if frag_id in a.motifs:
            a = sequence_motif(a, frag_id, n_steps=4, step_semitones=5)
    return a


def reharmonized(analysis):
    """
    Re-run harmonic generation on all sections with high complexity.

    Strips existing chord plans and lets the harmonic generator rebuild them
    with richer chord vocabulary (chord_complexity=0.82).
    """
    a = deepcopy(analysis)
    for sec in a.sections:
        sec.harmonic_plan = []
        sec.extra_params["chord_complexity"] = 0.82
        sec.generation_mode = "generate"
    return a


# ── Registry ──────────────────────────────────────────────────────────────────

PRESETS: dict[str, callable] = {
    "jazz":           jazz,
    "baroque":        baroque,
    "classical":      classical,
    "folk":           folk,
    "ambient":        ambient,
    "rock":           rock,
    "lofi":           lofi,
    "minimalist":     minimalist,
    "dark_minor":     dark_minor,
    "relative_minor": relative_minor,
    "epic":           epic,
    "mirror_world":   mirror_world,
    "retrograde":     retrograde,
    "kaleidoscope":   kaleidoscope,
    "fragmented":     fragmented,
    "reharmonized":   reharmonized,
}

# Convenience aliases matching common import patterns
JAZZ           = jazz
BAROQUE        = baroque
CLASSICAL      = classical
FOLK           = folk
AMBIENT        = ambient
ROCK           = rock
LOFI           = lofi
MINIMALIST     = minimalist
DARK_MINOR     = dark_minor
RELATIVE_MINOR = relative_minor
EPIC           = epic
MIRROR_WORLD   = mirror_world
RETROGRADE     = retrograde
KALEIDOSCOPE   = kaleidoscope
FRAGMENTED     = fragmented
REHARMONIZED   = reharmonized
