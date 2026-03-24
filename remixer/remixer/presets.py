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
    assign_motif,
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
        density = _lead_density(sec)

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


def _melodic_lead(section) -> tuple:
    """
    Return (role, seq, pitched_events) for the primary melody voice.

    Soprano is preferred when it has ≥ 4 pitched notes — even if alto has more
    notes, soprano is the designated lead (it's the highest-pitched channel).

    When soprano is empty/sparse, selects the voice with the SLOWEST average
    note duration among voices that pass two filters:
      - avg octave ≥ 3  (lower = bass register)
      - avg note duration ≥ 0.20 beats  (shorter = arpeggio, not melody)
    The slowest-note voice is the melody; fast arpeggios have more notes but
    shorter durations and are inner-voice texture, not the lead melody.

    Returns ("soprano", [], []) when no qualifying lead is found (causing
    callers to fall back to full generation or density=0).
    """
    _RHYTHM  = {"bass", "drums"}
    _MIN_OCT = 3
    _MIN_DUR = 0.20

    # ── 1. Prefer soprano when it has meaningful content ──────────────────────
    sop_seq     = section.voice_sequences.get("soprano", [])
    sop_pitched = [ev for ev in sop_seq if ev.get("degree") != "rest"]
    if len(sop_pitched) >= 4:
        return "soprano", sop_seq, sop_pitched

    # ── 2. Soprano empty/sparse — find the most melodic other voice ───────────
    best_role    = "soprano"
    best_seq:    list = []
    best_pitched: list = []
    best_score   = -1.0

    for role, seq in section.voice_sequences.items():
        if role in _RHYTHM:
            continue
        pitched = [ev for ev in seq if ev.get("degree") != "rest"]
        if len(pitched) < 4:
            continue
        avg_oct = sum(ev.get("octave", 4) for ev in pitched) / len(pitched)
        avg_dur = sum(ev.get("duration", 0.5) for ev in pitched) / len(pitched)
        if avg_oct < _MIN_OCT or avg_dur < _MIN_DUR:
            continue
        # Melody score: rewards slow notes AND enough notes to form a full line.
        score = avg_dur * (len(pitched) ** 0.5)
        if score > best_score:
            best_score   = score
            best_role    = role
            best_seq     = seq
            best_pitched = pitched

    return best_role, best_seq, best_pitched


def _lead_voice_name(section) -> str:
    """Return the role name of the primary melodic voice (see _melodic_lead)."""
    role, _, _ = _melodic_lead(section)
    return role


def _lead_density(section) -> float:
    """
    Return the note density (notes-per-beat) of the primary melodic voice.

    A slow-note melody (e.g. soprano at 0.6 n/beat) can absorb more baroque
    counterpoint than a fast arpeggio (4 n/beat).  Uses soprano when present;
    falls back to the slowest-note melodic voice otherwise.  Returns 0.0 when
    no qualifying melody is found (triggering the full/sparse texture tier).
    """
    role, seq, pitched = _melodic_lead(section)
    if not pitched:
        return 0.0
    total_dur = sum(ev.get("duration", 0.5) for ev in seq)
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
        density = _lead_density(sec)

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
        density = _lead_density(sec)

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
        density = _lead_density(sec)

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
    a = deepcopy(analysis)
    for sec in a.sections:
        hints = dict(
            bass_type="sustained", inner_voice_style="block_chords",
            swing=0.0, use_ornaments=False,
            rhythmic_density=0.15, counterpoint=0.10,
        )
        for k, v in hints.items():
            if hasattr(sec.texture, k):
                setattr(sec.texture, k, v)
        sec.generation_mode = "arrange"
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
        density = _lead_density(sec)

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
    with a slow-swing feel at 85 % of original tempo.  Dense melodies
    (> 1.5 n/beat) get block chord pads instead of countermelody so the
    lo-fi vibe doesn't compete with a busy Kirby melody.
    """
    a = deepcopy(analysis)
    for sec in a.sections:
        density = _lead_density(sec)
        if density > 1.5:
            # Dense melody: pad chords only, no countermelody
            hints = dict(
                bass_type="walking", inner_voice_style="block_chords",
                swing=0.35, use_ornaments=False,
                counterpoint=0.10, rhythmic_density=0.25,
            )
        else:
            hints = dict(
                bass_type="walking", inner_voice_style="countermelody",
                swing=0.40, use_ornaments=False,
                counterpoint=0.30, rhythmic_density=0.30,
            )
        for k, v in hints.items():
            if hasattr(sec.texture, k):
                setattr(sec.texture, k, v)
        sec.generation_mode = "arrange"
    a = tempo_scale(a, 0.85)
    return a


def minimalist(analysis):
    """
    Minimalist: original melody + bare sustained bass pulses, no drums.

    Strips accompaniment to a single sustained bass note per chord change.
    Very sparse (density 0.08) so even dense Kirby melodies stand alone.
    """
    a = deepcopy(analysis)
    for sec in a.sections:
        hints = dict(
            bass_type="sustained", inner_voice_style="block_chords",
            swing=0.0, use_ornaments=False,
            rhythmic_density=0.08, counterpoint=0.0,
        )
        for k, v in hints.items():
            if hasattr(sec.texture, k):
                setattr(sec.texture, k, v)
        sec.generation_mode = "arrange"
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
    High-energy orchestral treatment — preserves Kirby melody.

    - All sections pushed to energy level 0.85+ with ascending arc.
    - Dense rock/orchestral accompaniment: driving bass, stacked chords,
      full drums.  Section density adapts so the accompaniment doesn't
      overwhelm a melody that's already dense.
    - Tempo boosted 10 % for extra drive.
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

        density = _lead_density(sec)
        if density > 2.0:
            # Already dense melody — support with bass + sparse chords only
            hints = dict(
                bass_type="rock", inner_voice_style="block_chords",
                swing=0.0, use_ornaments=False,
                counterpoint=0.05, rhythmic_density=0.50,
                rhythmic_regularity=0.85,
            )
        else:
            # Sparse/moderate melody — full epic treatment
            hints = dict(
                bass_type="rock", inner_voice_style="block_chords",
                swing=0.0, use_ornaments=False,
                counterpoint=0.15, rhythmic_density=0.70,
                rhythmic_regularity=0.85,
            )
        for k, v in hints.items():
            if hasattr(sec.texture, k):
                setattr(sec.texture, k, v)
        sec.generation_mode = "arrange"

    a = tempo_scale(a, 1.10)
    return a


# ── Voice-level presets ───────────────────────────────────────────────────────

def mirror_world(analysis):
    """
    Invert the melody in every section (uses the richest melodic voice).

    Every ascending interval becomes descending by the same distance,
    mirrored around the opening note.  Harmony and rhythm unchanged.
    Works on soprano, alto, or inner voices — whichever carries the melody.
    """
    a = deepcopy(analysis)
    for sec in a.sections:
        if not sec.voice_sequences:
            continue
        lead = _lead_voice_name(sec)
        a = invert_melody(a, sec.id, lead)
    return a


def retrograde(analysis):
    """
    Retrograde the melody pitch order in every section.

    The sequence of pitches is reversed while note durations stay in their
    original positions, preserving rhythm while reversing the melodic arc.
    Works on soprano, alto, or inner voices — whichever carries the melody.
    """
    a = deepcopy(analysis)
    for sec in a.sections:
        if not sec.voice_sequences:
            continue
        lead = _lead_voice_name(sec)
        a = retrograde_melody(a, sec.id, lead)
    return a


def kaleidoscope(analysis):
    """
    Alternates inversion and retrograde across sections for a fractured effect.

    Odd sections (0-indexed) get melodic inversion; even get retrograde.
    Works on soprano, alto, or inner voices — whichever carries the melody.
    """
    a = deepcopy(analysis)
    for i, sec in enumerate(a.sections):
        if not sec.voice_sequences:
            continue
        lead = _lead_voice_name(sec)
        if i % 2 == 0:
            a = invert_melody(a, sec.id, lead)
        else:
            a = retrograde_melody(a, sec.id, lead)
    return a


# ── Motivic development presets ───────────────────────────────────────────────

def fragmented(analysis):
    """
    Fragment the first motif to its 3-note opening cell, sequence it in
    ascending fourths, then assign it as the theme for every section.

    The original melody is replaced by a newly generated melody built from
    the fragment — bass/inner voices provide harmonic support.  This creates
    an obsessive, minimal development where a tiny motivic cell drives the
    whole piece.
    """
    a = deepcopy(analysis)
    if not a.motifs:
        return a
    mid = next(iter(a.motifs))
    m   = a.motifs[mid]
    # Build fragment + sequenced version
    theme_id = mid
    if len(m.intervals) >= 3:
        a = fragment_motif(a, mid, start=0, length=3)
        frag_id = f"{mid}_frag0"
        if frag_id in a.motifs:
            a = sequence_motif(a, frag_id, n_steps=4, step_semitones=5)
            seq_id = f"{frag_id}_seq"
            theme_id = seq_id if seq_id in a.motifs else frag_id
    # Assign the fragment as theme to every section and trigger generation
    a = assign_motif(a, theme_id)
    for sec in a.sections:
        sec.generation_mode = "generate"
    return a


def reharmonized(analysis):
    """
    Keep the original melody but rebuild harmonies with richer chord vocabulary.

    Strips existing chord plans and re-runs the harmonic generator at
    chord_complexity=0.82 (chromatic extensions, modal mixture, etc.).
    The melody is preserved via arrange mode; only bass and inner voices
    are regenerated against the new harmonic plan.
    """
    a = deepcopy(analysis)
    for sec in a.sections:
        sec.harmonic_plan = []
        sec.extra_params["chord_complexity"] = 0.82
        sec.generation_mode = "arrange"
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
