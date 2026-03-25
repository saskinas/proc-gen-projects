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
  jazz          Swing rhythm, walking bass fills, humanised timing.
  baroque       Mordent/turn ornaments, walking bass fills, no swing.
  classical     Alberti bass arpeggiation, clean melody, thinned inner voices.
  folk          Simple broken-chord bass, humanised timing.
  ambient       Thinned voices, soft dynamics, half tempo.
  rock          Driving eighth-note bass, backbeat accents, 110% tempo.
  dark_minor    Parallel mode swap to harmonic minor (same key, darker colour).
  relative_minor Transpose to the relative minor key (−3 semitones + minor).
  epic          High energy across all sections, develop main motif as sequence.
  mirror_world  Invert the soprano melody in every section (melodic mirror).
  retrograde    Retrograde soprano pitch order per section (time reversal).
  kaleidoscope  Alternates invert/retrograde per section for a fractured effect.
  fragmented    Fragment the first motif to its opening cell + sequence it.
  reharmonized  Re-run harmonic generation on all sections with high complexity.
  developed     Classical motivic development cycling techniques per section.
  embellished   Ornamental passing/neighbor tones with increasing density.
  reimagined    Same contour, different intervals — recognisable but fresh.
  evolved       Progressive interval expansion with motif family substitution.
  grooved       Rhythmic displacement focus — same pitches, different groove.
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
    vary_intervals,
    embellish_melody,
    rhythmic_displace,
    contour_remap,
    motif_substitute,
    melodic_development,
    _section_tonal,
)
from music_generator._voice_styler import style_voices


# ── Style presets ─────────────────────────────────────────────────────────────

def jazz(analysis):
    """
    Swing feel, walking bass, jazzy chord extensions, slightly relaxed tempo.

    Applies swing rhythm, walking bass fills, and humanisation directly to the
    existing voice sequences — all original voices are preserved with stylistic
    modification rather than being discarded and regenerated.
    """
    a = deepcopy(analysis)

    for sec in a.sections:
        lead = _lead_voice_name(sec)
        tpc, sivs = _section_tonal(sec, a)
        style_voices(sec, tpc, sivs, "jazz", lead_voice=lead)

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
    Baroque style: ornaments on melody, walking bass fills, no swing.

    Adds mordents/turns to the lead voice, fills bass rests with stepwise
    passing tones, and lightly ornaments inner voices.  All original voices
    are preserved.

    Tempo is capped at 140 BPM to keep ornaments playable.
    """
    a = deepcopy(analysis)

    for sec in a.sections:
        lead = _lead_voice_name(sec)
        tpc, sivs = _section_tonal(sec, a)
        style_voices(sec, tpc, sivs, "baroque", lead_voice=lead)

    _MAX_BPM = 140
    if a.tempo_bpm > _MAX_BPM:
        a = tempo_scale(a, _MAX_BPM / a.tempo_bpm)
    return a


def classical(analysis):
    """
    Classical style: Alberti bass arpeggiation, clean melody, thinned inner voices.

    Breaks sustained bass notes into Alberti-pattern arpeggios, keeps the
    melody untouched, and slightly thins inner voices for clarity.

    Tempo capped at 130 BPM.
    """
    a = deepcopy(analysis)

    for sec in a.sections:
        lead = _lead_voice_name(sec)
        tpc, sivs = _section_tonal(sec, a)
        style_voices(sec, tpc, sivs, "classical", lead_voice=lead)

    _MAX_BPM = 130
    if a.tempo_bpm > _MAX_BPM:
        a = tempo_scale(a, _MAX_BPM / a.tempo_bpm)
    return a


def folk(analysis):
    """
    Folk style: simple broken-chord bass arpeggiation, humanised melody.

    Breaks sustained bass notes into simple root-third-fifth arpeggios and
    adds subtle timing humanisation across all voices.
    """
    a = deepcopy(analysis)

    for sec in a.sections:
        lead = _lead_voice_name(sec)
        tpc, sivs = _section_tonal(sec, a)
        style_voices(sec, tpc, sivs, "folk", lead_voice=lead)

    return a


def ambient(analysis):
    """Slow, sparse, sustained — half tempo, thinned voices, soft dynamics."""
    a = deepcopy(analysis)
    for sec in a.sections:
        lead = _lead_voice_name(sec)
        tpc, sivs = _section_tonal(sec, a)
        style_voices(sec, tpc, sivs, "ambient", lead_voice=lead)
    a = tempo_scale(a, 0.50)
    return a


def rock(analysis):
    """
    Driving rock feel: repeated eighth-note bass, backbeat accents, louder dynamics.

    Converts sustained bass notes into driving eighth-note pulses, adds
    velocity accents on backbeats, and boosts tempo 10%.
    """
    a = deepcopy(analysis)

    for sec in a.sections:
        lead = _lead_voice_name(sec)
        tpc, sivs = _section_tonal(sec, a)
        style_voices(sec, tpc, sivs, "rock", lead_voice=lead)

    a = tempo_scale(a, 1.10)
    return a


def lofi(analysis):
    """
    Lo-fi chill: lazy swing, soft dynamics, humanised timing, 85% tempo.

    Applies gentle swing to all voices, reduces velocity for a softer feel,
    and thins inner voices slightly. All original voices preserved.
    """
    a = deepcopy(analysis)
    for sec in a.sections:
        lead = _lead_voice_name(sec)
        tpc, sivs = _section_tonal(sec, a)
        style_voices(sec, tpc, sivs, "lofi", lead_voice=lead)
    a = tempo_scale(a, 0.85)
    return a


def minimalist(analysis):
    """
    Minimalist: sparse voices, soft dynamics, reduced tempo.

    Heavily thins accompaniment voices, softens all dynamics, and slows
    tempo by 25%. The melody voice is preserved with lighter dynamics.
    """
    a = deepcopy(analysis)
    for sec in a.sections:
        lead = _lead_voice_name(sec)
        tpc, sivs = _section_tonal(sec, a)
        style_voices(sec, tpc, sivs, "minimalist", lead_voice=lead)
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
    High-energy treatment: driving bass, strong accents, boosted dynamics.

    Converts sustained bass to driving eighth-note pulses, applies downbeat
    accents across all voices, and boosts velocity for fuller dynamics.
    Tempo boosted 10%.
    """
    a = deepcopy(analysis)

    for sec in a.sections:
        lead = _lead_voice_name(sec)
        tpc, sivs = _section_tonal(sec, a)
        style_voices(sec, tpc, sivs, "epic", lead_voice=lead)

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
    # Collect (section_id, lead_voice) pairs before mutating — invert_melody
    # returns a fresh deepcopy each time, so iterating over a.sections directly
    # would leave sec pointing at stale objects after the first reassignment.
    targets = [
        (sec.id, _lead_voice_name(sec))
        for sec in a.sections
        if sec.voice_sequences
    ]
    for sec_id, lead in targets:
        a = invert_melody(a, sec_id, lead)
    return a


def retrograde(analysis):
    """
    Retrograde the melody pitch order in every section.

    The sequence of pitches is reversed while note durations stay in their
    original positions, preserving rhythm while reversing the melodic arc.
    Works on soprano, alto, or inner voices — whichever carries the melody.
    """
    a = deepcopy(analysis)
    targets = [
        (sec.id, _lead_voice_name(sec))
        for sec in a.sections
        if sec.voice_sequences
    ]
    for sec_id, lead in targets:
        a = retrograde_melody(a, sec_id, lead)
    return a


def kaleidoscope(analysis):
    """
    Alternates inversion and retrograde across sections for a fractured effect.

    Odd sections (0-indexed) get melodic inversion; even get retrograde.
    Works on soprano, alto, or inner voices — whichever carries the melody.
    """
    a = deepcopy(analysis)
    targets = [
        (i, sec.id, _lead_voice_name(sec))
        for i, sec in enumerate(a.sections)
        if sec.voice_sequences
    ]
    for i, sec_id, lead in targets:
        if i % 2 == 0:
            a = invert_melody(a, sec_id, lead)
        else:
            a = retrograde_melody(a, sec_id, lead)
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


# ── Melodic variation presets ────────────────────────────────────────────────

def developed(analysis):
    """
    Classical motivic development: each section gets a different treatment.

    Cycles through expand / contract / sequence_up / ornament / displace
    across sections so the melody evolves progressively.
    """
    a = deepcopy(analysis)
    techniques = ['contract', 'expand', 'sequence_up', 'ornament', 'displace', 'expand']
    targets = [
        (i, sec.id, _lead_voice_name(sec))
        for i, sec in enumerate(a.sections)
        if sec.voice_sequences
    ]
    for i, sec_id, lead in targets:
        tech = techniques[i % len(techniques)]
        a = melodic_development(a, sec_id, lead, technique=tech, seed=i * 7)
    return a


def embellished(analysis):
    """
    Add ornamental passing/neighbor tones throughout the melody.

    Density increases across sections: early sections get light ornamentation
    (20%), later sections get heavier embellishment (50%).  Rhythm is also
    slightly displaced in later sections for groove variety.
    """
    a = deepcopy(analysis)
    n = max(len(a.sections), 1)
    targets = [
        (i, sec.id, _lead_voice_name(sec))
        for i, sec in enumerate(a.sections)
        if sec.voice_sequences
    ]
    for i, sec_id, lead in targets:
        t = i / max(n - 1, 1)
        dens = 0.20 + t * 0.30
        a = embellish_melody(a, sec_id, lead, density=dens, seed=i * 13)
        if t > 0.5:
            a = rhythmic_displace(a, sec_id, lead,
                                  displacement=0.125, probability=0.25, seed=i * 17)
    return a


def reimagined(analysis):
    """
    Same contour, different intervals — a recognisable but fresh melody.

    Uses contour_remap to preserve the up/down shape of the original melody
    while replacing the interval sizes.  Early sections use whole-step motion
    (smooth), middle sections use thirds, and the final section widens to
    quartal intervals.  Combined with light embellishment and rhythmic
    displacement for Suno AI stems.
    """
    a = deepcopy(analysis)
    n = max(len(a.sections), 1)
    step_schedule = [2, 2, 3, 3, 4, 5]
    targets = [
        (i, sec.id, _lead_voice_name(sec))
        for i, sec in enumerate(a.sections)
        if sec.voice_sequences
    ]
    for i, sec_id, lead in targets:
        step = step_schedule[i % len(step_schedule)]
        a = contour_remap(a, sec_id, lead, step_size=step, seed=i * 11)
        if i % 2 == 1:
            a = embellish_melody(a, sec_id, lead, density=0.25, seed=i * 19)
        t = i / max(n - 1, 1)
        if t > 0.4:
            a = rhythmic_displace(a, sec_id, lead,
                                  displacement=0.125, probability=0.20, seed=i * 23)
    return a


def evolved(analysis):
    """
    Progressive interval expansion with motif substitution.

    Each section's melody intervals are scaled slightly wider than the
    previous (factor 1.0 -> 1.5).  Where motifs from the same contour
    family exist, occurrences are swapped with family members.
    """
    a = deepcopy(analysis)
    n = max(len(a.sections), 1)
    targets = [
        (i, sec.id, _lead_voice_name(sec))
        for i, sec in enumerate(a.sections)
        if sec.voice_sequences
    ]
    for i, sec_id, lead in targets:
        t = i / max(n - 1, 1)
        factor = 1.0 + t * 0.5
        a = vary_intervals(a, sec_id, lead, factor=factor, seed=i * 31)
        a = motif_substitute(a, sec_id, lead, seed=i * 37)
    return a


def grooved(analysis):
    """
    Rhythmic variation focus — same pitches, different groove.

    Early sections get light anticipations, middle sections get heavier
    syncopation, and the final section combines displacement with interval
    contraction for a tight, driving feel.
    """
    a = deepcopy(analysis)
    n = max(len(a.sections), 1)
    targets = [
        (i, sec.id, _lead_voice_name(sec))
        for i, sec in enumerate(a.sections)
        if sec.voice_sequences
    ]
    for i, sec_id, lead in targets:
        t = i / max(n - 1, 1)
        disp = 0.125 + t * 0.25
        prob = 0.25 + t * 0.30
        a = rhythmic_displace(a, sec_id, lead,
                              displacement=disp, probability=prob, seed=i * 41)
        if t > 0.65:
            a = vary_intervals(a, sec_id, lead, factor=0.8, seed=i * 43)
    return a



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
    "developed":      developed,
    "embellished":    embellished,
    "reimagined":     reimagined,
    "evolved":        evolved,
    "grooved":        grooved,
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
DEVELOPED      = developed
EMBELLISHED    = embellished
REIMAGINED     = reimagined
EVOLVED        = evolved
GROOVED        = grooved
