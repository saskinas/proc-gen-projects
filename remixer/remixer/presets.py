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
Style transforms (voice-level):
  jazz          Swing rhythm, walking bass fills, humanised timing.
  baroque       Mordent/turn ornaments, walking bass fills, no swing.
  classical     Alberti bass arpeggiation, clean melody, thinned inner voices.
  folk          Simple broken-chord bass, humanised timing.
  ambient       Thinned voices, soft dynamics, half tempo.
  rock          Driving eighth-note bass, backbeat accents, 110% tempo.
  lofi          Lazy swing, soft dynamics, humanised timing, 85% tempo.
  minimalist    Sparse voices, soft dynamics, reduced tempo.

Tonal transforms:
  dark_minor    Parallel mode swap to harmonic minor (same key, darker colour).
  relative_minor Transpose to the relative minor key (−3 semitones + minor).

Energy:
  epic          High energy across all sections, driving bass, strong accents.

Voice/melodic transforms:
  mirror_world  Invert the soprano melody in every section (melodic mirror).
  retrograde    Retrograde soprano pitch order per section (time reversal).
  kaleidoscope  Alternates invert/retrograde per section for a fractured effect.

Motivic development:
  fragmented    Fragment the first motif to its opening cell + sequence it.
  reharmonized  Re-run harmonic generation on all sections with high complexity.

Melodic variation:
  developed     Classical motivic development cycling techniques per section.
  embellished   Ornamental passing/neighbor tones with increasing density.
  reimagined    Same contour, different intervals — recognisable but fresh.
  evolved       Progressive interval expansion with motif family substitution.
  grooved       Rhythmic displacement focus — same pitches, different groove.

Inspired-by (new compositions from source DNA):
  inspired_by       New melodies over the source's harmonic progressions + motifs.
  inspired_sequel   New harmony + melodies seeded from the source's motifs.
  inspired_contrast Contrasting mood (opposite mode/energy), source motifs as seeds.

Accompaniment (generate music that plays alongside the original):
  accompany         Add missing voices (counter, bass, drums) around the original.
  duet              Add a countermelody voice — a duet partner for the melody.
  call_response     Re-arrange with responsive counterlines in melody gaps.
  harmonize         Add parallel harmony voices (3rds, 6ths) and supportive bass.
  backing_band      Full rhythm section: bass + drums + chords + countermelody.
  jazz_combo        Jazz trio accompaniment: walking bass, swing drums, comping.
  orchestral_accompany  Orchestral pad: strings, arpeggiated inner voices, deep bass.
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
from music_generator.ir import TextureHints, EnergyProfile
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
    Fragment the first motif to its opening cell, sequence it with ascending
    whole-step transpositions, then assign it as the theme for every section.

    The original melody is replaced by a newly generated melody built from
    the fragment — bass/inner voices provide harmonic support.  This creates
    an obsessive, minimal development where a tiny motivic cell drives the
    whole piece.

    Uses whole-step (2-semitone) transposition between copies to keep the
    bridge intervals small and musically smooth.
    """
    a = deepcopy(analysis)
    if not a.motifs:
        return a
    mid = next(iter(a.motifs))
    m   = a.motifs[mid]
    # Build fragment + sequenced version
    theme_id = mid
    frag_len = min(3, len(m.intervals)) if len(m.intervals) >= 2 else 0
    if frag_len >= 2:
        a = fragment_motif(a, mid, start=0, length=frag_len)
        frag_id = f"{mid}_frag0"
        if frag_id in a.motifs:
            # Use whole-step (2 semitones) transposition for smoother bridges.
            # The bridge interval = step_semitones - total_rise_of_fragment.
            # Whole steps keep bridges small regardless of fragment contour.
            a = sequence_motif(a, frag_id, n_steps=3, step_semitones=2)
            seq_id = f"{frag_id}_seq"
            theme_id = seq_id if seq_id in a.motifs else frag_id
    # Assign the fragment as theme and clear voice sequences so the generator
    # builds entirely fresh melodies from the fragment rather than being
    # influenced by the original stored sequences.
    a = assign_motif(a, theme_id)
    for sec in a.sections:
        sec.voice_sequences = {}
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



# ── Musical DNA extraction ──────────────────────────────────────────────────
#
# These helpers analyse source voice_sequences to extract rich musical
# characteristics — rhythmic patterns, melodic shape, step/leap ratio,
# ornament density, register, and energy contour.  The "inspired by"
# presets inject these characteristics into the generator parameters so
# the generated output *sounds like* it belongs with the original.


def _extract_musical_dna(section, analysis=None) -> dict:
    """
    Extract rich musical characteristics from a section's voice sequences.

    Parameters
    ----------
    section   SectionSpec with voice_sequences to analyse.
    analysis  Optional MusicalAnalysis — provides fallback key/mode when the
              section doesn't carry its own key/mode override in extra_params.

    Returns a dict of generation-parameter-ready values:
      rhythmic_density, rhythmic_regularity, step_leap_ratio,
      sequence_probability, use_ornaments, melodic_direction,
      bass_type, inner_voice_style, num_voices, counterpoint,
      drum_style, drum_intensity, swing,
      energy_level, energy_arc, instruments
    """
    vs = section.voice_sequences
    dna: dict = {}

    # ── Voice count ──
    primary_voices = [
        k for k in vs if k not in ("drums",)
        and not k.endswith("_layer_2")
        and not k.endswith("_layer_3")
        and not k.endswith("_layer_4")
    ]
    dna["num_voices"] = min(4, max(1, len(primary_voices)))

    # ── Lead melody analysis ──
    _, lead_seq, lead_pitched = _melodic_lead(section)
    if lead_pitched:
        durations = [ev.get("duration", 0.5) for ev in lead_pitched]
        total_dur = sum(ev.get("duration", 0.5) for ev in lead_seq)
        notes_per_beat = len(lead_pitched) / max(total_dur, 1.0)

        # Rhythmic density — from actual note rate
        dna["rhythmic_density"] = round(
            min(0.9, max(0.15, 0.1 + notes_per_beat * 0.18)), 2
        )

        # Rhythmic regularity — from duration variance
        # Low variance = metronomic, high variance = syncopated
        if len(durations) >= 3:
            mean_dur = sum(durations) / len(durations)
            variance = sum((d - mean_dur) ** 2 for d in durations) / len(durations)
            cv = (variance ** 0.5) / max(mean_dur, 0.01)  # coefficient of variation
            # CV near 0 = regular, CV > 1 = very irregular
            dna["rhythmic_regularity"] = round(max(0.1, min(0.95, 1.0 - cv * 0.7)), 2)
        else:
            dna["rhythmic_regularity"] = 0.6

        # Step/leap ratio — from actual intervals
        from music_generator import SCALE_INTERVALS, ROOT_TO_PC
        _fallback_key = analysis.key if analysis else "C"
        _fallback_mode = analysis.mode if analysis else "major"
        sec_key = section.extra_params.get("key", _fallback_key)
        sec_mode = section.extra_params.get("mode", _fallback_mode)
        tpc = ROOT_TO_PC.get(sec_key, 0)
        scale_ivs = SCALE_INTERVALS.get(sec_mode, SCALE_INTERVALS["major"])

        midi_pitches = []
        for ev in lead_pitched:
            deg = ev.get("degree", 1)
            if deg == "rest" or deg == 0:
                continue
            octave = ev.get("octave", 4)
            alter = ev.get("alter", 0)
            try:
                iv = scale_ivs[int(deg) - 1]
            except (IndexError, ValueError):
                iv = 0
            midi_pitches.append((octave + 1) * 12 + (tpc + iv + alter) % 12)

        if len(midi_pitches) >= 2:
            intervals = [abs(midi_pitches[i + 1] - midi_pitches[i])
                         for i in range(len(midi_pitches) - 1)]
            steps = sum(1 for iv in intervals if iv <= 2)
            dna["step_leap_ratio"] = round(steps / max(len(intervals), 1), 2)

            # Sequence probability — detect repeated interval patterns
            if len(intervals) >= 4:
                # Check for 2-note and 3-note patterns that repeat
                pattern_matches = 0
                for size in (2, 3):
                    for i in range(len(intervals) - size * 2 + 1):
                        pat = intervals[i:i + size]
                        nxt = intervals[i + size:i + size * 2]
                        if len(nxt) == size and all(
                            abs(a - b) <= 1 for a, b in zip(pat, nxt)
                        ):
                            pattern_matches += 1
                dna["sequence_probability"] = round(
                    min(0.8, max(0.1, pattern_matches / max(len(intervals), 1) * 3)), 2
                )
            else:
                dna["sequence_probability"] = 0.3

            # Melodic direction — from overall pitch contour
            if len(midi_pitches) >= 4:
                mid_idx = len(midi_pitches) // 2
                first_half_avg = sum(midi_pitches[:mid_idx]) / mid_idx
                second_half_avg = sum(midi_pitches[mid_idx:]) / (len(midi_pitches) - mid_idx)
                quarter_idx = len(midi_pitches) // 4
                middle_avg = sum(midi_pitches[quarter_idx:mid_idx + quarter_idx]) / max(
                    mid_idx, 1)
                overall_avg = sum(midi_pitches) / len(midi_pitches)

                if middle_avg > overall_avg + 1:
                    dna["melodic_direction"] = "arch"
                elif second_half_avg > first_half_avg + 2:
                    dna["melodic_direction"] = "ascending"
                elif first_half_avg > second_half_avg + 2:
                    dna["melodic_direction"] = "descending"
                else:
                    dna["melodic_direction"] = "arch"
        else:
            dna["step_leap_ratio"] = 0.65
            dna["sequence_probability"] = 0.3
            dna["melodic_direction"] = "arch"

        # Ornament detection — short note pairs (grace notes, mordents)
        short_pairs = 0
        for i in range(len(durations) - 1):
            if durations[i] < 0.2 and durations[i + 1] < 0.2:
                short_pairs += 1
        dna["use_ornaments"] = short_pairs >= 2

    else:
        dna["rhythmic_density"] = 0.45
        dna["rhythmic_regularity"] = 0.6
        dna["step_leap_ratio"] = 0.65
        dna["sequence_probability"] = 0.3
        dna["melodic_direction"] = "arch"
        dna["use_ornaments"] = False

    # ── Bass analysis ──
    bass_seq = vs.get("bass", [])
    bass_pitched = [ev for ev in bass_seq if ev.get("degree") != "rest"]
    if bass_pitched:
        bass_durs = [ev.get("duration", 0.5) for ev in bass_pitched]
        avg_bass_dur = sum(bass_durs) / len(bass_durs)
        if avg_bass_dur > 1.5:
            dna["bass_type"] = "sustained"
        elif avg_bass_dur > 0.8:
            dna["bass_type"] = "walking"
        elif avg_bass_dur > 0.4:
            dna["bass_type"] = "arpeggiated"
        else:
            dna["bass_type"] = "rock"
    else:
        dna["bass_type"] = "sustained"

    # ── Inner voice analysis ──
    inner_keys = [k for k in vs if k.startswith("inner_") or k == "alto"]
    if inner_keys:
        inner_all = []
        for k in inner_keys:
            inner_all.extend(ev for ev in vs[k] if ev.get("degree") != "rest")
        if inner_all:
            inner_durs = [ev.get("duration", 0.5) for ev in inner_all]
            avg_inner = sum(inner_durs) / len(inner_durs)
            if avg_inner < 0.2:
                dna["inner_voice_style"] = "arpeggiated"
            elif avg_inner < 0.5:
                dna["inner_voice_style"] = "running_sixteenths"
            elif avg_inner > 1.0:
                dna["inner_voice_style"] = "block_chords"
            else:
                dna["inner_voice_style"] = "countermelody"
        else:
            dna["inner_voice_style"] = "block_chords"
    else:
        dna["inner_voice_style"] = "block_chords"

    # ── Drums ──
    drum_seq = vs.get("drums", [])
    if drum_seq:
        drum_pitched = [ev for ev in drum_seq if ev.get("pitch", "rest") != "rest"
                        and ev.get("degree") != "rest"]
        if drum_pitched:
            drum_dur = sum(ev.get("duration", 0.5) for ev in drum_seq)
            hits_per_beat = len(drum_pitched) / max(drum_dur, 1.0)
            dna["drum_style"] = "rock"
            dna["drum_intensity"] = round(min(0.9, max(0.2, hits_per_beat * 0.25)), 2)
        else:
            dna["drum_style"] = "rock"
            dna["drum_intensity"] = 0.4
    else:
        dna["drum_style"] = "none"
        dna["drum_intensity"] = 0.0

    # ── Swing detection ──
    if lead_pitched and len(lead_pitched) >= 6:
        durations = [ev.get("duration", 0.5) for ev in lead_pitched]
        # Detect swing: alternating long-short pairs
        swing_pairs = 0
        total_pairs = 0
        for i in range(0, len(durations) - 1, 2):
            total_pairs += 1
            ratio = durations[i] / max(durations[i + 1], 0.01)
            if 1.3 < ratio < 2.5:  # swing ratio range
                swing_pairs += 1
        if total_pairs >= 3 and swing_pairs / total_pairs > 0.4:
            dna["swing"] = 0.55
        else:
            dna["swing"] = 0.0
    else:
        dna["swing"] = 0.0

    # ── Energy from velocity ──
    all_pitched = []
    for role, seq in vs.items():
        if role == "drums":
            continue
        all_pitched.extend(ev for ev in seq if ev.get("degree") != "rest")
    if all_pitched:
        velocities = [ev.get("velocity", 64) for ev in all_pitched]
        avg_vel = sum(velocities) / len(velocities)
        dna["energy_level"] = round(min(0.95, max(0.15, (avg_vel - 40) / 90)), 2)

        # Detect energy arc from velocity trend across the section
        if len(velocities) >= 8:
            q = len(velocities) // 4
            v1 = sum(velocities[:q]) / q
            v2 = sum(velocities[q:2 * q]) / q
            v3 = sum(velocities[2 * q:3 * q]) / q
            v4 = sum(velocities[3 * q:]) / max(len(velocities) - 3 * q, 1)

            if v2 > v1 + 3 and v3 > v2 + 3:
                dna["energy_arc"] = "ascending"
            elif v2 < v1 - 3 and v3 < v2 - 3:
                dna["energy_arc"] = "descending"
            elif v2 > v1 and v3 > v4:
                dna["energy_arc"] = "arch"
            else:
                dna["energy_arc"] = "arch"
        else:
            dna["energy_arc"] = "arch"
    else:
        dna["energy_level"] = 0.5
        dna["energy_arc"] = "arch"

    # ── Counterpoint from voice independence ──
    if dna["num_voices"] >= 3:
        dna["counterpoint"] = 0.5
    elif dna["num_voices"] == 2:
        dna["counterpoint"] = 0.3
    else:
        dna["counterpoint"] = 0.1

    # ── Instruments from source programs ──
    src_progs = section.extra_params.get("_source_programs", {})
    if src_progs:
        dna["instruments"] = [f"gm_prog_{p}" for p in src_progs.values()
                              if not isinstance(p, str)]

    return dna


def _apply_dna_to_section(sec, dna: dict, *, clear_harmony: bool = False):
    """Apply extracted musical DNA to a section for generation."""
    sec.texture = TextureHints(
        num_voices=dna.get("num_voices", 3),
        rhythmic_density=dna.get("rhythmic_density", 0.5),
        bass_type=dna.get("bass_type", "sustained"),
        inner_voice_style=dna.get("inner_voice_style", "block_chords"),
        drum_style=dna.get("drum_style", "none"),
        drum_intensity=dna.get("drum_intensity", 0.5),
        counterpoint=dna.get("counterpoint", 0.3),
        swing=dna.get("swing", 0.0),
        melodic_direction=dna.get("melodic_direction", "arch"),
        use_ornaments=dna.get("use_ornaments", False),
        step_leap_ratio=dna.get("step_leap_ratio", 0.65),
    )
    sec.energy = EnergyProfile(
        level=dna.get("energy_level", 0.5),
        arc=dna.get("energy_arc", "arch"),
    )
    sec.extra_params["rhythmic_regularity"] = dna.get("rhythmic_regularity", 0.6)
    sec.extra_params["sequence_probability"] = dna.get("sequence_probability", 0.3)
    # Force motif theme so the generator always applies the motif, not just sometimes
    sec.extra_params["melodic_theme"] = "motif"

    if dna.get("instruments"):
        sec.extra_params["instruments"] = dna["instruments"]

    if clear_harmony:
        sec.harmonic_plan = []

    sec.voice_sequences = {}
    sec.generation_mode = "generate"


def _estimate_texture_from_voices(section) -> TextureHints:
    """
    Infer appropriate TextureHints from the source's voice_sequences.
    Thin wrapper around _extract_musical_dna for backward compatibility.
    """
    dna = _extract_musical_dna(section, analysis=None)
    return TextureHints(
        num_voices=dna.get("num_voices", 3),
        rhythmic_density=dna.get("rhythmic_density", 0.5),
        bass_type=dna.get("bass_type", "sustained"),
        drum_style=dna.get("drum_style", "none"),
        drum_intensity=dna.get("drum_intensity", 0.5),
        counterpoint=dna.get("counterpoint", 0.3),
    )


def _estimate_energy(section) -> EnergyProfile:
    """Estimate energy profile from the source section's voice sequences."""
    dna = _extract_musical_dna(section, analysis=None)
    return EnergyProfile(
        level=dna.get("energy_level", 0.5),
        arc=dna.get("energy_arc", "arch"),
    )


def inspired_by(analysis):
    """
    Generate new melodies over the source's harmonic progressions and motifs.

    Extracts the full musical DNA from the source — rhythmic density,
    step/leap ratio, melodic direction, ornament usage, bass style,
    inner voice texture, drum intensity, swing, and energy arc — then
    regenerates all voices using these parameters.  The harmonic
    progressions and motifs are preserved, so the result sounds like
    an alternate version that belongs in the same soundtrack.

    Preserves: key, mode, tempo, time signature, section structure,
    harmonic plans, motifs.
    Transfers: rhythmic feel, melodic character, texture, energy.
    Generates: new melodies, countermelodies, bass lines, drums.
    """
    a = deepcopy(analysis)

    # Extract DNA from each section BEFORE clearing voices
    dna_per_section = [_extract_musical_dna(sec, analysis=a) for sec in a.sections]

    for sec, dna in zip(a.sections, dna_per_section):
        _apply_dna_to_section(sec, dna, clear_harmony=False)
        # Vary motif development across sections for variety
        roles = [s.role for s in a.sections]
        sec_idx = a.sections.index(sec)
        if sec.role == "intro" or sec_idx == 0:
            sec.extra_params["melodic_theme_dev"] = "build"
        elif sec.role == "chorus":
            sec.extra_params["melodic_theme_dev"] = "flat"
        elif sec.role == "bridge":
            sec.extra_params["melodic_theme_dev"] = "fragment"
        elif sec.role == "outro" or sec_idx == len(a.sections) - 1:
            sec.extra_params["melodic_theme_dev"] = "fragment"
        else:
            sec.extra_params["melodic_theme_dev"] = "sequence"

    return a


def inspired_sequel(analysis):
    """
    New harmony AND melodies seeded from the source's motifs.

    Transfers the complete musical DNA (rhythm, texture, energy, melodic
    character) but regenerates the harmonic progressions too.  The motifs
    provide thematic continuity — the generator develops the same motivic
    cells in each phrase — but the chord changes underneath are fresh.

    The harmonic parameters are tuned to produce interesting progressions:
    moderate complexity, strong circle-of-fifths motion, moderate harmonic
    rhythm.  The result sounds like a sequel — same world, new chapter.
    """
    a = deepcopy(analysis)

    dna_per_section = [_extract_musical_dna(sec, analysis=a) for sec in a.sections]

    for sec, dna in zip(a.sections, dna_per_section):
        _apply_dna_to_section(sec, dna, clear_harmony=True)
        sec.extra_params["chord_complexity"] = 0.55
        sec.extra_params["circle_of_fifths"] = 0.65
        sec.extra_params["harmonic_rhythm"] = min(
            0.7, dna.get("rhythmic_density", 0.5) * 0.9
        )
        # Vary development style per section
        sec_idx = a.sections.index(sec)
        if sec_idx == 0:
            sec.extra_params["melodic_theme_dev"] = "build"
        elif sec.role == "chorus":
            sec.extra_params["melodic_theme_dev"] = "flat"
        else:
            sec.extra_params["melodic_theme_dev"] = "sequence"

    return a


def inspired_contrast(analysis):
    """
    Contrasting mood — opposite mode, inverted energy, reflected motifs.

    Extracts the full musical DNA from the source but inverts the emotional
    character: major → harmonic minor, high energy → calm, ascending →
    descending.  Motif intervals are inverted (ascending phrases become
    descending), creating a "shadow" version.

    The rhythmic DNA (density, regularity, swing, bass style) is preserved
    so the contrast feels like it belongs to the same piece structurally.
    """
    a = deepcopy(analysis)

    # Flip mode
    mode_map = {
        "major": "harmonic_minor",
        "minor": "major",
        "harmonic_minor": "major",
        "dorian": "phrygian",
        "phrygian": "dorian",
        "lydian": "mixolydian",
        "mixolydian": "lydian",
        "pentatonic_major": "pentatonic_minor",
        "pentatonic_minor": "pentatonic_major",
        "blues": "major",
    }
    a.mode = mode_map.get(a.mode, "minor" if a.mode == "major" else "major")

    # Invert motif intervals (ascending → descending and vice versa)
    for mid, motif in a.motifs.items():
        motif.intervals = [-iv for iv in motif.intervals]
        motif.inverted = [-iv for iv in motif.inverted]
        motif.retrograde = list(reversed(motif.intervals))
        motif.retrograde_inversion = list(reversed(motif.inverted))
        motif.contour = [-c for c in motif.contour]

    dna_per_section = [_extract_musical_dna(sec, analysis=a) for sec in a.sections]

    for sec, dna in zip(a.sections, dna_per_section):
        # Invert energy
        dna["energy_level"] = round(max(0.2, min(0.9, 1.0 - dna["energy_level"])), 2)
        arc_map = {"ascending": "descending", "descending": "ascending",
                   "arch": "arch", "flat": "flat"}
        dna["energy_arc"] = arc_map.get(dna.get("energy_arc", "arch"), "arch")

        # Invert melodic direction
        dir_map = {"ascending": "descending", "descending": "ascending",
                   "arch": "arch", "free": "free"}
        dna["melodic_direction"] = dir_map.get(
            dna.get("melodic_direction", "arch"), "arch"
        )

        _apply_dna_to_section(sec, dna, clear_harmony=True)
        sec.extra_params["chord_complexity"] = 0.65
        sec.extra_params["modal_mixture"] = 0.3
        sec.extra_params["melodic_theme_dev"] = "fragment"

    return a


# ── Accompaniment presets ─────────────────────────────────────────────────────


def accompany(analysis):
    """
    Generate complementary accompaniment that plays alongside the original.

    Keeps ALL original voices intact and layers newly-generated parts on top:
    countermelody, harmony pad, bass fill, and/or drums — whichever the
    original is missing.  Generated parts use the source's harmonic plan
    so they fit perfectly, and are mixed at lower velocity so they sit
    behind the original.

    Works with both MIDI and audio-transcribed sources.
    """
    a = deepcopy(analysis)
    for sec in a.sections:
        sec.generation_mode = "accompany"
        # Let accompany_section auto-detect what to add
    return a


def duet(analysis):
    """
    Generate a countermelody voice that creates a duet with the original.

    Replays all original voices and adds one countermelody that harmonises
    with and responds to the source melody.  The countermelody uses the
    source's harmonic plan and motifs, creating a voice that sounds like
    it belongs with the original.

    High counterpoint and voice independence settings produce a melodically
    independent duet partner, not just block-chord harmony.
    """
    a = deepcopy(analysis)
    for sec in a.sections:
        sec.generation_mode = "accompany"
        sec.extra_params["_accompany_roles"] = ["countermelody"]
        # High counterpoint + independence for a true duet partner
        sec.extra_params["counterpoint"] = 0.75
        sec.extra_params["voice_independence"] = 0.8
        sec.extra_params["inner_voice_style"] = "countermelody"
        sec.extra_params["step_leap_ratio"] = 0.55  # allow leaps for melodic interest
        sec.extra_params["use_ornaments"] = True
        sec.extra_params["sequence_probability"] = 0.4
    return a


def call_response(analysis):
    """
    Generate response phrases that fill gaps in the original melody.

    Keeps the original melody and generates answering phrases using the
    source's motifs.  The response voice enters during rests and sustains
    in the lead melody, creating a conversational call-and-response texture.

    Uses arrange mode: the soprano (melody) is preserved; all other voices
    are regenerated with high voice independence and counterpoint settings
    to create responsive counterlines.
    """
    a = deepcopy(analysis)
    for sec in a.sections:
        sec.generation_mode = "arrange"
        # High voice independence so the generated voices feel like "responses"
        sec.extra_params["counterpoint"] = 0.75
        sec.extra_params["voice_independence"] = 0.8
        sec.extra_params["inner_voice_style"] = "countermelody"
        # Slightly lower density so responses don't crowd the melody
        # _lead_density returns notes/beat; map to 0.0-1.0 range for rhythmic_density
        raw_density = _lead_density(sec)
        mapped_density = min(0.85, max(0.2, 0.1 + raw_density * 0.18))
        sec.texture = TextureHints(
            rhythmic_density=mapped_density * 0.8,
            num_voices=min(4, (sec.texture.num_voices or 3)),
            counterpoint=0.75,
        )
    return a


def harmonize(analysis):
    """
    Add harmony voices (parallel 3rds and 6ths) to the original melody.

    Takes the soprano melody and generates closely-spaced harmony voices
    that follow the melody in diatonic parallel motion.  Also adds a
    supportive bass line and optional drums.

    The result sounds like a choir or ensemble arrangement of the melody.
    """
    a = deepcopy(analysis)
    for sec in a.sections:
        sec.generation_mode = "accompany"
        sec.extra_params["_accompany_roles"] = ["countermelody", "pad"]
        # Block chords create parallel harmony feel
        sec.extra_params["inner_voice_style"] = "block_chords"
        sec.extra_params["counterpoint"] = 0.2  # low = parallel motion
        sec.extra_params["voice_independence"] = 0.3
        sec.extra_params["step_leap_ratio"] = 0.85  # mostly stepwise = parallel
    return a


def backing_band(analysis):
    """
    Generate a full backing band: bass, drums, chords, and countermelody.

    Keeps the original melody and generates a complete rhythm section and
    harmony instruments around it.  Uses the source's harmonic progressions
    and energy profile to create a backing track that fits naturally.

    Ideal for creating karaoke-style accompaniment or adding a band to
    a solo melody.
    """
    a = deepcopy(analysis)
    for sec in a.sections:
        sec.generation_mode = "accompany"
        sec.extra_params["_accompany_roles"] = [
            "countermelody", "pad", "bass_fill", "drums",
        ]
        # Full band texture
        sec.extra_params["num_voices"] = 4
        sec.extra_params["bass_type"] = "walking"
        if not sec.texture.drum_style or sec.texture.drum_style == "none":
            sec.texture = TextureHints(
                drum_style="rock",
                drum_intensity=0.5,
                num_voices=4,
                bass_type="walking",
            )
    return a


def jazz_combo(analysis):
    """
    Jazz combo accompaniment: walking bass + jazz drums + comping chords.

    Preserves the original melody and layers a jazz rhythm section underneath.
    Walking bass, brush/swing drums, and block-chord comping create a classic
    jazz trio/quartet feel behind any melody.
    """
    a = deepcopy(analysis)
    for sec in a.sections:
        sec.generation_mode = "accompany"
        sec.extra_params["_accompany_roles"] = [
            "countermelody", "bass_fill", "drums",
        ]
        sec.extra_params["bass_type"] = "walking"
        sec.extra_params["swing"] = 0.6
        sec.extra_params["inner_voice_style"] = "block_chords"
        sec.extra_params["chord_complexity"] = 0.7
        sec.extra_params["harmonic_rhythm"] = 0.6
        sec.texture = TextureHints(
            drum_style="jazz",
            drum_intensity=0.4,
            swing=0.6,
            num_voices=3,
        )
    a = tempo_scale(a, 0.92)
    return a


def orchestral_accompany(analysis):
    """
    Orchestral accompaniment: strings pad, arpeggiated inner voices, bass.

    Wraps the original melody in a lush orchestral texture.  String pads
    provide sustained harmony, arpeggiated inner voices add movement,
    and a deep bass provides the foundation.

    The generated parts use string and brass instruments for a cinematic feel.
    """
    a = deepcopy(analysis)
    for sec in a.sections:
        sec.generation_mode = "accompany"
        sec.extra_params["_accompany_roles"] = [
            "countermelody", "pad", "bass_fill",
        ]
        sec.extra_params["instruments"] = ["strings"]
        sec.extra_params["inner_voice_style"] = "arpeggiated"
        sec.extra_params["num_voices"] = 4
        sec.extra_params["counterpoint"] = 0.5
        sec.extra_params["bass_type"] = "sustained"
        sec.energy = EnergyProfile(
            level=min(0.85, sec.energy.level * 1.2),
            arc=sec.energy.arc,
        )
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
    "grooved":          grooved,
    "inspired_by":         inspired_by,
    "inspired_sequel":     inspired_sequel,
    "inspired_contrast":   inspired_contrast,
    "accompany":           accompany,
    "duet":                duet,
    "call_response":       call_response,
    "harmonize":           harmonize,
    "backing_band":        backing_band,
    "jazz_combo":          jazz_combo,
    "orchestral_accompany": orchestral_accompany,
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
GROOVED           = grooved
INSPIRED_BY       = inspired_by
INSPIRED_SEQUEL   = inspired_sequel
INSPIRED_CONTRAST = inspired_contrast
ACCOMPANY              = accompany
DUET                   = duet
CALL_RESPONSE          = call_response
HARMONIZE              = harmonize
BACKING_BAND           = backing_band
JAZZ_COMBO             = jazz_combo
ORCHESTRAL_ACCOMPANY   = orchestral_accompany
