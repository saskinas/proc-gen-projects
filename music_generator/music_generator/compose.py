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


# ── Transition generation ─────────────────────────────────────────────────────

def generate_transition(
    spec: "TransitionSpec",
    prev_score,
    next_key: str,
    next_mode: str,
    time_signature: list,
    tempo_bpm: int,
    seed: int = 0,
):
    """
    Build a short MusicScore bridging two sections.

    spec          — TransitionSpec describing the type and duration
    prev_score    — MusicScore of the completed section (provides voice roles)
    next_key      — tonic of the incoming section (for pickup/link)
    next_mode     — mode of the incoming section

    Returns None for type "none".
    """
    from music_generator import (
        MusicScore, Track, Note, ROOT_TO_PC,
        scale_midi_in_range, chord_midi_in_range,
        midi_to_note, VOICE_RANGES,
    )
    import random

    if spec.type == "none":
        return None

    rng           = random.Random(seed)
    num, den      = int(time_signature[0]), int(time_signature[1])
    beats_per_bar = num * (4.0 / den)
    duration      = spec.duration_beats if spec.duration_beats > 0 else beats_per_bar

    roles     = prev_score.metadata.get("voices", [])
    non_drum  = [r for r in roles if r != "drums"]
    track_map = {
        r: prev_score.tracks[i]
        for i, r in enumerate(roles)
        if i < len(prev_score.tracks)
    }

    tonic_pc   = ROOT_TO_PC.get(next_key, 0)
    tracks:    list = []
    out_roles: list = []

    if spec.type == "pickup":
        for role in non_drum:
            instr      = track_map[role].instrument if role in track_map else "piano"
            v_lo, v_hi = VOICE_RANGES.get(role, (48, 84))
            if role == "soprano":
                center = (v_lo + v_hi) // 2 + 4
                scale  = sorted(scale_midi_in_range(tonic_pc, next_mode, center - 12, center + 1))
                n      = max(3, min(8, int(duration * 4)))
                run    = scale[-n:] if len(scale) >= n else scale
                sub    = duration / max(len(run), 1)
                notes  = [Note(midi_to_note(p), sub, min(112, 68 + i * 6))
                          for i, p in enumerate(run)]
            else:
                t     = track_map.get(role)
                last  = next((n for n in reversed(t.notes) if n.velocity > 0), None) if t else None
                pitch = last.pitch if last else midi_to_note(v_lo + 12)
                notes = [Note(pitch, duration, 52)]
            tracks.append(Track(instrument=instr, notes=notes))
            out_roles.append(role)

    elif spec.type == "link":
        dom_pc = (tonic_pc + 7) % 12
        for role in non_drum:
            instr      = track_map[role].instrument if role in track_map else "piano"
            v_lo, v_hi = VOICE_RANGES.get(role, (36, 84))
            dom_ns     = chord_midi_in_range(dom_pc, "dom7", v_lo, v_hi)
            if not dom_ns:
                dom_ns = chord_midi_in_range(dom_pc, "maj", v_lo, v_hi)
            pitch = midi_to_note(dom_ns[len(dom_ns) // 2] if dom_ns else v_lo + 7)
            tracks.append(Track(instrument=instr, notes=[Note(pitch, duration, 65)]))
            out_roles.append(role)

    elif spec.type == "fill":
        for role in non_drum:
            instr = track_map[role].instrument if role in track_map else "piano"
            tracks.append(Track(instrument=instr, notes=[Note("C4", duration, 0)]))
            out_roles.append(role)
        if "drums" in roles:
            fill_notes = []
            n_hits = max(4, int(duration * 4))
            sub    = duration / n_hits
            half   = n_hits // 2
            seq    = (["snare"] * half + ["tom_hi", "tom_hi_mid", "tom_lo_mid", "tom_floor"]
                      * ((n_hits - half + 3) // 4))[:n_hits]
            cursor = 0.0
            for k, voice in enumerate(seq):
                if k > 0:
                    fill_notes.append(Note("rest", sub, 0))
                    cursor += sub
                vel = min(127, int(65 + k * (55 / max(n_hits - 1, 1))))
                fill_notes.append(Note(voice, 0.0, vel))
            if cursor < duration - 0.001:
                fill_notes.append(Note("rest", duration - cursor, 0))
            tracks.append(Track(instrument="drums", notes=fill_notes))
            out_roles.append("drums")

    if not tracks:
        return None

    return MusicScore(
        tempo_bpm=tempo_bpm,
        time_signature=tuple(time_signature),
        tracks=tracks,
        metadata={"voices": out_roles, "form": "transition", "intent": {}},
    )


# ── Degree-encoded pitch helpers ──────────────────────────────────────────────

def _decode_note_event(ev: dict, tonic_pc: int, mode: str) -> str:
    """
    Convert a degree-encoded NoteEvent dict to a note-name string like "C#4".

    Handles three formats:
      New  — {"degree": int/str, "alter": int, "octave": int, …}
      Old  — {"pitch": "C#4", …}          (backward compat)
      Drum — {"pitch": "kick", …}         (handled by caller)

    The degree encoding is transform-safe:
    - After transpose(+n): tonic_pc shifts → decoded pitch shifts by n st.
    - After mode_swap(m):  SCALE_INTERVALS[m] changes → degree 3 becomes the
      third of the new mode (major third in major, minor third in minor), giving
      automatic parallel-mode melody substitution.
    """
    from music_generator import NOTE_NAMES, SCALE_INTERVALS

    # ── Backward compat: old absolute-pitch format ──
    if "pitch" in ev and "degree" not in ev:
        return ev["pitch"]      # note name string or "rest" — pass through

    degree = ev.get("degree", "rest")
    if degree == "rest":
        return "rest"

    octave = ev.get("octave", 4)
    alter  = ev.get("alter",  0)

    if degree == 0:
        # Chromatic escape: alter is raw semitones above tonic
        raw_pc = (tonic_pc + alter) % 12
    else:
        scale_ivs = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])
        try:
            diatonic_iv = scale_ivs[int(degree) - 1]
        except (IndexError, ValueError):
            diatonic_iv = 0
        raw_pc = (tonic_pc + diatonic_iv + alter) % 12

    return f"{NOTE_NAMES[raw_pc]}{octave}"


def _phrase_motif_from_sequence(
    soprano_seq: list[dict],
    tonic_pc:    int,
    mode:        str,
) -> dict | None:
    """
    Derive a MotifDef-compatible theme dict from a soprano voice sequence.

    The theme is built from the MIDI pitches decoded from the sequence in
    the given key/mode, so it always reflects the actual sounding melody
    (including any mode-swap or transposition that has already been applied
    to analysis.key / analysis.mode before this call).

    Returns None if fewer than 2 pitched events are present.
    """
    from music_generator import SCALE_INTERVALS

    scale_ivs = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])
    midi_pitches: list[int] = []
    durations:    list[float] = []

    for ev in soprano_seq:
        deg = ev.get("degree", "rest")
        if deg == "rest":
            continue
        octave = ev.get("octave", 4)
        alter  = ev.get("alter",  0)
        if deg == 0:
            raw_pc = (tonic_pc + alter) % 12
        else:
            try:
                iv = scale_ivs[int(deg) - 1]
            except (IndexError, ValueError):
                iv = 0
            raw_pc = (tonic_pc + iv + alter) % 12
        midi_pitches.append((octave + 1) * 12 + raw_pc)
        durations.append(float(ev.get("duration", 1.0)))

    if len(midi_pitches) < 2:
        return None

    intervals = [midi_pitches[i + 1] - midi_pitches[i] for i in range(len(midi_pitches) - 1)]
    return {
        "type":      "motif",
        "intervals": intervals,
        "durations": durations,
        "inverted":  [-x for x in intervals],
        "retrograde": list(reversed(intervals)),
    }


# ── Replay mode ───────────────────────────────────────────────────────────────

def replay_section(
    section:     "SectionSpec",
    analysis:    "MusicalAnalysis",
    base_params: dict,
):
    """
    Render captured voice_sequences as a MusicScore.

    Pitches are decoded using analysis.key and analysis.mode, so the output
    automatically adapts to any prior transpose() or mode_swap() transform —
    the stored degree-encoded sequences are symbolically correct.

    Returns None if the section has no voice_sequences.
    """
    from music_generator import MusicScore, Track, Note, ROOT_TO_PC

    seqs = section.voice_sequences
    if not seqs:
        return None

    # Use section-local key/mode when modulate_section() has set them
    sec_key  = section.extra_params.get("key",  analysis.key)
    sec_mode = section.extra_params.get("mode", analysis.mode)
    tonic_pc = ROOT_TO_PC.get(sec_key, ROOT_TO_PC.get(analysis.key, 0))

    instrs        = base_params.get("instruments") or ["square"]
    default_instr = instrs[0] if instrs else "square"

    # Look up original GM instruments from the analysis if available.
    # These are stored by ir_assembler from the source MIDI's program-change events.
    # GM programme number → export.py instrument name (covers common families).
    def _gm_to_instr(prog: int) -> str:
        """Map GM program number (0-127) to the nearest export.py instrument name."""
        if   prog <  8:  return "piano"
        elif prog < 16:  return "piano"         # chromatic perc → piano
        elif prog < 24:  return "organ"
        elif prog < 32:  return "guitar"
        elif prog < 40:  return "bass_guitar"
        elif prog < 56:  return "strings"
        elif prog < 64:  return "trumpet"
        elif prog < 80:  return "flute"
        elif prog < 96:  return "square"        # synth lead / NES-style
        elif prog < 112: return "strings"       # synth pad
        else:            return "piano"

    # Per-voice instrument: prefer original GM program if present, otherwise fall back.
    src_progs = section.extra_params.get("_source_programs", {})

    def _instr_for(voice: str) -> str:
        if voice == "drums":
            return "drums"
        if voice in src_progs:
            # Preserve the exact GM program number from the source MIDI so
            # timbre (e.g. Electric Piano 2 vs Acoustic Grand) is not altered.
            return f"gm_prog_{src_progs[voice]}"
        if voice == "bass":
            # Triangle channel carries the bass in NES music — use triangle
            # instrument so the bass sounds lower/softer than melody voices.
            return "triangle"
        return default_instr

    # Replay all voices in a stable order:
    #   soprano → inner voices → alto (+ alto_layer_N) → bass (+ bass_layer_N) → drums
    # Polyphonic-layer voices (alto_layer_2, bass_layer_2, inner_1_layer_2, …) are
    # generated by ir_assembler when the source channel contains chord content.
    # They carry the same instrument as their primary voice and are replayed here so
    # all original chord tones are preserved in the output.
    inner_keys = sorted(k for k in seqs if k.startswith("inner_"))
    alto_layer_keys = sorted(k for k in seqs if k.startswith("alto_layer_"))
    bass_layer_keys = sorted(k for k in seqs if k.startswith("bass_layer_"))
    voice_order = (
        [v for v in ["soprano"] if v in seqs]
        + inner_keys
        + ([v for v in ["alto"] if v in seqs] + alto_layer_keys)
        + ([v for v in ["bass"] if v in seqs] + bass_layer_keys)
        + [v for v in ["drums"] if v in seqs]
    )

    def _avg_note_dur(ev_list: list[dict]) -> float:
        """Mean duration of pitched (non-rest) events."""
        pitched = [ev for ev in ev_list if ev.get("degree") != "rest"]
        if not pitched:
            return 0.0
        return sum(ev.get("duration", 0.5) for ev in pitched) / len(pitched)

    tracks: list = []
    roles:  list = []

    for voice in voice_order:
        ev_list = seqs.get(voice)
        if not ev_list:
            continue
        is_drum = voice == "drums"
        # Skip arpeggio-style inner/alto voices (avg note duration < 0.20 beats).
        # NES arpeggio channels rapidly cycle pitches to fake chords; when
        # replayed as separate MIDI tracks they sound like noise rather than
        # harmony. Soprano and bass are always kept as the structural voices.
        if not is_drum and voice not in ("soprano", "bass") and not voice.startswith("bass_layer_"):
            if _avg_note_dur(ev_list) < 0.20:
                continue
        notes = []
        for ev in ev_list:
            if is_drum:
                pitch = ev.get("pitch", "rest")
                if pitch == "rest" or ev.get("degree") == "rest":
                    notes.append(Note("rest", ev["duration"], 0))
                else:
                    notes.append(Note(pitch, ev["duration"], ev.get("velocity", 80)))
            else:
                pitch_name = _decode_note_event(ev, tonic_pc, sec_mode)
                notes.append(Note(pitch_name, ev["duration"], ev.get("velocity", 80)))

        if not notes:
            continue
        tracks.append(Track(instrument=_instr_for(voice), notes=notes))
        roles.append(voice)

    if not tracks:
        return None

    return MusicScore(
        tempo_bpm      = analysis.tempo_bpm,
        time_signature = tuple(analysis.time_signature),
        tracks         = tracks,
        metadata       = {"voices": roles, "form": "replay", "intent": {}},
    )


# ── Arrange mode ─────────────────────────────────────────────────────────────

def _find_lead_voice(section: "SectionSpec"):
    """
    Return (role, seq) for the primary melodic voice when soprano is empty/sparse.

    Selection criteria (all must be met):
      - Not bass or drums
      - Average octave ≥ 3  (octave < 3 = bass/contra register, not melody)
      - Average pitched-note duration ≥ 0.20 beats  (shorter = arpeggio texture)
      - At least 4 pitched notes

    Among qualifying voices, the one with the SLOWEST average note duration is
    returned — the melody voice in game music plays longer notes than the fast
    arpeggio inner voices that have many more (shorter) events.

    Returns (None, []) when no qualifying voice is found, which causes
    arrange_section to fall back to full generation.
    """
    _RHYTHM   = {"bass", "drums"}
    _MIN_OCT  = 3     # below octave 3 = bass register
    _MIN_DUR  = 0.20  # avg note dur below this = arpeggio, not melody

    best_role  = None
    best_score = -1.0

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
        # avg_dur alone would pick a 4-note phrase over a 39-note one if slightly
        # slower; the sqrt(n) factor prevents that.
        score = avg_dur * (len(pitched) ** 0.5)
        if score > best_score:
            best_score = score
            best_role  = role

    return best_role, section.voice_sequences.get(best_role, [])


def arrange_section(
    section:     "SectionSpec",
    analysis:    "MusicalAnalysis",
    base_params: dict,
    seed:        int = 0,
):
    """
    Arrange mode: replay the soprano (primary melody) from the original
    voice sequences, regenerate ALL other voices in the target style.

    The result is "original Kirby melody + new accompaniment":
      soprano  — replayed from voice_sequences (key/mode adaptive)
      alto     — freshly generated in target style
      inner    — freshly generated
      bass     — freshly generated
      drums    — freshly generated

    Falls back to full generation for sections where the soprano has fewer
    than 4 pitched notes (e.g. intro/verse sections in Vegetable Valley
    where the melody simply doesn't play).

    Why only soprano?  Inner voices in SNES game music are typically fast
    arpeggio accompaniment patterns.  Replaying them in a jazz or baroque
    context clashes with the new style.  The soprano carries the recognisable
    melody; everything else should breathe in the new idiom.
    """
    from copy import deepcopy
    from music_generator import MusicScore

    _ensure_paths()

    # ── 1. Check soprano has meaningful content ────────────────────────────────
    sop_seq     = section.voice_sequences.get("soprano", [])
    pitched_sop = [ev for ev in sop_seq if ev.get("degree") != "rest"]

    if len(pitched_sop) < 4:
        # Soprano is silent/sparse — find the richest melodic voice instead.
        # In game music the 'melody' is not always in the highest-pitched channel
        # (e.g. Grape Garden intro and Vegetable Valley verse carry the melody in
        # the 'alto' slot because a different SNES channel carries it there).
        lead_role, lead_seq = _find_lead_voice(section)
        pitched_lead = [ev for ev in lead_seq if ev.get("degree") != "rest"]
        if len(pitched_lead) < 4:
            # No usable melody anywhere — full generation.
            accomp_sec = deepcopy(section)
            accomp_sec.voice_sequences = {}
            accomp_sec.generation_mode = "generate"
            return generate_section(accomp_sec, analysis, base_params, seed=seed)
        # Promote the lead voice to soprano so the merge logic below treats it
        # as the preserved melody (keeps it, discards generated soprano).
        sop_seq = lead_seq

    # ── 2. Build a replay score with ONLY soprano ──────────────────────────────
    sop_only_sec = deepcopy(section)
    sop_only_sec.voice_sequences = {"soprano": sop_seq}
    melody = replay_section(sop_only_sec, analysis, base_params)
    if melody is None:
        return generate_section(section, analysis, base_params, seed=seed)

    # ── 3. Generate a fresh full arrangement (all voices) ─────────────────────
    accomp_sec = deepcopy(section)
    accomp_sec.voice_sequences = {}
    accomp_sec.generation_mode = "generate"
    accomp = generate_section(accomp_sec, analysis, base_params, seed=seed)

    # ── 4. Merge: soprano from replay, everything else from generated ──────────
    # The generated section's soprano is discarded; all other generated voices
    # (alto/countermelody, bass, drums, inner voices) make up the new style.
    _MELODY_ROLES = {"soprano"}

    melody_voices = melody.metadata.get("voices", [])
    accomp_voices = accomp.metadata.get("voices", [t.instrument for t in accomp.tracks])

    melody_map = {
        role: melody.tracks[i]
        for i, role in enumerate(melody_voices)
        if i < len(melody.tracks)
    }
    accomp_map = {
        role: accomp.tracks[i]
        for i, role in enumerate(accomp_voices)
        if i < len(accomp.tracks)
    }

    merged_tracks = []
    merged_roles  = []

    # Primary melody from replay
    for role in melody_voices:
        if role in _MELODY_ROLES and role in melody_map:
            merged_tracks.append(melody_map[role])
            merged_roles.append(role)

    # Everything else (alto, bass, drums, …) from generated style
    for role in accomp_voices:
        if role not in _MELODY_ROLES and role in accomp_map:
            merged_tracks.append(accomp_map[role])
            merged_roles.append(role)

    if not merged_tracks:
        return melody

    return MusicScore(
        tempo_bpm      = melody.tempo_bpm,
        time_signature = melody.time_signature,
        tracks         = merged_tracks,
        metadata       = {"voices": merged_roles, "form": "arrange", "intent": {}},
    )


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

    # ── 8. Inject theme ──
    # Priority: explicit motif_id > phrase motif derived from voice_sequences > none
    motif_theme = analysis.get_motif_for_section(section)
    if motif_theme:
        d.override("theme", _make_theme_injector(motif_theme))
    elif section.voice_sequences:
        # Derive a phrase-level melodic theme from the lead melodic voice.
        # Uses _find_lead_voice which prefers soprano (≥ 4 notes) and falls
        # back to the most-melodic non-bass/drums voice when soprano is sparse.
        _, lead_seq = _find_lead_voice(section)
        if not lead_seq:
            lead_seq = section.voice_sequences.get("soprano", [])
        if lead_seq:
            from music_generator import ROOT_TO_PC
            sec_key  = section.extra_params.get("key",  analysis.key)
            sec_mode = section.extra_params.get("mode", analysis.mode)
            tpc      = ROOT_TO_PC.get(sec_key, ROOT_TO_PC.get(analysis.key, 0))
            phrase_theme = _phrase_motif_from_sequence(lead_seq, tpc, sec_mode)
            if phrase_theme:
                d.override("theme", _make_theme_injector(phrase_theme))

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
                # Emit MIDI program-change marker when instrument changes mid-piece
                if role in instruments and instruments[role] != track.instrument:
                    merged[role].append(Note(f"__pc__{track.instrument}", 0.0, 0))
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
    from music_generator.ir import TransitionSpec  # noqa: PLC0415

    scores = []
    for i, section in enumerate(analysis.sections):
        section_seed = seed + i * 1000
        # Route each section through replay or the generative pipeline.
        #
        # generation_mode == "auto":
        #   Replay voice_sequences when present (degree encoding auto-adapts to
        #   transpose/mode_swap). Fall through to generate_section when absent.
        # generation_mode == "replay":
        #   Always replay.  Use when you want exact notes with harmonic-only transforms.
        # generation_mode == "generate":
        #   Always run the generative pipeline.  Set automatically by style/texture/
        #   energy transforms (apply_style_preset, change_texture, etc.) so that
        #   those changes are actually heard rather than overridden by stored notes.
        #   voice_sequences remain available as a phrase-motif template for the generator.
        mode  = section.generation_mode
        score = None
        if mode == "arrange" and section.voice_sequences:
            score = arrange_section(section, analysis, base_params, seed=section_seed)
        elif mode != "generate" and section.voice_sequences:
            score = replay_section(section, analysis, base_params)
        if score is None:
            score = generate_section(section, analysis, base_params, seed=section_seed)
        scores.append(score)

        # Generate transition material after this section (not after the last)
        if i < len(analysis.sections) - 1:
            next_sec  = analysis.sections[i + 1]
            trans_key  = next_sec.extra_params.get("key",  analysis.key)
            trans_mode = next_sec.extra_params.get("mode", analysis.mode)
            trans = generate_transition(
                spec=section.transition,
                prev_score=score,
                next_key=trans_key,
                next_mode=trans_mode,
                time_signature=list(analysis.time_signature),
                tempo_bpm=analysis.tempo_bpm,
                seed=section_seed + 500,
            )
            if trans is not None:
                scores.append(trans)

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
