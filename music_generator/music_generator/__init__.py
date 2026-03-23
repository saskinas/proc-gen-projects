"""
MusicDomain — procedural music composition.

Uses music theory (scales, diatonic harmony, voice leading, counterpoint,
rhythmic elaboration) to generate structured multi-voice MIDI-ready scores.

The parameter space is intentionally rich: by tuning the knobs below, a
designer can reproduce many historical styles without hard-coded presets.

Example — Bach-style two-voice invention in D minor::

    designer = Designer(domain=MusicDomain())
    designer.set("key", "D")
    designer.set("mode", "minor")
    designer.set("form", "binary")
    designer.set("num_voices", 2)
    designer.set("counterpoint", 0.85)
    designer.set("imitation", 0.6)
    designer.set("inner_voice_style", "running_sixteenths")
    designer.set("bass_type", "walking")
    designer.set("circle_of_fifths", 0.8)
    designer.set("chord_complexity", 0.5)
    designer.set("rhythmic_density", 0.8)
    designer.set("use_ornaments", True)
    designer.set("instruments", ["harpsichord"])
    designer.set("tempo_bpm", 72)
    score = designer.generate()
    midi_bytes = to_midi(score)

Designer knobs
--------------
Structure
    form                str   "binary" | "ternary" | "rondo" | "through_composed"
    num_phrases         int   2–8      number of phrases in the piece
    phrase_length       int   2–16     bars per phrase

Pitch / Harmony
    key                 str   "C" | "D" | "Eb" | "F#" … any enharmonic root
    mode                str   "major" | "minor" | "harmonic_minor" | "dorian"
                              | "phrygian" | "lydian" | "mixolydian"
    chord_complexity    float 0.0 triads only → 1.0 7ths, dim7, secondary dominants
    harmonic_rhythm     float 0.0 one chord per phrase → 1.0 one chord per beat
    circle_of_fifths    float 0.0 free motion → 1.0 strong V–I dominant resolution
    modal_mixture       float 0.0 purely diatonic → 1.0 chromatic borrowing
    pedal_point         float 0.0–1.0  probability of inserting a pedal section

Texture / Voices
    num_voices          int   1–4
    texture             str   "monophonic" | "homophonic" | "polyphonic"
    counterpoint        float 0.0 block chords → 1.0 fully independent voices
    imitation           float 0.0 no imitation → 1.0 strict canon / fugue-like
    voice_independence  float 0.0 parallel motion → 1.0 contrary motion

Bass
    bass_type           str   "sustained" | "walking" | "alberti"
                              | "arpeggiated" | "pedal"

Inner voices
    inner_voice_style   str   "block_chords" | "arpeggiated"
                              | "countermelody" | "running_sixteenths"

Melody
    melodic_range       int   1–3   octave span the melody may traverse
    step_leap_ratio     float 0.0 all leaps → 1.0 all steps
    sequence_probability float 0.0–1.0  likelihood of melodic sequences
    use_ornaments       bool  add trills and mordents
    melodic_direction   str   "ascending" | "descending" | "arch" | "free"

Rhythm
    tempo_bpm           int   40–240
    time_signature      list  [4,4] | [3,4] | [6,8] | [2,4] | [2,2]
    rhythmic_density    float 0.0 whole notes → 1.0 thirty-second notes
    rhythmic_regularity float 0.0 syncopated → 1.0 metronomic
    swing               float 0.0 straight → 1.0 full swing feel

Instrumentation
    instruments         list  up to 4; any of: "piano", "harpsichord", "organ",
                              "strings", "violin", "viola", "cello", "contrabass",
                              "flute", "oboe", "clarinet", "bassoon", "horn",
                              "trumpet", "bass", "guitar"

Themes
    melodic_theme       str   "none" | "motif" | "fugue_subject"
                              recurring melodic idea applied at phrase starts
                              with cycled transformations (original, inverted,
                              retrograde, retrograde-inversion)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from procgen.domains.base import DomainBase
from procgen.core import Intent, GenerationPlan
from procgen.generators import GeneratorBase


# ── Music Theory Constants ─────────────────────────────────────────────────────

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

ROOT_TO_PC: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11,
}

SCALE_INTERVALS: dict[str, list[int]] = {
    "major":             [0, 2, 4, 5, 7, 9, 11],
    "minor":             [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor":    [0, 2, 3, 5, 7, 8, 11],
    "dorian":            [0, 2, 3, 5, 7, 9, 10],
    "phrygian":          [0, 1, 3, 5, 7, 8, 10],
    "lydian":            [0, 2, 4, 6, 7, 9, 11],
    "mixolydian":        [0, 2, 4, 5, 7, 9, 10],
    # Rock / blues scales — padded to 7 entries for degree compatibility
    "pentatonic_major":  [0, 2, 4, 7, 9, 9, 9],
    "pentatonic_minor":  [0, 3, 5, 7, 10, 10, 10],
    "blues":             [0, 3, 5, 6, 7, 10, 10],
}

# Diatonic triad quality per scale degree (index = degree-1)
DIATONIC_TRIADS: dict[str, list[str]] = {
    "major":             ["maj", "min", "min", "maj", "maj", "min", "dim"],
    "minor":             ["min", "dim", "maj", "min", "min", "maj", "maj"],
    "harmonic_minor":    ["min", "dim", "aug", "min", "maj", "maj", "dim"],
    "dorian":            ["min", "min", "maj", "maj", "min", "dim", "maj"],
    "phrygian":          ["min", "maj", "maj", "min", "dim", "maj", "min"],
    "lydian":            ["maj", "maj", "min", "dim", "maj", "min", "min"],
    "mixolydian":        ["maj", "min", "dim", "maj", "min", "min", "maj"],
    # Rock / blues — degrees 6-7 are duplicates (max_deg=5 in harmonic planner)
    "pentatonic_major":  ["maj", "min", "min", "maj", "min", "min", "min"],
    "pentatonic_minor":  ["min", "maj", "min", "min", "maj", "maj", "maj"],
    "blues":             ["dom7", "dom7", "min", "maj", "dom7", "min", "min"],
}

CHORD_INTERVALS: dict[str, list[int]] = {
    "maj":     [0, 4, 7],
    "min":     [0, 3, 7],
    "dim":     [0, 3, 6],
    "aug":     [0, 4, 8],
    "dom7":    [0, 4, 7, 10],
    "maj7":    [0, 4, 7, 11],
    "min7":    [0, 3, 7, 10],
    "dim7":    [0, 3, 6, 9],
    "hdim7":   [0, 3, 6, 10],
    "minmaj7": [0, 3, 7, 11],
    "sus4":    [0, 5, 7],
    "power":   [0, 7],         # root + fifth only (rock power chord)
    "sus2":    [0, 2, 7],      # suspended 2nd
}

# Modes that use only 5 scale degrees (harmonic planner limits to degrees 1-5)
_PENTATONIC_MODES = frozenset({"pentatonic_major", "pentatonic_minor", "blues"})

# MIDI voice ranges (inclusive)
VOICE_RANGES: dict[str, tuple[int, int]] = {
    "soprano": (60, 81),   # C4–A5
    "alto":    (53, 72),   # F3–C5
    "tenor":   (48, 69),   # C3–A4
    "bass":    (36, 60),   # C2–C4
}

VOICE_ROLES: dict[int, list[str]] = {
    1: ["soprano"],
    2: ["soprano", "bass"],
    3: ["soprano", "alto", "bass"],
    4: ["soprano", "alto", "tenor", "bass"],
}


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class HarmonicEvent:
    root_pc: int         # pitch class 0-11
    chord_type: str      # "maj", "min", "dom7", etc.
    duration_beats: float
    degree: int          # scale degree 1-7
    phrase_idx: int = 0  # which phrase (0-based) this event belongs to


@dataclass
class Note:
    pitch: str           # e.g. "C4", "F#3"
    duration: float      # in beats
    velocity: int        # 0-127


@dataclass
class Track:
    instrument: str
    notes: list[Note]


@dataclass
class MusicScore:
    tempo_bpm: int
    time_signature: tuple[int, int]
    tracks: list[Track]
    metadata: dict = field(default_factory=dict)

    def describe(self) -> str:
        intent = self.metadata.get("intent", {})
        lines = [
            "MusicScore",
            f"  Tempo      : {self.tempo_bpm} BPM",
            f"  Time sig   : {self.time_signature[0]}/{self.time_signature[1]}",
            f"  Voices     : {len(self.tracks)}",
            f"  Total notes: {sum(len(t.notes) for t in self.tracks)}",
        ]
        if intent:
            lines.append(
                f"  Key/Mode   : {intent.get('key', '?')} {intent.get('mode', '?')}"
            )
            lines.append(
                f"  Form       : {intent.get('form', '?')}  "
                f"{intent.get('num_phrases', '?')} phrases × "
                f"{intent.get('phrase_length', '?')} bars"
            )
        return "\n".join(lines)


# ── Pitch Helpers ──────────────────────────────────────────────────────────────

def midi_to_note(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def note_str_to_midi(s: str) -> int:
    """Convert note string like 'C4' or 'F#3' or 'Bb2' to MIDI number."""
    # find where the octave number starts (last '-' or digit run)
    i = len(s)
    while i > 1 and (s[i - 1].isdigit() or (i == len(s) and s[i - 1] == '-')):
        i -= 1
    name = s[:i]
    octave = int(s[i:]) if s[i:] else 4
    pc = ROOT_TO_PC.get(name, 0)
    return (octave + 1) * 12 + pc


def scale_midi_in_range(root_pc: int, mode: str, lo: int, hi: int) -> list[int]:
    ivs = set(SCALE_INTERVALS[mode])
    return [m for m in range(lo, hi + 1) if (m % 12 - root_pc) % 12 in ivs]


def chord_midi_in_range(root_pc: int, chord_type: str, lo: int, hi: int) -> list[int]:
    ivs = set(CHORD_INTERVALS.get(chord_type, [0, 4, 7]))
    return [m for m in range(lo, hi + 1) if (m % 12 - root_pc) % 12 in ivs]


def degree_root_pc(tonic_pc: int, mode: str, degree: int) -> int:
    """Pitch class of scale degree `degree` (1-based)."""
    return (tonic_pc + SCALE_INTERVALS[mode][degree - 1]) % 12


def nearest_pitch(target: int, candidates: list[int]) -> int:
    if not candidates:
        return target
    return min(candidates, key=lambda c: abs(c - target))


def closest_above(midi: int, candidates: list[int]) -> int:
    above = [c for c in candidates if c > midi]
    return min(above) if above else (max(candidates) if candidates else midi + 2)


def closest_below(midi: int, candidates: list[int]) -> int:
    below = [c for c in candidates if c < midi]
    return max(below) if below else (min(candidates) if candidates else midi - 2)


# ── Generator 1: Harmonic Plan ─────────────────────────────────────────────────

class HarmonicPlanGenerator(GeneratorBase):
    """
    Generates the harmonic skeleton as a flat sequence of HarmonicEvent objects.

    Chord selection is driven by:
    - circle_of_fifths  (how strongly dominant → tonic motion is preferred)
    - harmonic_rhythm   (how many chord changes per phrase)
    - chord_complexity  (whether to extend triads with 7ths / altered chords)
    - modal_mixture     (borrowing from parallel major/minor)
    - pedal_point       (chance of a sustained-bass pedal section)
    - form              (binary midpoint ends on V; final phrase ends on I)
    """

    def generate(self, params: dict, context: dict) -> list[HarmonicEvent]:
        rng = self.rng()
        tonic_pc   = ROOT_TO_PC[params["key"]]
        mode       = params["mode"]
        num_phrases = params["num_phrases"]
        phrase_bars = params["phrase_length"]
        beats_bar   = params["time_signature"][0]
        phrase_beats = phrase_bars * beats_bar
        cof          = params["circle_of_fifths"]
        complexity   = params["chord_complexity"]
        mixture      = params["modal_mixture"]
        pedal_prob   = params["pedal_point"]
        form         = params["form"]
        harm_rhythm  = params["harmonic_rhythm"]
        power_chords = params.get("power_chords", False)
        max_deg      = 5 if mode in _PENTATONIC_MODES else 7

        # chords per phrase: 0.0→1, 0.5→phrase_bars, 1.0→2*phrase_bars
        raw = phrase_bars * (2 ** (harm_rhythm * 2 - 1))
        chords_pp = max(1, min(int(round(raw)), phrase_beats))
        beats_per_chord = phrase_beats / chords_pp

        events: list[HarmonicEvent] = []
        half = num_phrases // 2

        for p in range(num_phrases):
            is_last     = (p == num_phrases - 1)
            is_midpoint = form in ("binary", "ternary") and (p == half - 1)

            # ── Final chord of phrase (cadence target) ──
            if is_last:
                cadence_end  = 1   # perfect authentic: V → I
                cadence_prev = 5
            elif is_midpoint:
                cadence_end  = 5   # half cadence: end on V
                cadence_prev = 2 if rng.random() < 0.55 else 4
            else:
                cadence_end  = (1 if (mode in _PENTATONIC_MODES or rng.random() < 0.65) else 6)
                cadence_prev = 5

            # ── Build degree sequence for this phrase ──
            degree_seq: list[int] = []
            current = 1
            for i in range(chords_pp):
                remaining = chords_pp - i
                if remaining == 1:
                    degree_seq.append(cadence_end)
                elif remaining == 2:
                    degree_seq.append(cadence_prev)
                elif remaining == 3 and cadence_prev == 5:
                    # ii or IV before V before I
                    degree_seq.append(2 if rng.random() < 0.5 else 4)
                else:
                    degree_seq.append(
                        self._weighted_next(rng, current, cof, max_deg)
                    )
                current = degree_seq[-1]

            # ── Optional pedal-point section (interior only) ──
            if rng.random() < pedal_prob and chords_pp >= 3 and not is_last:
                pedal_deg = 5 if mode in ("major", "lydian", "mixolydian") else 1
                mid = chords_pp // 2
                degree_seq[1:mid] = [pedal_deg] * (mid - 1)

            # ── Convert to HarmonicEvents ──
            for deg in degree_seq:
                root  = degree_root_pc(tonic_pc, mode, deg)
                ctype = DIATONIC_TRIADS[mode][deg - 1]

                # Modal mixture: borrow from parallel mode
                if mixture > 0 and rng.random() < mixture * 0.35:
                    alt = "minor" if mode in ("major", "lydian", "mixolydian") else "major"
                    ctype = DIATONIC_TRIADS[alt][deg - 1]

                # Seventh chord extension
                if rng.random() < complexity * 0.75:
                    ctype = self._add_seventh(ctype, rng, complexity)

                # Secondary dominant: V/V (applied dominant)
                if (complexity > 0.45
                        and deg in (2, 3, 4, 5, 6)
                        and rng.random() < complexity * 0.25):
                    sec_root = (root + 7) % 12
                    events.append(HarmonicEvent(
                        root_pc=sec_root,
                        chord_type="dom7" if complexity > 0.6 else "maj",
                        duration_beats=beats_per_chord * 0.5,
                        degree=deg,
                        phrase_idx=p,
                    ))
                    events.append(HarmonicEvent(
                        root_pc=root,
                        chord_type=ctype,
                        duration_beats=beats_per_chord * 0.5,
                        degree=deg,
                        phrase_idx=p,
                    ))
                    continue

                # Dominant always prefers dom7 when cof is strong
                if deg == 5 and ctype in ("maj", "maj7") and rng.random() < cof * 0.8:
                    ctype = "dom7"

                # Power chords override: strip to root + fifth only
                if power_chords:
                    ctype = "power"

                events.append(HarmonicEvent(root, ctype, beats_per_chord, deg, p))

        return events

    @staticmethod
    def _weighted_next(rng, current: int, cof: float, max_degree: int = 7) -> int:
        w = {d: 1.0 for d in range(1, max_degree + 1)}
        w[current] = 0.25   # penalise immediate repetition

        # V → I is the strongest resolution
        if current == 5 and 1 in w:
            w[1] *= 1 + 6 * cof
        # vii° → I
        if current == 7 and 1 in w:
            w[1] *= 1 + 4 * cof
        # Predominant → V
        if current in (2, 4) and 5 in w:
            w[5] *= 1 + 3 * cof
        # Circle-of-fifths descending root motion
        down5 = ((current - 2) % max_degree) + 1
        if down5 in w:
            w[down5] *= 1 + 2 * cof
        # Deceptive option (V → vi)
        if current == 5 and 6 in w:
            w[6] *= 1 + (1 - cof) * 1.5

        total = sum(w.values())
        r = rng.random() * total
        cum = 0.0
        for d in sorted(w):
            cum += w[d]
            if r <= cum:
                return d
        return 1

    @staticmethod
    def _add_seventh(ctype: str, rng, complexity: float) -> str:
        if ctype == "dim" and rng.random() < complexity * 0.5:
            return "dim7"
        return {"maj": "maj7", "min": "min7", "dim": "hdim7", "aug": "aug"}.get(ctype, ctype)


# ── Generator 2: Voice Leading ─────────────────────────────────────────────────

class VoiceLeadingGenerator(GeneratorBase):
    """
    Assigns a MIDI pitch to each voice for every HarmonicEvent.

    Returns dict[voice_name → list[(midi_note, duration_beats)]].

    Voice-leading principles applied:
    - Prefer minimal motion from one chord to the next (smooth voice leading)
    - Higher voice_independence → favour contrary motion against soprano
    - No voice crossing (soprano > alto > tenor > bass enforced)
    - Imitation: alto (and tenor if present) echoes soprano line delayed by
      offset_events events, transposed down a fifth
    """

    def generate(self, params: dict, context: dict) -> dict[str, list[tuple[int, float]]]:
        rng = self.rng()
        events: list[HarmonicEvent] = context["harmonic_plan"]
        num_voices   = params["num_voices"]
        voice_indep  = params["voice_independence"]
        imitation    = params["imitation"]
        tonic_pc     = ROOT_TO_PC[params["key"]]
        mode         = params["mode"]

        roles = VOICE_ROLES[num_voices]
        lines: dict[str, list[tuple[int, float]]] = {r: [] for r in roles}

        # Leading-tone semitones above tonic per mode
        _lt_semi = {
            "major": 11, "minor": 10, "harmonic_minor": 11,
            "dorian": 10, "phrygian": 10, "lydian": 11, "mixolydian": 10,
        }
        leading_tone_pc = (tonic_pc + _lt_semi.get(mode, 11)) % 12
        prev_ev = None

        # Sensible starting pitches
        cur: dict[str, int] = {
            "soprano": 67,  # G4
            "alto":    62,  # D4
            "tenor":   55,  # G3
            "bass":    48,  # C3
        }

        for ev in events:
            chord_tones: dict[str, int] = {}

            for voice in roles:
                lo, hi = VOICE_RANGES[voice]
                candidates = chord_midi_in_range(ev.root_pc, ev.chord_type, lo, hi)
                if not candidates:
                    # Fall back: root in octave inside range
                    r = lo + (ev.root_pc - lo % 12) % 12
                    candidates = [r + 12 * i for i in range(3) if lo <= r + 12 * i <= hi] or [lo]

                prev = cur[voice]
                best = nearest_pitch(prev, candidates)

                # Tendency tone: resolve leading tone up, chordal 7th down (V→I)
                if prev_ev is not None and ev.degree == 1 and prev_ev.degree in (5, 7):
                    pc = prev % 12
                    if pc == leading_tone_pc:
                        # Leading tone → tonic (step up)
                        tonic_c = [c for c in candidates if c % 12 == tonic_pc]
                        if tonic_c:
                            best = min(tonic_c, key=lambda c: abs(c - (prev + 1)))
                    elif (prev_ev.chord_type in ("dom7",) and
                          pc == (prev_ev.root_pc + 10) % 12):
                        # Chordal 7th → 3rd or root of tonic (step down)
                        target_pcs = {tonic_pc, (tonic_pc + 4) % 12}
                        down_c = [c for c in candidates
                                  if c % 12 in target_pcs and c <= prev]
                        if down_c:
                            best = max(down_c)

                # Contrary motion with soprano for inner / bass voices
                if voice != "soprano" and rng.random() < voice_indep * 0.45:
                    sop = cur.get("soprano", 67)
                    # Prefer moving opposite to soprano's last direction
                    if lines.get("soprano"):
                        sop_prev = lines["soprano"][-1][0]
                        going_up = (sop > sop_prev)
                        pool = (
                            [c for c in candidates if c < prev]
                            if going_up else
                            [c for c in candidates if c > prev]
                        )
                        if pool:
                            best = nearest_pitch(prev, pool)

                chord_tones[voice] = best
                cur[voice] = best

            # Enforce strict voice ordering (no crossing)
            order = [v for v in ["soprano", "alto", "tenor", "bass"] if v in chord_tones]
            for i in range(len(order) - 1):
                v_hi, v_lo = order[i], order[i + 1]
                if chord_tones[v_hi] < chord_tones[v_lo]:
                    chord_tones[v_hi], chord_tones[v_lo] = chord_tones[v_lo], chord_tones[v_hi]

            for voice in roles:
                lines[voice].append((chord_tones[voice], ev.duration_beats))

            prev_ev = ev

        # ── Imitation (canon-style entry) ──
        if imitation > 0 and num_voices >= 2:
            sop = lines["soprano"]
            n   = len(sop)
            offset = max(1, int(imitation * 4))

            for imit_voice in ["alto", "tenor"]:
                if imit_voice not in roles:
                    continue
                if rng.random() > imitation:
                    continue

                lo, hi = VOICE_RANGES[imit_voice]
                cover_len = int(n * imitation * 0.6)
                for i in range(offset, min(n, offset + cover_len)):
                    src_midi = sop[i - offset][0] - 7  # down a fifth
                    # clamp to voice range
                    while src_midi < lo:
                        src_midi += 12
                    while src_midi > hi:
                        src_midi -= 12
                    lines[imit_voice][i] = (src_midi, lines[imit_voice][i][1])

        return lines


# ── Shared utility ─────────────────────────────────────────────────────────────

def _subdiv(density: float, dur: float) -> float:
    """Map rhythmic density [0,1] to a note subdivision duration in beats."""
    if density < 0.15:
        return dur
    if density < 0.30:
        return max(2.0, dur)
    if density < 0.50:
        return 1.0
    if density < 0.65:
        return 0.5
    if density < 0.82:
        return 0.25
    return 0.125


# ── Generator 2b: Theme ────────────────────────────────────────────────────────

class ThemeGenerator(GeneratorBase):
    """
    Generates a short melodic motif that recurs across phrases.

    Produces semitone-interval sequences (not absolute pitches) so the theme
    can be applied at any transposition.  Stores four transformations:
    original, inverted, retrograde, and retrograde-inversion.

    When melodic_theme == "none", returns a no-op dict so downstream
    generators can always call context["theme"] safely.
    """

    def generate(self, params: dict, context: dict) -> dict:
        theme_type = params.get("melodic_theme", "none")
        if theme_type == "none":
            return {"type": "none", "intervals": [], "durations": []}

        rng      = self.rng()
        density  = params["rhythmic_density"]
        step_leap = params["step_leap_ratio"]

        # Length: motif = 4-5 notes, fugue_subject = 6-8 notes
        n = rng.randint(4, 5) if theme_type == "motif" else rng.randint(6, 8)

        # Build interval sequence (semitones)
        intervals: list[int] = []
        for _ in range(n - 1):
            if rng.random() < step_leap:
                intervals.append(rng.choice([-2, -1, 1, 2]))
            else:
                intervals.append(rng.choice([-7, -5, -4, -3, 3, 4, 5, 7]))

        # Durations based on density
        base = _subdiv(density, 1.0)
        durations: list[float] = [base] * n

        # Fugue subjects get a characteristic long-short-short rhythm
        if theme_type == "fugue_subject" and density > 0.3:
            durations = [base * 2 if i % 3 == 0 else base for i in range(n)]

        return {
            "type":       theme_type,
            "intervals":  intervals,
            "durations":  durations,
            "inverted":   [-iv for iv in intervals],
            "retrograde": list(reversed(intervals)),
        }


# ── Generator 3: Rhythmic Elaboration ──────────────────────────────────────────

class ElaborationGenerator(GeneratorBase):
    """
    Subdivides each voice's harmonic-rhythm note into a melodic/rhythmic figure.

    soprano   → melodic elaboration (steps, leaps, ornaments, sequences)
    bass      → bass pattern (walking, alberti, arpeggiated, pedal, sustained)
    inner     → inner voice style (block, arpeggiated, countermelody, running 16ths)
    """

    def generate(self, params: dict, context: dict) -> dict[str, list[Note]]:
        rng = self.rng()
        voice_lines: dict[str, list[tuple[int, float]]] = context["voice_lines"]
        harm_events: list[HarmonicEvent] = context["harmonic_plan"]
        tonic_pc    = ROOT_TO_PC[params["key"]]
        mode        = params["mode"]
        density     = params["rhythmic_density"]
        bass_type   = params["bass_type"]
        inner_style = params["inner_voice_style"]
        use_orn     = params["use_ornaments"]
        step_leap   = params["step_leap_ratio"]
        seq_prob    = params["sequence_probability"]
        mel_dir     = params["melodic_direction"]
        regularity  = params["rhythmic_regularity"]
        swing       = params["swing"]
        theme       = context.get("theme", {"type": "none", "intervals": [], "durations": []})
        tonic_pc    = ROOT_TO_PC[params["key"]]
        mode_str    = params["mode"]
        ending      = params.get("ending", "held")
        intro       = params.get("intro",  "none")
        total_events = len(harm_events)

        # Identify which event indices are phrase starts; compute phrase lengths
        phrase_start_evs: set[int] = set()
        seen_phrases: set[int] = set()
        phrase_lengths: dict[int, int] = {}
        phrase_start_idx: dict[int, int] = {}
        for i, ev in enumerate(harm_events):
            phrase_lengths[ev.phrase_idx] = phrase_lengths.get(ev.phrase_idx, 0) + 1
            if ev.phrase_idx not in seen_phrases:
                phrase_start_evs.add(i)
                phrase_start_idx[ev.phrase_idx] = i
                seen_phrases.add(ev.phrase_idx)

        total_phrases = len(seen_phrases)
        theme_dev     = params.get("melodic_theme_dev", "flat")

        def _vel_mod(ev_idx: int) -> int:
            """Arch-shaped velocity modifier peaking at 60% through each phrase."""
            if ev_idx >= len(harm_events):
                return 0
            ev      = harm_events[ev_idx]
            p_len   = phrase_lengths.get(ev.phrase_idx, 1)
            p_start = phrase_start_idx.get(ev.phrase_idx, 0)
            t = (ev_idx - p_start) / max(p_len - 1, 1)
            # Rise from 0→+8 over first 60%, fall from +8→-8 over last 40%
            if t <= 0.6:
                mod = int(t / 0.6 * 8)
            else:
                mod = int(8 - (t - 0.6) / 0.4 * 16)
            return max(-10, min(10, mod))

        result: dict[str, list[Note]] = {}

        for voice, line in voice_lines.items():
            notes: list[Note] = []
            is_bass    = (voice == "bass")
            is_soprano = (voice == "soprano")
            last_sop_midi: list[int] = []   # for melodic sequence detection

            for ev_idx, (midi, dur) in enumerate(line):
                ev = harm_events[ev_idx] if ev_idx < len(harm_events) else None
                root_pc    = ev.root_pc    if ev else tonic_pc
                chord_type = ev.chord_type if ev else "maj"
                v_lo, v_hi = VOICE_RANGES.get(voice, (36, 84))
                lo_search  = max(midi - 14, v_lo - 1)   # stay near voice range
                hi_search  = min(midi + 14, v_hi + 1)
                chord_ns   = chord_midi_in_range(root_pc, chord_type, lo_search, hi_search)
                scale_ns   = scale_midi_in_range(tonic_pc, mode,      lo_search, hi_search)

                base_vel = 90 if is_soprano else (78 if is_bass else 66)
                vel = max(42, min(110, base_vel + _vel_mod(ev_idx)))

                # ── Ritardando: slow the penultimate event ──
                is_penultimate = (ending == "ritardando" and
                                  ev_idx == total_events - 2)
                ev_density = max(0.0, density - 0.35) if is_penultimate else density

                # ── Held ending: sustain final chord, soprano snaps to tonic ──
                if ending != "none" and ev_idx == total_events - 1:
                    if is_soprano:
                        tonic_opts = [c for c in chord_ns if c % 12 == tonic_pc]
                        hold_midi  = nearest_pitch(midi, tonic_opts) if tonic_opts else midi
                    else:
                        hold_midi = midi
                    notes.append(Note(midi_to_note(hold_midi), dur, min(110, vel + 4)))
                    if is_soprano:
                        last_sop_midi = [note_str_to_midi(notes[-1].pitch)]
                    continue

                # ── Upbeat intro: ascending pickup figure in soprano on first event ──
                if intro == "upbeat" and ev_idx == 0 and is_soprano:
                    pickup_sub = _subdiv(density, 1.0)
                    n_pickup   = min(3, max(1, int(dur * 0.4 / max(pickup_sub, 0.125))))
                    run_lo     = max(midi - n_pickup * 2 - 2, v_lo)
                    run_scale  = scale_midi_in_range(tonic_pc, mode_str, run_lo, midi)
                    run_pts    = sorted(run_scale)[-n_pickup:]
                    if not run_pts:
                        run_pts = [max(v_lo, midi - 2)]
                    pickup_notes = [
                        Note(midi_to_note(p), pickup_sub, max(45, vel - 8 + i * 3))
                        for i, p in enumerate(run_pts)
                    ]
                    rem = dur - sum(n.duration for n in pickup_notes)
                    if rem >= pickup_sub * 0.5:
                        pickup_notes.append(Note(midi_to_note(midi), rem, vel))
                    notes.extend(pickup_notes)
                    last_sop_midi = [note_str_to_midi(n.pitch) for n in pickup_notes[:6]]
                    continue

                if is_soprano:
                    phrase_num = harm_events[ev_idx].phrase_idx if ev_idx < len(harm_events) else 0
                    p_len   = phrase_lengths.get(phrase_num, 1)
                    p_start = phrase_start_idx.get(phrase_num, 0)
                    phrase_progress = (ev_idx - p_start) / max(p_len - 1, 1)
                    sub_notes = self._soprano(
                        rng, midi, dur, ev_density, chord_ns, scale_ns,
                        step_leap, seq_prob, mel_dir, use_orn, vel, last_sop_midi,
                        theme=theme, tonic_pc=tonic_pc, mode_str=mode_str,
                        is_phrase_start=(ev_idx in phrase_start_evs),
                        phrase_num=phrase_num,
                        phrase_progress=phrase_progress,
                        regularity=regularity,
                        total_phrases=total_phrases,
                        theme_dev=theme_dev,
                    )
                    notes.extend(sub_notes)
                    last_sop_midi = [note_str_to_midi(n.pitch) for n in sub_notes[:6]]

                elif is_bass:
                    sub_notes = self._bass(
                        rng, midi, dur, bass_type, chord_ns, scale_ns, ev_density, vel
                    )
                    notes.extend(sub_notes)

                else:
                    sub_notes = self._inner(
                        rng, midi, dur, inner_style, chord_ns, scale_ns,
                        ev_density, vel, voice
                    )
                    notes.extend(sub_notes)

            # Apply swing to all notes in this voice
            if swing > 0.0:
                notes = self._apply_swing(notes, swing, params["time_signature"][0])

            result[voice] = notes

        return result

    # ── Soprano melody ───────────────────────────────────────────────────────

    def _soprano(self, rng, midi, dur, density, chord_ns, scale_ns,
                 step_leap, seq_prob, mel_dir, use_orn, vel, last_midi: list[int],
                 theme=None, tonic_pc: int = 0, mode_str: str = "major",
                 is_phrase_start: bool = False, phrase_num: int = 0,
                 phrase_progress: float = 0.5, regularity: float = 0.6,
                 total_phrases: int = 1, theme_dev: str = "flat") -> list[Note]:

        # ── Apply melodic theme at phrase starts ──
        if theme and theme["type"] != "none" and is_phrase_start:
            # Development stage: 0.0 = first phrase, 1.0 = last phrase
            stage = phrase_num / max(total_phrases - 1, 1) if total_phrases > 1 else 0.5

            # Select transformation variant (cycles through 4 classic forms)
            all_variants = [
                theme.get("intervals",  []),
                theme.get("inverted",   []),
                theme.get("retrograde", []),
                list(reversed(theme.get("inverted", []))),
            ]
            base_ivs  = all_variants[phrase_num % 4]
            base_durs = theme.get("durations", [1.0])
            n_full    = len(base_durs)

            # Motif development arc — controls how many notes are used
            if theme_dev == "build":
                # Sparse → full: start with 2 notes, grow each phrase
                n_use = max(2, int(2 + stage * (n_full - 2) + 0.5))
            elif theme_dev == "fragment":
                # Full → sparse: start complete, shrink each phrase
                n_use = max(2, int(n_full - stage * (n_full - 2) + 0.5))
            elif theme_dev == "arch":
                # Peak at middle: grow then shrink
                n_use = max(2, int(n_full * (1 - abs(2 * stage - 1)) + 0.5))
                n_use = max(2, n_use)
            else:
                # "flat" and "sequence": use full motif
                n_use = n_full

            ivs  = base_ivs[:max(0, n_use - 1)]
            durs = base_durs[:n_use]

            # Sequence: shift starting pitch up by 2 semitones each phrase
            seq_shift  = phrase_num * 2 if theme_dev == "sequence" else 0
            start_midi = midi + seq_shift

            theme_total = sum(durs)

            # Build theme pitches
            all_scale  = scale_midi_in_range(tonic_pc, mode_str, start_midi - 24, start_midi + 24)
            t_pitches: list[int] = [start_midi]
            for iv in ivs:
                next_p = t_pitches[-1] + iv
                nearby = [p for p in all_scale if abs(p - next_p) <= 2]
                t_pitches.append(nearest_pitch(next_p, nearby) if nearby else next_p)

            theme_notes: list[Note] = [
                Note(midi_to_note(p), d, max(50, vel - i))
                for i, (p, d) in enumerate(zip(t_pitches, durs))
            ]

            # Fill any remaining event duration after the theme
            remaining = dur - theme_total
            if remaining > 0.1:
                last_p   = t_pitches[-1]
                fill_sub = _subdiv(density, remaining)
                n_fill   = max(1, int(round(remaining / fill_sub)))
                fill_pts = [last_p]
                for _ in range(1, n_fill):
                    prev  = fill_pts[-1]
                    steps = [p for p in scale_ns if 0 < abs(p - prev) <= 2]
                    fill_pts.append(rng.choice(steps) if steps else prev)
                theme_notes += [
                    Note(midi_to_note(p), fill_sub, max(50, vel - 10 - i))
                    for i, p in enumerate(fill_pts)
                ]
            return theme_notes

        sub = _subdiv(density, dur)
        n   = max(1, int(round(dur / sub)))
        if n == 1:
            return [Note(midi_to_note(midi), dur, vel)]

        pitches = [midi]
        # Phrase-level direction: use phrase_progress for arch shape
        if mel_dir == "ascending":
            direction = 1
        elif mel_dir == "descending":
            direction = -1
        elif mel_dir == "arch":
            direction = 1 if phrase_progress < 0.6 else -1
        else:
            direction = rng.choice([-1, 1])

        for i in range(1, n):
            prev = pitches[-1]
            if mel_dir == "arch":
                # Within-event: also shift direction at local midpoint for finer shaping
                local_dir = 1 if (phrase_progress < 0.6 and i < n * 0.7) else -1
                direction = local_dir

            if rng.random() < step_leap:
                # Stepwise from scale
                steps = [p for p in scale_ns if 0 < abs(p - prev) <= 2]
                directed = [p for p in steps if (p - prev) * direction > 0]
                pool = directed or steps or scale_ns
            else:
                # Harmonic leap
                leaps = [p for p in chord_ns if abs(p - prev) in (3, 4, 5, 7)]
                directed = [p for p in leaps if (p - prev) * direction > 0]
                pool = directed or leaps or chord_ns or scale_ns

            if pool:
                pitches.append(nearest_pitch(prev + direction, pool))
            else:
                pitches.append(prev)

        # Melodic sequence: shift last motif by the interval between last motif
        # start and current note
        if last_midi and len(last_midi) >= 2 and rng.random() < seq_prob:
            interval = pitches[0] - last_midi[0]
            if 0 < abs(interval) <= 5:
                seq = [m + interval for m in last_midi[:n]]
                if len(seq) < n:
                    seq.extend([seq[-1]] * (n - len(seq)))
                pitches = seq[:n]

        # Ornament: trill on a randomly selected note
        if use_orn and n >= 3 and rng.random() < 0.22:
            pos = rng.randint(0, n - 2)
            upper = closest_above(pitches[pos], scale_ns)
            trill_dur = sub * 0.5
            notes_out: list[Note] = []
            for i, p in enumerate(pitches):
                if i == pos:
                    notes_out += [
                        Note(midi_to_note(p),     trill_dur, vel),
                        Note(midi_to_note(upper), trill_dur, vel - 8),
                    ]
                else:
                    notes_out.append(Note(midi_to_note(p), sub, vel - min(i * 1, 12)))
            return notes_out

        # Mordent: quick lower-neighbour turn on first note
        if use_orn and n >= 2 and rng.random() < 0.18:
            lower = closest_below(pitches[0], scale_ns)
            third  = sub / 3
            rest: list[Note] = [Note(midi_to_note(p), sub, vel - 5) for p in pitches[1:]]
            return [
                Note(midi_to_note(pitches[0]), third, vel),
                Note(midi_to_note(lower),       third, vel - 10),
                Note(midi_to_note(pitches[0]), third, vel),
            ] + rest

        # Rhythmic variety: long-short patterns scaled by (1 - regularity)
        if n >= 2 and rng.random() < (1.0 - regularity) * 0.55:
            profile = rng.choice(["long_short", "short_long"])
            if profile == "long_short":
                raw_durs = [sub * (1.5 if i % 2 == 0 else 0.5) for i in range(n)]
            else:
                raw_durs = [sub * (0.5 if i % 2 == 0 else 1.5) for i in range(n)]
            # Normalize so durations sum to n*sub (preserving approximate total)
            total_d = sum(raw_durs)
            scale   = (n * sub) / total_d if total_d > 0 else 1.0
            raw_durs = [max(0.125, d * scale) for d in raw_durs]
        else:
            raw_durs = [sub] * n

        return [
            Note(midi_to_note(p), raw_durs[i], max(45, vel - i * 2))
            for i, p in enumerate(pitches)
        ]

    # ── Bass patterns ────────────────────────────────────────────────────────

    def _bass(self, rng, midi, dur, bass_type, chord_ns, scale_ns, density, vel) -> list[Note]:
        if bass_type == "sustained":
            return [Note(midi_to_note(midi), dur, vel)]

        elif bass_type == "pedal":
            sub = max(0.5, _subdiv(density * 0.6, dur))
            n   = max(1, int(round(dur / sub)))
            return [Note(midi_to_note(midi), sub, vel - (4 if i else 0)) for i in range(n)]

        elif bass_type == "walking":
            sub = 1.0   # quarter notes
            n   = max(1, int(round(dur / sub)))
            if n == 1:
                return [Note(midi_to_note(midi), dur, vel)]
            pitches = [midi]   # voice leading already placed midi on a chord tone
            for beat in range(1, n):
                prev_p = pitches[-1]
                # Strong beats (even within the event): target chord tones
                # Weak beats (odd): use scale passing tones
                if beat % 2 == 0 and chord_ns:
                    opts = [p for p in chord_ns if 0 < abs(p - prev_p) <= 5]
                    if not opts:
                        opts = [p for p in chord_ns if p != prev_p]
                else:
                    opts = [p for p in scale_ns if 0 < abs(p - prev_p) <= 2]
                    if not opts:
                        opts = [p for p in scale_ns if 0 < abs(p - prev_p) <= 4]
                if opts:
                    # Approach: bias direction toward nearest chord tone above/below
                    bias = 1 if rng.random() < 0.55 else -1
                    pitches.append(nearest_pitch(prev_p + bias, opts))
                else:
                    pitches.append(prev_p)
            return [Note(midi_to_note(p), sub, vel - (3 if i else 0))
                    for i, p in enumerate(pitches)]

        elif bass_type == "alberti":
            s_chord = sorted(chord_ns) if chord_ns else [midi, midi + 4, midi + 7]
            low  = s_chord[0]
            mid  = s_chord[len(s_chord) // 2]
            high = s_chord[-1]
            pat  = [low, high, mid, high]
            sub  = 0.5
            n    = max(1, int(round(dur / sub)))
            return [
                Note(midi_to_note(pat[i % 4]), sub, vel - (0 if i % 4 == 0 else 6))
                for i in range(n)
            ]

        elif bass_type == "arpeggiated":
            s_chord = sorted(chord_ns) if chord_ns else [midi, midi + 4, midi + 7]
            sub  = _subdiv(density, dur)
            n    = max(1, int(round(dur / sub)))
            pat  = s_chord + s_chord[-2:0:-1]   # up then back down
            return [
                Note(midi_to_note(pat[i % len(pat)]), sub, vel - (3 if i else 0))
                for i in range(n)
            ]

        elif bass_type == "rock":
            # Driving quarter-note root-fifth pattern: R, 5th, R, 8va
            sub = 1.0
            n   = max(1, int(round(dur / sub)))
            if n == 1:
                return [Note(midi_to_note(midi), dur, vel)]
            fifth  = midi + 7
            octave = midi + 12
            if octave > VOICE_RANGES["bass"][1]:
                octave = fifth
            pat = [midi, fifth, midi, octave]
            return [
                Note(midi_to_note(pat[i % len(pat)]), sub,
                     vel + (4 if i % 2 == 0 else -4))
                for i in range(n)
            ]

        return [Note(midi_to_note(midi), dur, vel)]

    # ── Inner voice patterns ─────────────────────────────────────────────────

    def _inner(self, rng, midi, dur, style, chord_ns, scale_ns, density, vel, voice) -> list[Note]:
        v = vel - 14   # inner voices quieter

        if style == "block_chords":
            return [Note(midi_to_note(midi), dur, v)]

        elif style == "arpeggiated":
            s_chord = sorted(chord_ns) if chord_ns else [midi]
            sub  = _subdiv(density * 0.8, dur)
            n    = max(1, int(round(dur / sub)))
            return [Note(midi_to_note(s_chord[i % len(s_chord)]), sub, v) for i in range(n)]

        elif style == "running_sixteenths":
            sub  = 0.25
            n    = max(1, int(round(dur / sub)))
            dir_ = 1 if rng.random() < 0.5 else -1
            pitches = [midi]
            for _ in range(1, n):
                prev  = pitches[-1]
                steps = [p for p in scale_ns if 0 < abs(p - prev) <= 2]
                directed = [p for p in steps if (p - prev) * dir_ > 0]
                if not directed:
                    dir_ *= -1
                    directed = [p for p in steps if (p - prev) * dir_ > 0]
                pitches.append(directed[0] if directed else prev)
            return [
                Note(midi_to_note(p), sub, v + (4 if i % 4 == 0 else 0))
                for i, p in enumerate(pitches)
            ]

        elif style == "countermelody":
            sub = _subdiv(density * 0.65, dur)
            n   = max(1, int(round(dur / sub)))
            prev_p = midi
            notes_out: list[Note] = []
            for _ in range(n):
                pool = chord_ns if rng.random() < 0.6 else scale_ns
                opts = [p for p in pool if p != prev_p] if pool else [prev_p]
                if opts:
                    delta = rng.choice([-2, -1, 1, 2])
                    prev_p = nearest_pitch(prev_p + delta, opts)
                notes_out.append(Note(midi_to_note(prev_p), sub, v))
            return notes_out

        return [Note(midi_to_note(midi), dur, v)]

    # ── Swing ────────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_swing(notes: list[Note], swing: float, beats_bar: int) -> list[Note]:
        """Lengthen every odd 8th note and shorten the following even one."""
        if not notes:
            return notes
        out: list[Note] = []
        tick = 0.0
        for n in notes:
            beat_pos = tick % 1.0   # position within a beat
            if abs(n.duration - 0.5) < 0.01 and abs(beat_pos - 0.5) < 0.05:
                # This is a "short" 8th in a pair — shorten further
                long  = 0.5 + swing * 0.1667
                short = 0.5 - swing * 0.1667
                # The previous note (if 8th on beat) should be lengthened
                if out and abs(out[-1].duration - 0.5) < 0.05:
                    out[-1] = Note(out[-1].pitch, long,  out[-1].velocity)
                    out.append(Note(n.pitch,   short, n.velocity))
                    tick += n.duration
                    continue
            out.append(n)
            tick += n.duration
        return out


# ── Generator 4: Percussion ──────────────────────────────────────────────────

class PercussionGenerator(GeneratorBase):
    """
    Generates a GM-compatible percussion track using rest-encoded Note timing.

    Encoding contract (shared with export.to_midi):
        Note("rest", gap_beats, 0)   — advance clock; no MIDI event emitted
        Note(voice_name, 0.0, vel)   — drum hit at current position

    Patterns are lists of (beat_within_bar, voice_name, vel_factor).
    """

    # ── Pattern banks ─────────────────────────────────────────────────────────

    PATTERNS_4_4: dict[str, list] = {
        "rock": [
            (0.0, "kick",         1.00),
            (0.5, "hihat_closed", 0.65),
            (1.0, "snare",        1.00),
            (1.5, "hihat_closed", 0.65),
            (2.0, "kick",         1.00),
            (2.5, "hihat_closed", 0.65),
            (3.0, "snare",        1.00),
            (3.5, "hihat_closed", 0.65),
        ],
        "jazz": [
            (0.0,  "ride",  0.80), (0.5,  "ride",  0.55),
            (1.0,  "ride",  0.72), (1.5,  "ride",  0.55),
            (2.0,  "kick",  0.70), (2.0,  "ride",  0.80),
            (2.5,  "ride",  0.55),
            (3.0,  "snare", 0.65), (3.0,  "ride",  0.72),
            (3.5,  "ride",  0.55),
        ],
        "blues": [
            (0.0,  "kick",         1.00), (0.0,  "hihat_closed", 0.60),
            (0.67, "hihat_closed", 0.50),
            (1.0,  "hihat_closed", 0.65),
            (1.33, "snare",        0.90), (1.33, "hihat_closed", 0.50),
            (2.0,  "kick",         1.00), (2.0,  "hihat_closed", 0.60),
            (2.67, "hihat_closed", 0.50),
            (3.0,  "hihat_closed", 0.65),
            (3.33, "snare",        0.90), (3.33, "hihat_closed", 0.50),
        ],
        "funk": [
            (0.0,  "kick",         1.00),
            (0.25, "hihat_closed", 0.55), (0.5,  "hihat_closed", 0.70),
            (0.75, "snare",        0.55), (0.75, "hihat_closed", 0.55),
            (1.0,  "hihat_closed", 0.70), (1.25, "kick",         0.85),
            (1.5,  "snare",        1.00), (1.5,  "hihat_closed", 0.70),
            (1.75, "hihat_closed", 0.55),
            (2.0,  "kick",         1.00),
            (2.25, "hihat_closed", 0.55), (2.5,  "hihat_closed", 0.70),
            (2.75, "hihat_closed", 0.55),
            (3.0,  "snare",        0.85), (3.0,  "hihat_closed", 0.70),
            (3.25, "kick",         0.70),
            (3.5,  "hihat_closed", 0.70),
            (3.75, "snare",        0.60), (3.75, "hihat_closed", 0.55),
        ],
        "latin": [
            (0.0,  "kick",      0.90), (0.0,  "ride",      0.70),
            (0.5,  "ride",      0.60), (0.75, "snare_rim", 0.80),
            (1.0,  "ride",      0.70), (1.5,  "kick",      0.75),
            (1.5,  "ride",      0.60),
            (2.0,  "ride",      0.70), (2.25, "snare_rim", 0.80),
            (2.5,  "ride",      0.60),
            (3.0,  "ride",      0.70), (3.0,  "kick",      0.85),
            (3.5,  "ride",      0.60), (3.75, "snare_rim", 0.75),
        ],
        "brush": [
            (0.0, "hihat_closed", 0.50), (0.5, "hihat_closed", 0.40),
            (1.0, "snare",        0.55), (1.5, "hihat_closed", 0.40),
            (2.0, "hihat_closed", 0.50), (2.5, "hihat_closed", 0.40),
            (3.0, "snare",        0.55), (3.5, "hihat_closed", 0.40),
        ],
    }

    PATTERNS_3_4: dict[str, list] = {
        "waltz": [
            (0.0, "kick",         1.00),
            (1.0, "hihat_closed", 0.70),
            (2.0, "hihat_closed", 0.70),
        ],
        "jazz": [
            (0.0, "kick",  0.80), (0.0, "ride", 0.80),
            (0.5, "ride",  0.55),
            (1.0, "ride",  0.70), (1.5, "ride", 0.55),
            (2.0, "snare", 0.65), (2.0, "ride", 0.70),
            (2.5, "ride",  0.55),
        ],
        "rock": [
            (0.0, "kick",         1.00),
            (0.5, "hihat_closed", 0.65),
            (1.0, "snare",        0.90),
            (1.5, "hihat_closed", 0.65),
            (2.0, "kick",         0.85),
            (2.5, "hihat_closed", 0.65),
        ],
    }

    PATTERNS_6_8: dict[str, list] = {
        "rock": [
            (0.0, "kick",         1.00), (0.5, "hihat_closed", 0.60),
            (1.0, "hihat_closed", 0.65), (1.5, "hihat_closed", 0.60),
            (2.0, "snare",        0.90), (2.5, "hihat_closed", 0.60),
            (3.0, "kick",         0.85), (3.5, "hihat_closed", 0.60),
            (4.0, "hihat_closed", 0.65), (4.5, "hihat_closed", 0.60),
            (5.0, "snare",        0.80), (5.5, "hihat_closed", 0.60),
        ],
        "jazz": [
            (0.0, "ride",  0.80), (1.0, "ride",  0.60),
            (2.0, "snare", 0.65), (2.0, "ride",  0.70),
            (3.0, "kick",  0.70), (3.0, "ride",  0.80),
            (4.0, "ride",  0.60),
            (5.0, "snare", 0.65), (5.0, "ride",  0.70),
        ],
        "latin": [
            (0.0, "kick",      1.00), (0.0, "ride",      0.75),
            (1.0, "ride",      0.60),
            (2.0, "snare_rim", 0.80), (2.0, "ride",      0.70),
            (3.0, "ride",      0.75),
            (4.0, "ride",      0.60),
            (5.0, "snare_rim", 0.75), (5.0, "ride",      0.70),
        ],
    }

    BASE_VEL: dict[str, int] = {
        "kick": 100, "snare": 90, "snare_rim": 72, "snare_electric": 90,
        "hihat_closed": 65, "hihat_pedal": 55, "hihat_open": 75,
        "ride": 68, "ride_bell": 80, "crash": 110, "crash2": 100,
        "tom_hi": 85, "tom_hi_mid": 82, "tom_lo_mid": 80, "tom_lo": 78, "tom_floor": 82,
        "cowbell": 75,
    }

    def generate(self, params: dict, context: dict) -> list:
        """Return list[Note] for the percussion track (rest-encoded timing)."""
        import random

        drum_style = params.get("drum_style", "none")
        if drum_style == "none":
            return []

        intensity     = float(params.get("drum_intensity", 0.7))
        swing         = float(params.get("swing", 0.0))
        time_sig      = params.get("time_signature", [4, 4])
        num, den      = int(time_sig[0]), int(time_sig[1])
        beats_per_bar = num * (4.0 / den)   # in quarter-note beats

        harm_events = context.get("harmonic_plan", [])
        total_beats = sum(ev.duration_beats for ev in harm_events)
        if total_beats <= 0:
            return []

        rng = random.Random(self.seed)

        # Select pattern bank by time signature numerator
        if num == 3:
            bank = self.PATTERNS_3_4
        elif num == 6:
            bank = self.PATTERNS_6_8
        else:
            bank = self.PATTERNS_4_4

        # Graceful fallback: requested style -> rock -> first available
        pattern = bank.get(drum_style) or bank.get("rock") or next(iter(bank.values()))

        phrase_boundaries = self._phrase_boundaries(harm_events)

        hits: list[tuple[float, str, int]] = []
        bar_start = 0.0
        bar_num   = 0

        while bar_start < total_beats - 0.001:
            bar_end = min(bar_start + beats_per_bar, total_beats)

            # Crash cymbal on phrase boundary downbeats (skip bar 0)
            if bar_start in phrase_boundaries and bar_num > 0:
                vel = int(min(127, self.BASE_VEL["crash"] * intensity))
                hits.append((bar_start, "crash", vel))

            for beat_in_bar, voice, vel_factor in pattern:
                abs_beat = bar_start + beat_in_bar
                if abs_beat >= bar_end - 0.001:
                    break
                abs_beat = self._swing_beat(abs_beat, swing, beats_per_bar)
                vel = int(min(127, max(1, self.BASE_VEL.get(voice, 80) * vel_factor * intensity)))
                hits.append((abs_beat, voice, vel))

            # Fill in the last bar before each phrase boundary
            next_boundary = next((b for b in sorted(phrase_boundaries) if b > bar_start), None)
            if next_boundary is not None and abs(next_boundary - bar_end) < 0.001:
                hits.extend(self._fill_hits(bar_start, beats_per_bar, intensity, rng))

            bar_start += beats_per_bar
            bar_num   += 1

        hits.sort(key=lambda h: (h[0], h[1]))
        return self._hits_to_notes(hits, total_beats)

    @staticmethod
    def _swing_beat(abs_beat: float, swing: float, beats_per_bar: float) -> float:
        """Push offbeat 8th-note positions (x.5) forward by swing * 0.167 beats."""
        if swing <= 0.0:
            return abs_beat
        frac = abs_beat % 1.0
        if abs(frac - 0.5) < 0.02:
            return abs_beat + swing * 0.167
        return abs_beat

    @staticmethod
    def _phrase_boundaries(harm_events) -> set:
        """Return set of absolute beat positions where phrase_idx changes."""
        boundaries: set[float] = set()
        abs_beat   = 0.0
        prev_idx   = None
        for ev in harm_events:
            if ev.phrase_idx != prev_idx and prev_idx is not None:
                boundaries.add(round(abs_beat, 6))
            prev_idx  = ev.phrase_idx
            abs_beat += ev.duration_beats
        return boundaries

    @staticmethod
    def _fill_hits(
        bar_start: float,
        beats_per_bar: float,
        intensity: float,
        rng,
    ) -> list[tuple[float, str, int]]:
        """Snare roll or tom fall in the last beat before a phrase boundary."""
        if intensity < 0.35 or rng.random() > intensity * 0.8:
            return []
        fill_start = bar_start + beats_per_bar - 1.0
        if fill_start < 0:
            return []
        hits = []
        if rng.random() < 0.5:
            # Snare roll: four 16th notes, building velocity
            for k in range(4):
                vel = int(min(127, (55 + k * 10) * intensity))
                hits.append((fill_start + k * 0.25, "snare", vel))
        else:
            # Tom fall: hi -> hi-mid -> lo-mid -> floor
            for k, tom in enumerate(["tom_hi", "tom_hi_mid", "tom_lo_mid", "tom_floor"]):
                vel = int(min(127, 82 * intensity))
                hits.append((fill_start + k * 0.25, tom, vel))
        return hits

    @staticmethod
    def _hits_to_notes(hits: list, total_beats: float) -> list:
        """
        Convert sorted (abs_beat, voice, vel) tuples to rest-encoded Notes.

        Note("rest", gap, 0)    — advance clock (velocity=0, no MIDI event)
        Note(voice, 0.0, vel)   — drum hit at current position (duration=0)
        """
        notes: list[Note] = []
        cursor = 0.0
        for abs_beat, voice, vel in hits:
            gap = abs_beat - cursor
            if gap > 0.001:
                notes.append(Note("rest", gap, 0))
                cursor = abs_beat
            notes.append(Note(voice, 0.0, vel))
        # Pad to total length
        if cursor < total_beats - 0.001:
            notes.append(Note("rest", total_beats - cursor, 0))
        return notes


# ── Generator 5: Score Assembly ────────────────────────────────────────────────

class ScoreAssemblyGenerator(GeneratorBase):
    """Maps elaborated voice notes to instruments and builds the MusicScore."""

    def generate(self, params: dict, context: dict) -> MusicScore:
        elaborated: dict[str, list[Note]] = context["elaboration"]
        perc_notes: list = context.get("percussion", [])
        instruments: list[str] = list(params["instruments"])
        num_voices  = params["num_voices"]
        roles       = VOICE_ROLES[num_voices]

        # Distribute instruments across roles
        while len(instruments) < len(roles):
            instruments.append(instruments[-1] if instruments else "piano")

        voice_instr = {
            "soprano": instruments[0],
            "alto":    instruments[1] if len(instruments) > 1 else instruments[0],
            "tenor":   instruments[2] if len(instruments) > 2 else instruments[0],
            "bass":    instruments[-1],
        }

        tracks = [
            Track(instrument=voice_instr[v], notes=elaborated.get(v, []))
            for v in roles
        ]

        if perc_notes:
            tracks.append(Track(instrument="drums", notes=perc_notes))

        all_roles = list(roles) + (["drums"] if perc_notes else [])

        return MusicScore(
            tempo_bpm      = int(params["tempo_bpm"]),
            time_signature = (int(params["time_signature"][0]),
                              int(params["time_signature"][1])),
            tracks         = tracks,
            metadata       = {
                "intent":  params.get("_intent_summary", {}),
                "voices":  all_roles,
                "form":    params.get("form"),
            },
        )


# ── MusicDomain ────────────────────────────────────────────────────────────────

class MusicDomain(DomainBase):
    """
    Procedural music composition domain.

    Produces a MusicScore ready for export via procgen.export.to_midi().
    See module docstring for full parameter documentation.
    """

    def __init__(self):
        from procgen.constraints import ConstraintSet, Rule
        _keys = sorted(ROOT_TO_PC.keys())
        _modes = list(SCALE_INTERVALS.keys())

        self.schema = ConstraintSet([
            # ── Structure ──
            Rule("form",                type=str,   default="binary",
                 choices=["binary", "ternary", "rondo", "through_composed"],
                 description="Large-scale musical form"),
            Rule("num_phrases",         type=int,   default=4,   range=(2, 8),
                 description="Number of phrases in the piece"),
            Rule("phrase_length",       type=int,   default=4,   range=(2, 16),
                 description="Bars per phrase"),

            # ── Pitch / Harmony ──
            Rule("key",                 type=str,   default="C", choices=_keys,
                 description="Tonic pitch class"),
            Rule("mode",                type=str,   default="major", choices=_modes,
                 description="Scale mode"),
            Rule("chord_complexity",    type=float, default=0.35, range=(0.0, 1.0),
                 description="0=triads only, 1=7ths, dim7s, secondary dominants"),
            Rule("harmonic_rhythm",     type=float, default=0.5,  range=(0.0, 1.0),
                 description="0=one chord/phrase, 1=one chord/beat"),
            Rule("circle_of_fifths",    type=float, default=0.6,  range=(0.0, 1.0),
                 description="Strength of dominant→tonic resolution"),
            Rule("modal_mixture",       type=float, default=0.1,  range=(0.0, 1.0),
                 description="Borrowing from parallel major/minor"),
            Rule("pedal_point",         type=float, default=0.1,  range=(0.0, 1.0),
                 description="Probability of bass pedal sections"),

            # ── Texture / Voices ──
            Rule("num_voices",          type=int,   default=3,   range=(1, 4),
                 description="Number of independent voices"),
            Rule("texture",             type=str,   default="polyphonic",
                 choices=["monophonic", "homophonic", "polyphonic"],
                 description="Overall texture type"),
            Rule("counterpoint",        type=float, default=0.5,  range=(0.0, 1.0),
                 description="0=block chords, 1=fully independent voices"),
            Rule("imitation",           type=float, default=0.0,  range=(0.0, 1.0),
                 description="Degree of canonic imitation between voices"),
            Rule("voice_independence",  type=float, default=0.5,  range=(0.0, 1.0),
                 description="0=parallel motion, 1=contrary motion"),

            # ── Bass ──
            Rule("bass_type",           type=str,   default="walking",
                 choices=["sustained", "walking", "alberti", "arpeggiated", "pedal", "rock"],
                 description="Bass voice rhythmic pattern"),

            # ── Inner voices ──
            Rule("inner_voice_style",   type=str,   default="countermelody",
                 choices=["block_chords", "arpeggiated", "countermelody", "running_sixteenths"],
                 description="Inner voice elaboration style"),

            # ── Melody ──
            Rule("melodic_range",       type=int,   default=2,   range=(1, 3),
                 description="Octave span the melody may traverse"),
            Rule("step_leap_ratio",     type=float, default=0.70, range=(0.0, 1.0),
                 description="0=all leaps, 1=all stepwise motion"),
            Rule("sequence_probability",type=float, default=0.3,  range=(0.0, 1.0),
                 description="Likelihood of repeating a melodic pattern at a new pitch"),
            Rule("use_ornaments",       type=bool,  default=False,
                 description="Add trills and mordents"),
            Rule("melodic_direction",   type=str,   default="free",
                 choices=["ascending", "descending", "arch", "free"],
                 description="Overall contour bias of the melody"),

            # ── Rhythm ──
            Rule("tempo_bpm",           type=int,   default=120,  range=(40, 240),
                 description="Beats per minute"),
            Rule("time_signature",      type=list,  default=[4, 4],
                 description="[numerator, denominator] e.g. [3,4] for 3/4"),
            Rule("rhythmic_density",    type=float, default=0.5,  range=(0.0, 1.0),
                 description="0=whole notes, 1=thirty-second notes"),
            Rule("rhythmic_regularity", type=float, default=0.6,  range=(0.0, 1.0),
                 description="0=syncopated, 1=metronomic subdivision"),
            Rule("swing",               type=float, default=0.0,  range=(0.0, 1.0),
                 description="Swing feel on 8th notes (0=straight, 1=full swing)"),

            # ── Instrumentation ──
            Rule("instruments",         type=list,  default=["piano"],
                 description="List of instruments (one per voice, or repeated)"),

            # ── Melodic theme ──
            Rule("melodic_theme",       type=str,   default="none",
                 choices=["none", "motif", "fugue_subject"],
                 description="Recurring motif applied at phrase starts with cycled transformations"),

            # ── Drums ──
            Rule("drum_style",          type=str,   default="none",
                 choices=["none", "rock", "jazz", "blues", "funk", "latin", "waltz", "brush"],
                 description="Percussion pattern style (none = no drums)"),
            Rule("drum_intensity",      type=float, default=0.7,  range=(0.0, 1.0),
                 description="Drum velocity scaling and fill density"),

            Rule("melodic_theme_dev",   type=str,   default="flat",
                 choices=["flat", "build", "fragment", "arch", "sequence"],
                 description="How the melodic theme evolves across phrases: "
                             "flat=cyclic transforms only, build=sparse-to-full, "
                             "fragment=full-to-sparse, arch=peak-at-middle, "
                             "sequence=shift motif up scale each phrase"),

            # ── Rock / power chords ──
            Rule("power_chords",        type=bool,  default=False,
                 description="Replace all triads with power chords (root+fifth only)"),

            # ── Beginning / Ending ──
            Rule("ending",              type=str,   default="held",
                 choices=["none", "held", "ritardando"],
                 description="none=normal, held=sustain final chord on tonic, "
                             "ritardando=slow penultimate event then hold"),
            Rule("intro",               type=str,   default="none",
                 choices=["none", "upbeat"],
                 description="none=normal start, upbeat=ascending pickup figure in soprano"),
        ])

    # ── plan ──────────────────────────────────────────────────────────────────

    def build_plan(self, intent: Intent) -> GenerationPlan:
        plan = GenerationPlan()

        keys = [
            "form", "num_phrases", "phrase_length",
            "key", "mode", "chord_complexity", "harmonic_rhythm",
            "circle_of_fifths", "modal_mixture", "pedal_point",
            "num_voices", "texture", "counterpoint", "imitation", "voice_independence",
            "bass_type", "inner_voice_style",
            "melodic_range", "step_leap_ratio", "sequence_probability",
            "use_ornaments", "melodic_direction",
            "tempo_bpm", "time_signature", "rhythmic_density", "rhythmic_regularity", "swing",
            "instruments",
            "melodic_theme",
            "melodic_theme_dev",
            "power_chords",
            "ending",
            "intro",
            "drum_style",
            "drum_intensity",
        ]
        p: dict[str, Any] = {k: intent.get(k) for k in keys}
        p["_intent_summary"] = dict(intent.items())

        plan.add_step("harmonic_plan", p)
        plan.add_step("theme",         p)
        plan.add_step("voice_lines",   p)
        plan.add_step("elaboration",   p)
        plan.add_step("percussion",    p)
        plan.add_step("score",         p)
        return plan

    # ── assemble ──────────────────────────────────────────────────────────────

    def assemble(self, outputs: dict[str, Any], plan: GenerationPlan) -> MusicScore:
        return outputs["score"]

    # ── generators ────────────────────────────────────────────────────────────

    @property
    def generators(self):
        return {
            "harmonic_plan": HarmonicPlanGenerator,
            "theme":         ThemeGenerator,
            "voice_lines":   VoiceLeadingGenerator,
            "elaboration":   ElaborationGenerator,
            "percussion":    PercussionGenerator,
            "score":         ScoreAssemblyGenerator,
        }
