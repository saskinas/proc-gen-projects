"""
remixer.pipeline — Core analyze / describe / remix pipeline.
"""

from __future__ import annotations

import pathlib
from typing import Callable, Union

from audio_listener import listen_midi
from music_generator.compose import generate_from_analysis
from music_generator import ROOT_TO_PC, SCALE_INTERVALS
from procgen import export


# ── Default base params (NES-appropriate, adapts to analysis) ─────────────────

_DEFAULT_BASE = dict(
    num_voices=3,
    texture="polyphonic",
    counterpoint=0.35,
    imitation=0.0,
    voice_independence=0.5,
    chord_complexity=0.20,
    harmonic_rhythm=0.4,
    circle_of_fifths=0.70,
    modal_mixture=0.05,
    pedal_point=0.0,
    melodic_range=2,
    step_leap_ratio=0.60,
    sequence_probability=0.35,
    use_ornaments=False,
    swing=0.0,
    rhythmic_regularity=0.70,
    rhythmic_density=0.55,
    melodic_theme="motif",
    melodic_theme_dev="flat",
    form="binary",
    num_phrases=2,
    phrase_length=4,
    bass_type="walking",
    inner_voice_style="countermelody",
    instruments=["square"],
    ending="held",
    intro="none",
    power_chords=False,
    drum_style="none",
    drum_intensity=0.5,
)


def _adapt_base(analysis, overrides: dict | None) -> dict:
    """Merge analysis-derived hints into base params."""
    from collections import Counter

    base = dict(_DEFAULT_BASE)

    # Voice count from analysis
    base["num_voices"] = max(2, min(4, max(
        (sec.texture.num_voices or 2) for sec in analysis.sections
    )))
    base["melodic_theme"] = "motif" if analysis.motifs else "none"

    # Drum style from most common detected style
    style_counts = Counter(
        sec.texture.drum_style
        for sec in analysis.sections
        if sec.texture.drum_style and sec.texture.drum_style != "none"
    )
    if style_counts:
        base["drum_style"] = style_counts.most_common(1)[0][0]
        intensities = [
            sec.texture.drum_intensity for sec in analysis.sections
            if sec.texture.drum_intensity and sec.texture.drum_intensity > 0
        ]
        base["drum_intensity"] = sum(intensities) / len(intensities) if intensities else 0.5

    # Detect instrument style from the source MIDI's program-change events.
    # If any voice uses an acoustic-range GM program (0-63: piano, guitar,
    # strings, brass, woodwinds…), the source is general MIDI rather than
    # NES/chiptune.  Use piano as the default instrument for generated voices
    # so the accompaniment blends with the replayed melody instead of clashing
    # with a synthetic square-wave sound.
    # NES-converted MIDIs either have no program changes (empty dict) or use
    # programs ≥ 80 (synth lead family), so they keep the "square" default.
    all_src_progs: set[int] = set()
    for sec in analysis.sections:
        progs = sec.extra_params.get("_source_programs", {})
        all_src_progs.update(progs.values())
    if all_src_progs and any(p < 64 for p in all_src_progs):
        base["instruments"] = ["piano"]

    if overrides:
        base.update(overrides)
    return base


# ── Public API ────────────────────────────────────────────────────────────────

def analyze(source: Union[str, pathlib.Path, bytes]):
    """
    Parse a MIDI file and return a MusicalAnalysis IR.

    Parameters
    ----------
    source  Path/str to a .mid file, or raw MIDI bytes.
    """
    return listen_midi(source)


def describe(analysis) -> str:
    """
    Return a formatted human-readable description of a MusicalAnalysis.

    Covers: key/mode/tempo/time signature, form, motifs with variants and
    occurrences, sections with voice inventory and harmonic plan summary.
    """
    lines: list[str] = []
    W = 66

    def hr(char="-"):
        lines.append("  " + char * W)

    def h1(title: str):
        lines.append("\n  " + "=" * W)
        lines.append(f"  {title}")
        lines.append("  " + "=" * W)

    # ── Header ────────────────────────────────────────────────────────────────
    h1(f"MUSICAL ANALYSIS")
    ts = analysis.time_signature
    lines.append(
        f"\n  Key / Mode   : {analysis.key} {analysis.mode}"
        f"    Tempo: {analysis.tempo_bpm} BPM"
        f"    Time: {ts[0]}/{ts[1]}"
    )
    lines.append(
        f"  Form hint    : {analysis.form_hint}"
        f"    Sections: {len(analysis.sections)}"
        f"    Motifs: {len(analysis.motifs)}"
    )

    # ── Motifs ────────────────────────────────────────────────────────────────
    if analysis.motifs:
        lines.append("\n  MOTIFS")
        hr()
        for m in analysis.motifs.values():
            contour_str = " ".join(
                "+" if c > 0 else ("-" if c < 0 else ".") for c in m.contour
            )
            occ_count = len(m.occurrences)
            transforms_seen = sorted(set(
                o.get("transform", "exact") for o in m.occurrences
            ))
            lines.append(
                f"  {m.id:<18} [{m.type}]  {m.rhythm_class}"
            )
            lines.append(
                f"    intervals : {m.intervals}"
            )
            lines.append(
                f"    contour   : [{contour_str}]"
            )
            if m.sequence_step is not None:
                lines.append(f"    seq step  : {m.sequence_step:+d} semitones")
            lines.append(
                f"    occurrences: {occ_count}"
                + (f"  variants: {', '.join(transforms_seen)}" if transforms_seen else "")
            )
            if m.occurrences:
                occ_parts = []
                for o in m.occurrences[:5]:
                    ph  = o.get("phrase_idx", "?")
                    tr  = o.get("transform", "exact")
                    occ_parts.append(f"ph{ph}({tr})")
                suffix = "…" if len(m.occurrences) > 5 else ""
                lines.append(f"    where     : {', '.join(occ_parts)}{suffix}")
            lines.append("")

    # ── Sections ──────────────────────────────────────────────────────────────
    lines.append("  SECTIONS")
    hr()
    for sec in analysis.sections:
        ep       = sec.extra_params
        key_str  = ep.get("key",  analysis.key)
        mode_str = ep.get("mode", analysis.mode)
        tx       = sec.texture
        ep_      = sec.extra_params
        n_ph     = ep_.get("num_phrases",   "?")
        ph_len   = ep_.get("phrase_length", "?")

        # Chord summary
        if sec.harmonic_plan:
            chord_strs = []
            for ev in sec.harmonic_plan[:6]:
                deg  = getattr(ev, "degree",    getattr(ev, "root_degree", "?"))
                qual = getattr(ev, "chord_type", getattr(ev, "quality", ""))
                qual_abbr = {
                    "maj": "", "major": "",
                    "min": "m", "minor": "m",
                    "dom7": "7", "dominant": "7",
                    "maj7": "M7",
                    "min7": "m7",
                    "dim": "°", "diminished": "°",
                    "aug": "+", "augmented": "+",
                    "half_diminished": "ø",
                }.get(qual, "")
                chord_strs.append(f"{_roman(deg)}{qual_abbr}")
            suffix = "…" if len(sec.harmonic_plan) > 6 else ""
            chords = "  ".join(chord_strs) + suffix
        else:
            chords = "(auto)"

        lines.append(
            f"  {sec.id:<22}  {sec.role:<8}  {key_str} {mode_str}"
        )
        lines.append(
            f"    energy={sec.energy.level:.2f}/{sec.energy.arc}"
            f"  phrases={n_ph}×{ph_len}bars"
            f"  mode={sec.generation_mode}"
        )
        lines.append(f"    chords : {chords}")

        # Voice inventory
        if sec.voice_sequences:
            voice_parts = []
            for vname, seq in sec.voice_sequences.items():
                n_ev = len([e for e in seq if e.get("degree") != "rest"])
                voice_parts.append(f"{vname}({n_ev})")
            lines.append(f"    voices : {', '.join(voice_parts)}")

        lines.append("")

    return "\n".join(lines)


def remix(
    source:    Union[str, pathlib.Path, bytes],
    *transform_fns: Callable,
    base_params: dict | None = None,
    seed: int = 42,
) -> bytes:
    """
    Full remix pipeline: analyze → transform → generate → MIDI bytes.

    Parameters
    ----------
    source        Path/str to .mid file, or raw MIDI bytes.
    *transform_fns  Zero or more (MusicalAnalysis → MusicalAnalysis) callables.
                    Use presets from remixer.presets or music_generator.transforms.
    base_params   Optional dict to override generation base params.
    seed          RNG seed for deterministic generation.

    Returns
    -------
    bytes — Standard MIDI file ready to write to disk.

    Example
    -------
        from remixer import remix
        from remixer.presets import JAZZ, DARK_MINOR

        # Apply both presets in sequence:
        out = remix("grape-garden.mid", JAZZ, DARK_MINOR, seed=7)
    """
    analysis = analyze(source)
    for fn in transform_fns:
        analysis = fn(analysis)
    adapted = _adapt_base(analysis, base_params)
    score   = generate_from_analysis(analysis, base_params=adapted, seed=seed)
    return export.to_midi(score)


def remix_to_file(
    source:    Union[str, pathlib.Path, bytes],
    out_path:  Union[str, pathlib.Path],
    *transform_fns: Callable,
    base_params: dict | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> None:
    """
    Remix and write the result to *out_path*, printing a summary if verbose.

    Output format is determined by the file extension:
      .mid / .midi  → Standard MIDI file
      .wav          → 16-bit PCM WAV (requires numpy + mido)

    Parameters mirror remix(); out_path is created (including parents) if needed.
    """
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    analysis = analyze(source)
    original_key  = analysis.key
    original_mode = analysis.mode
    original_bpm  = analysis.tempo_bpm

    for fn in transform_fns:
        analysis = fn(analysis)

    adapted   = _adapt_base(analysis, base_params)
    score     = generate_from_analysis(analysis, base_params=adapted, seed=seed)
    midi_data = export.to_midi(score)

    if out_path.suffix.lower() == ".wav":
        from procgen.synthesize import midi_to_wav
        out_path.write_bytes(midi_to_wav(midi_data))
    else:
        out_path.write_bytes(midi_data)

    if verbose:
        n_notes = sum(len(t.notes) for t in score.tracks)
        xforms  = " -> ".join(f.__name__ for f in transform_fns) or "(none)"
        print(
            f"  {out_path.name:<42} "
            f"{original_key} {original_mode} -> {analysis.key} {analysis.mode}  "
            f"{original_bpm}->{analysis.tempo_bpm}bpm  "
            f"{n_notes}n  [{xforms}]"
        )


def remix_to_wav(
    source:    Union[str, pathlib.Path, bytes],
    out_path:  Union[str, pathlib.Path],
    *transform_fns: Callable,
    base_params: dict | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> None:
    """
    Remix and write the result as a WAV audio file.

    Equivalent to remix_to_file() with a .wav out_path.
    Requires numpy and mido to be installed.
    """
    out_path = pathlib.Path(out_path).with_suffix(".wav")
    remix_to_file(source, out_path, *transform_fns,
                  base_params=base_params, seed=seed, verbose=verbose)


# ── Internal helper ───────────────────────────────────────────────────────────

_ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII"]

def _roman(deg) -> str:
    try:
        return _ROMAN[int(deg)]
    except (IndexError, ValueError, TypeError):
        return str(deg)
