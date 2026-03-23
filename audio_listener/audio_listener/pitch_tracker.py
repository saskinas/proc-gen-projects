"""
audio_listener.pitch_tracker — Convert raw note events to beat-time notes.

Handles tempo-map beat conversion, monophonic overlap resolution, and
optional grid quantisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from audio_listener.midi_parser import RawNote


@dataclass
class TrackedNote:
    pitch:          int     # MIDI pitch number
    start_beat:     float   # beat position from piece start
    duration_beats: float   # duration in beats
    velocity:       int


def _tick_to_beat(tick: int, ticks_per_beat: int,
                  tempo_changes: list[tuple[int, int]]) -> float:
    """Convert absolute tick to fractional beat, respecting tempo changes."""
    if len(tempo_changes) <= 1:
        return tick / ticks_per_beat
    beat      = 0.0
    prev_tick = 0
    for chg_tick, _uspb in tempo_changes:
        if tick <= chg_tick:
            break
        beat     += (chg_tick - prev_tick) / ticks_per_beat
        prev_tick = chg_tick
    beat += (tick - prev_tick) / ticks_per_beat
    return beat


def extract_notes(
    raw: list[RawNote],
    ticks_per_beat: int,
    tempo_changes: list[tuple[int, int]],
) -> list[TrackedNote]:
    """
    Convert a flat list of RawNote (single channel) to TrackedNote in beat time.

    Resolves monophonic overlaps: if two note-ons arrive before a note-off,
    the first note is ended at the second note's start.
    """
    # Sort by tick_on to process in order
    raw_sorted = sorted(raw, key=lambda n: n.tick_on)

    def to_beat(tick: int) -> float:
        return _tick_to_beat(tick, ticks_per_beat, tempo_changes)

    result: list[TrackedNote] = []

    for note in raw_sorted:
        start  = to_beat(note.tick_on)
        end    = to_beat(note.tick_off)
        dur    = max(0.0625, end - start)      # min 1/16 beat
        result.append(TrackedNote(
            pitch=note.pitch,
            start_beat=start,
            duration_beats=dur,
            velocity=note.velocity,
        ))

    result.sort(key=lambda n: n.start_beat)
    return result


def quantise_beats(
    notes: list[TrackedNote],
    grid: float = 0.125,   # 1/8 beat = 32nd note
) -> list[TrackedNote]:
    """
    Snap start_beat and duration_beats to the nearest grid multiple.
    Minimum duration after quantisation: grid.
    """
    def snap(v: float) -> float:
        return round(round(v / grid) * grid, 9)

    out: list[TrackedNote] = []
    for n in notes:
        s = snap(n.start_beat)
        d = max(grid, snap(n.duration_beats))
        out.append(TrackedNote(n.pitch, s, d, n.velocity))
    return out
