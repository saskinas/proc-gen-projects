"""
audio_listener.tempo_meter — Tempo BPM and time signature detection.
"""

from __future__ import annotations

from audio_listener.midi_parser import MidiData
from audio_listener.pitch_tracker import TrackedNote


def detect_tempo(data: MidiData) -> int:
    """
    Return tempo in BPM from MIDI tempo meta events.
    Clamps to [40, 240]. Defaults to 120 if no meta event.

    Note: Without a tempo meta event, BPM cannot be determined from
    note onset intervals alone (ticks are relative, not absolute time).
    """
    meta_bpm = data.tempo_bpm  # 120 if no meta event
    return max(40, min(240, meta_bpm))


def detect_tempo_map(data: MidiData) -> list[tuple[float, int]]:
    """
    Return all tempo changes as (beat_position, bpm) pairs.

    Empty list if there's only one tempo (no mid-piece changes).
    When there are multiple tempo changes, includes the initial tempo at beat 0.
    """
    if len(data.tempo_changes) <= 1:
        return []

    result = []
    for tick, uspb in data.tempo_changes:
        beat = data.tick_to_beat(tick)
        bpm = max(40, min(240, int(60_000_000 / uspb)))
        result.append((beat, bpm))

    return result


def detect_time_signature(data: MidiData) -> list[int]:
    """
    Return [numerator, denominator] of the prevailing time signature.
    Primary source: MIDI time signature meta events.
    Default: [4, 4].
    """
    if data.time_sig_events:
        _, num, den = data.time_sig_events[0]
        # Sanity-check: only accept common time signatures
        if num in (2, 3, 4, 6, 9, 12) and den in (2, 4, 8, 16):
            return [num, den]
    return [4, 4]
