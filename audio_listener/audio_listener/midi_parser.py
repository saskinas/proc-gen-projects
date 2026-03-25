"""
audio_listener.midi_parser — Pure-Python Standard MIDI File (SMF) reader.

Supports Format 0 (single track) and Format 1 (multi-track).
Outputs a MidiData object with all note events, tempo changes, and
time-signature changes needed for downstream music analysis.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class RawNote:
    """One sounding note extracted from a MIDI track."""
    channel:    int    # 0-indexed (channel 9 = GM drums)
    pitch:      int    # MIDI note number 0-127
    velocity:   int    # 1-127
    tick_on:    int    # absolute tick of note-on
    tick_off:   int    # absolute tick of note-off

    @property
    def duration_ticks(self) -> int:
        return max(1, self.tick_off - self.tick_on)


@dataclass
class MidiData:
    """Parsed representation of a Standard MIDI File."""
    format:           int                            # 0 or 1
    ticks_per_beat:   int                            # PPQ (pulses per quarter note)
    tempo_changes:    list[tuple[int, int]]          # (abs_tick, microseconds_per_beat)
    time_sig_events:  list[tuple[int, int, int]]     # (abs_tick, numerator, denominator)
    notes:            list[RawNote]                  # all notes, all channels
    track_names:      dict[int, str]                 # channel → track name (from meta)
    total_ticks:      int
    program_numbers:  dict[int, int] = field(        # channel → first GM program number (0-indexed)
        default_factory=dict
    )

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def tempo_bpm(self) -> int:
        if self.tempo_changes:
            uspb = self.tempo_changes[0][1]
            return max(1, int(60_000_000 / uspb))
        return 120

    @property
    def time_signature(self) -> tuple[int, int]:
        if self.time_sig_events:
            return (self.time_sig_events[0][1], self.time_sig_events[0][2])
        return (4, 4)

    def tick_to_beat(self, tick: int) -> float:
        """Convert absolute tick to fractional beat, accounting for tempo changes."""
        if len(self.tempo_changes) <= 1:
            return tick / self.ticks_per_beat
        beat      = 0.0
        prev_tick = 0
        for chg_tick, _uspb in self.tempo_changes:
            if tick <= chg_tick:
                break
            beat     += (chg_tick - prev_tick) / self.ticks_per_beat
            prev_tick = chg_tick
        beat += (tick - prev_tick) / self.ticks_per_beat
        return beat

    def notes_for_channel(self, channel: int) -> list[RawNote]:
        return [n for n in self.notes if n.channel == channel]

    def channels(self) -> set[int]:
        return {n.channel for n in self.notes}


# ── Parser ────────────────────────────────────────────────────────────────────

def _read_varlen(data: bytes, pos: int) -> tuple[int, int]:
    """Read a MIDI variable-length quantity. Returns (value, new_pos)."""
    value = 0
    while pos < len(data):
        byte   = data[pos]; pos += 1
        value  = (value << 7) | (byte & 0x7F)
        if not (byte & 0x80):
            break
    return value, pos


def _parse_track(data: bytes) -> tuple[list[RawNote], list, list, dict[int, str], dict[int, int], int]:
    """
    Parse one MTrk chunk.

    Returns:
        notes           : list[RawNote]
        tempo_changes   : list[(abs_tick, uspb)]
        time_sig_events : list[(abs_tick, num, den)]
        track_names     : dict[channel, name]  (from TrackName / InstrumentName meta)
        program_numbers : dict[channel, int]   (most common program change per channel)
        last_tick       : absolute tick of last event
    """
    notes: list[RawNote]           = []
    tempo_changes: list            = []
    time_sig_events: list          = []
    track_names: dict[int, str]    = {}
    program_numbers: dict[int, int] = {}       # channel → most common GM program
    _prog_counts: dict[int, dict[int, int]] = {}  # channel → {prog → count}

    # open_notes[(channel, pitch)] = list of (tick_on, velocity)
    open_notes: dict[tuple[int, int], list[tuple[int, int]]] = {}

    abs_tick       = 0
    running_status = 0
    i              = 0
    last_channel   = 0

    while i < len(data):
        delta, i  = _read_varlen(data, i)
        abs_tick += delta

        if i >= len(data):
            break

        byte = data[i]

        # Running status: if high bit is 0, reuse last status
        if byte & 0x80:
            running_status = byte
            i += 1
        else:
            byte = running_status
            # don't advance i — the byte IS a data byte

        status_type = byte & 0xF0
        channel     = byte & 0x0F
        last_channel = channel

        # ── Note events ───────────────────────────────────────────────────────

        if status_type == 0x90:                      # Note On
            pitch = data[i]; i += 1
            vel   = data[i]; i += 1
            if vel > 0:
                open_notes.setdefault((channel, pitch), []).append((abs_tick, vel))
            else:
                # Note On vel=0 = Note Off
                stack = open_notes.get((channel, pitch), [])
                if stack:
                    t_on, v_on = stack.pop(0)
                    dur = max(1, abs_tick - t_on)
                    notes.append(RawNote(channel, pitch, v_on, t_on, t_on + dur))

        elif status_type == 0x80:                    # Note Off
            pitch  = data[i]; i += 1
            _vel   = data[i]; i += 1
            stack  = open_notes.get((channel, pitch), [])
            if stack:
                t_on, v_on = stack.pop(0)
                dur = max(1, abs_tick - t_on)
                notes.append(RawNote(channel, pitch, v_on, t_on, t_on + dur))

        # ── Other channel events ──────────────────────────────────────────────

        elif status_type == 0xA0:   i += 2          # Poly aftertouch
        elif status_type == 0xB0:   i += 2          # Control change
        elif status_type == 0xC0:                   # Program change
            prog = data[i]; i += 1
            if channel not in _prog_counts:
                _prog_counts[channel] = {}
            _prog_counts[channel][prog] = _prog_counts[channel].get(prog, 0) + 1
        elif status_type == 0xD0:   i += 1          # Channel pressure
        elif status_type == 0xE0:   i += 2          # Pitch bend

        # ── Meta events ───────────────────────────────────────────────────────

        elif byte == 0xFF:
            meta_type = data[i]; i += 1
            meta_len, i = _read_varlen(data, i)
            meta_data   = data[i:i + meta_len]; i += meta_len

            if meta_type == 0x51 and meta_len >= 3:
                uspb = (meta_data[0] << 16) | (meta_data[1] << 8) | meta_data[2]
                tempo_changes.append((abs_tick, uspb))

            elif meta_type == 0x58 and meta_len >= 2:
                num = meta_data[0]
                den = 2 ** meta_data[1]
                time_sig_events.append((abs_tick, num, den))

            elif meta_type in (0x03, 0x04):         # Track name / Instrument name
                name = meta_data.decode("latin-1", errors="replace").strip()
                if name:
                    track_names[last_channel] = name

            elif meta_type == 0x2F:                  # End of track
                break

        # ── SysEx ─────────────────────────────────────────────────────────────

        elif byte in (0xF0, 0xF7):
            slen, i = _read_varlen(data, i)
            i += slen

        else:
            # Unknown: skip one byte and hope running status recovers
            i += 1

    # Close any notes still open at end of track
    for (ch, pitch), stack in open_notes.items():
        for (t_on, v_on) in stack:
            dur = max(1, abs_tick - t_on)
            notes.append(RawNote(ch, pitch, v_on, t_on, t_on + dur))

    # Resolve most common program per channel
    for ch, counts in _prog_counts.items():
        program_numbers[ch] = max(counts, key=counts.get)

    return notes, tempo_changes, time_sig_events, track_names, program_numbers, abs_tick


def parse_midi(data: bytes) -> MidiData:
    """
    Parse raw Standard MIDI File bytes into a MidiData object.

    Supports Format 0 (single track, all channels mixed) and Format 1
    (multi-track; track 0 is typically the tempo/meta track).
    """
    if data[:4] != b"MThd":
        raise ValueError("Not a Standard MIDI File (missing MThd header)")

    hdr_len = struct.unpack(">I", data[4:8])[0]
    fmt, num_tracks, tpb = struct.unpack(">HHH", data[8:14])

    if tpb & 0x8000:
        raise ValueError("SMPTE-timecode MIDI not supported; only PPQ ticks")

    # Collect raw track chunks
    pos = 8 + hdr_len
    track_chunks: list[bytes] = []
    for _ in range(num_tracks):
        if data[pos:pos+4] != b"MTrk":
            break
        trk_len = struct.unpack(">I", data[pos+4:pos+8])[0]
        track_chunks.append(data[pos+8 : pos+8+trk_len])
        pos += 8 + trk_len

    all_notes: list[RawNote]    = []
    all_tempo: list             = []
    all_tsig:  list             = []
    all_names: dict[int, str]   = {}
    all_progs: dict[int, int]   = {}
    max_tick   = 0

    for trk in track_chunks:
        nts, tmp, tsig, names, progs, last = _parse_track(trk)
        all_notes.extend(nts)
        all_tempo.extend(tmp)
        all_tsig.extend(tsig)
        all_names.update(names)
        # Keep the first program seen per channel across all tracks
        for ch, prog in progs.items():
            if ch not in all_progs:
                all_progs[ch] = prog
        max_tick = max(max_tick, last)

    # Ensure at least a default tempo
    if not all_tempo:
        all_tempo = [(0, 500_000)]   # 120 BPM

    all_tempo.sort(key=lambda x: x[0])
    all_tsig.sort(key=lambda x: x[0])

    # total_ticks: max of last event across all channels
    if all_notes:
        max_tick = max(max_tick, max(n.tick_off for n in all_notes))

    return MidiData(
        format=fmt,
        ticks_per_beat=tpb,
        tempo_changes=all_tempo,
        time_sig_events=all_tsig,
        notes=all_notes,
        track_names=all_names,
        total_ticks=max_tick,
        program_numbers=all_progs,
    )
