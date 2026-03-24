"""
procgen.export — Export generated content to standard formats.

Exporters:
    to_json(obj)            → JSON string (any domain output)
    to_midi(score)          → bytes (Standard MIDI File, Type 0)
    to_svg_dungeon(dungeon) → SVG string (visual dungeon map)
    to_svg_terrain(terrain) → SVG string (colour terrain map)
    to_markdown(outline)    → Markdown string (narrative outline)
"""

from __future__ import annotations
import json
import struct
import math
from typing import Any


# ---------------------------------------------------------------------------
# Generic JSON
# ---------------------------------------------------------------------------

def to_json(obj: Any, indent: int = 2) -> str:
    """
    Serialise any procgen output to JSON.
    Handles dataclasses, nested lists, and primitive types.
    """
    def _default(o):
        if hasattr(o, "__dataclass_fields__"):
            d = {}
            for f in o.__dataclass_fields__:
                d[f] = getattr(o, f)
            return d
        if isinstance(o, tuple):
            return list(o)
        raise TypeError(f"Cannot serialise {type(o)}")
    return json.dumps(obj, default=_default, indent=indent)


# ---------------------------------------------------------------------------
# MIDI export
# ---------------------------------------------------------------------------

_NOTE_MIDI: dict[str, int] = {}
_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
for _oct in range(0, 9):
    for _i, _n in enumerate(_NOTES):
        _NOTE_MIDI[f"{_n}{_oct}"] = (_oct + 1) * 12 + _i

# GM drum map: voice name -> MIDI note number (channel 10 / index 9)
_DRUM_MIDI: dict[str, int] = {
    "kick":           36,
    "snare":          38,
    "snare_rim":      37,
    "snare_electric": 40,
    "hihat_closed":   42,
    "hihat_pedal":    44,
    "hihat_open":     46,
    "tom_hi":         50,
    "tom_hi_mid":     48,
    "tom_lo_mid":     47,
    "tom_lo":         45,
    "tom_floor":      43,
    "crash":          49,
    "crash2":         57,
    "ride":           51,
    "ride_bell":      53,
    "cowbell":        56,
}


def _var_len(value: int) -> bytes:
    """Encode an integer as MIDI variable-length quantity."""
    result = [value & 0x7F]
    value >>= 7
    while value:
        result.insert(0, (value & 0x7F) | 0x80)
        value >>= 7
    return bytes(result)


def _midi_tempo(bpm: int) -> bytes:
    """Convert BPM to MIDI microseconds-per-beat (3 bytes)."""
    uspb = int(60_000_000 / bpm)
    return bytes([(uspb >> 16) & 0xFF, (uspb >> 8) & 0xFF, uspb & 0xFF])


def to_midi(score: Any, ticks_per_beat: int = 480) -> bytes:
    """
    Convert a MusicScore to a Standard MIDI File (Type 1).

    Returns raw bytes that can be written to a .mid file.

    Args:
        score           — MusicScore instance
        ticks_per_beat  — MIDI timing resolution

    Example::
        data = to_midi(score)
        with open("output.mid", "wb") as f:
            f.write(data)
    """
    tracks_data = []

    # --- Tempo track (track 0) ---
    tempo_track = bytearray()
    # Set tempo
    tempo_track += _var_len(0)          # delta time 0
    tempo_track += b"\xFF\x51\x03"      # Meta: set tempo
    tempo_track += _midi_tempo(score.tempo_bpm)
    # Time signature
    num, den = score.time_signature
    log2_den = int(math.log2(den))
    tempo_track += _var_len(0)
    tempo_track += b"\xFF\x58\x04"      # Meta: time sig
    tempo_track += bytes([num, log2_den, 24, 8])
    # End of track
    tempo_track += _var_len(0) + b"\xFF\x2F\x00"
    tracks_data.append(bytes(tempo_track))

    # --- Instrument tracks ---
    _PROGRAM: dict[str, int] = {
        "piano": 0, "harpsichord": 6, "bass": 32, "drums": 0,
        "strings": 48, "guitar": 25, "violin": 40, "viola": 41,
        "cello": 42, "contrabass": 43, "flute": 73, "organ": 16,
        "lute": 24, "oboe": 68, "clarinet": 71, "bassoon": 70,
        "trumpet": 56, "horn": 60,
        "electric_guitar": 29, "dist_guitar": 30, "bass_guitar": 33,
        # Synth / NES-style voices
        "square": 80,    "square_wave": 80,   # Square Lead (closest to NES pulse)
        "sawtooth": 81,  "saw_wave":    81,   # Sawtooth Lead
        "triangle": 65,  "triangle_wave": 65, # Alto Sax (bright hollow, ~triangle)
        "synth_lead": 80, "synth_bass": 38,   # Synth Bass 1
        "chiptune": 80,  "nes_pulse": 80,
    }
    # MIDI has 16 channels (0-15); channel 9 is reserved for drums.
    # Assign each non-drum track its own unique channel, skipping 9.
    _PITCHED_CHANNELS = [c for c in range(16) if c != 9]  # 15 available channels
    _pitched_idx = 0
    _channel_map: dict[int, int] = {}
    for idx, t in enumerate(score.tracks):
        if t.instrument == "drums":
            _channel_map[idx] = 9
        else:
            _channel_map[idx] = _PITCHED_CHANNELS[_pitched_idx % len(_PITCHED_CHANNELS)]
            _pitched_idx += 1

    for track_idx, track in enumerate(score.tracks):
        channel = _channel_map[track_idx]
        instr = track.instrument
        if instr.startswith("gm_prog_"):
            try:
                program = int(instr[8:]) & 0x7F
            except ValueError:
                program = 0
        else:
            program = _PROGRAM.get(instr, 0)

        track_bytes = bytearray()

        # Program change
        track_bytes += _var_len(0)
        track_bytes += bytes([0xC0 | channel, program])

        # Convert notes to events (absolute tick)
        events = []
        abs_tick = 0
        if channel == 9:
            # Drum track: rest-encoded timing
            # Note("rest", gap, 0)      -> advance clock only
            # Note(voice, 0.0, vel)     -> drum hit at current position
            for note in track.notes:
                dur_ticks = int(note.duration * ticks_per_beat)
                if note.velocity == 0:
                    abs_tick += dur_ticks
                else:
                    midi_note = _DRUM_MIDI.get(note.pitch, None)
                    if midi_note is None:
                        try:
                            midi_note = int(note.pitch)  # numeric MIDI note stored as string
                        except (ValueError, TypeError):
                            midi_note = _NOTE_MIDI.get(note.pitch, 36)
                    events.append((abs_tick, "note_on", channel, midi_note, note.velocity))
                    off_tick = abs_tick + max(1, ticks_per_beat // 8)
                    events.append((off_tick, "note_off", channel, midi_note, 0))
                    abs_tick += dur_ticks
        else:
            for note in track.notes:
                dur_ticks = int(note.duration * ticks_per_beat)
                if note.pitch.startswith("__pc__"):
                    instr_name = note.pitch[6:]
                    if instr_name.startswith("gm_prog_"):
                        try:
                            pc_prog = int(instr_name[8:]) & 0x7F
                        except ValueError:
                            pc_prog = 0
                    else:
                        pc_prog = _PROGRAM.get(instr_name, 0)
                    events.append((abs_tick, "program_change", channel, pc_prog, 0))
                    abs_tick += dur_ticks
                    continue
                if note.pitch == "rest":
                    abs_tick += dur_ticks
                    continue
                midi_note = _NOTE_MIDI.get(note.pitch, 60)
                events.append((abs_tick, "note_on",  channel, midi_note, note.velocity))
                events.append((abs_tick + dur_ticks, "note_off", channel, midi_note, 0))
                abs_tick += dur_ticks

        # Sort by time (note_off before note_on at same tick)
        events.sort(key=lambda e: (e[0], {"note_off": 0, "program_change": 1, "note_on": 2}.get(e[1], 2)))

        prev_tick = 0
        for abs_tick, kind, ch, pitch, vel in events:
            delta = abs_tick - prev_tick
            prev_tick = abs_tick
            track_bytes += _var_len(delta)
            if kind == "program_change":
                track_bytes += bytes([0xC0 | ch, pitch & 0x7F])
            else:
                status = (0x90 | ch) if kind == "note_on" else (0x80 | ch)
                track_bytes += bytes([status, pitch & 0x7F, vel & 0x7F])

        # End of track
        track_bytes += _var_len(0) + b"\xFF\x2F\x00"
        tracks_data.append(bytes(track_bytes))

    # --- Assemble SMF ---
    num_tracks = len(tracks_data)
    header = struct.pack(">4sIHHH", b"MThd", 6, 1, num_tracks, ticks_per_beat)
    body = bytearray()
    for td in tracks_data:
        body += struct.pack(">4sI", b"MTrk", len(td))
        body += td

    return header + bytes(body)


# ---------------------------------------------------------------------------
# SVG dungeon map
# ---------------------------------------------------------------------------

def to_svg_dungeon(dungeon: Any, cell_size: int = 60) -> str:
    """
    Render a DungeonMap as an SVG string.

    Each room is drawn as a rectangle with connections as lines.
    """
    rooms = dungeon.rooms
    n = len(rooms)

    # Lay out rooms in a rough grid
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    padding = cell_size
    room_w = cell_size * 2
    room_h = cell_size
    gap_x = cell_size * 3
    gap_y = cell_size * 2

    svg_w = cols * gap_x + padding * 2
    svg_h = rows * gap_y + padding * 2

    def room_centre(room_id: int):
        col = room_id % cols
        row = room_id // cols
        cx = padding + col * gap_x + room_w // 2
        cy = padding + row * gap_y + room_h // 2
        return cx, cy

    def room_rect(room_id: int):
        cx, cy = room_centre(room_id)
        return cx - room_w // 2, cy - room_h // 2

    _TYPE_COLORS = {
        "start":    "#4a9e6b",
        "boss":     "#c0392b",
        "exit":     "#8e44ad",
        "treasure": "#d4ac0d",
        "puzzle":   "#2980b9",
        "enemy":    "#7f8c8d",
    }

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" '
        f'style="background:#1a1a2e;font-family:monospace">',
        '<defs><marker id="arr" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">'
        '<path d="M0,0 L6,3 L0,6 Z" fill="#555"/></marker></defs>',
    ]

    # Draw connections
    drawn = set()
    for room in rooms:
        for conn_id in room.connections:
            pair = tuple(sorted([room.id, conn_id]))
            if pair in drawn:
                continue
            drawn.add(pair)
            x1, y1 = room_centre(room.id)
            x2, y2 = room_centre(conn_id)
            lines.append(
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                f'stroke="#444" stroke-width="2" stroke-dasharray="4 2"/>'
            )

    # Draw rooms
    for room in rooms:
        rx, ry = room_rect(room.id)
        color = _TYPE_COLORS.get(room.room_type, "#555")
        border = "#fff" if room.id in (dungeon.start_room, dungeon.boss_room, dungeon.exit_room) else "#888"
        bw = 2 if border == "#888" else 3

        lines.append(
            f'<rect x="{rx}" y="{ry}" width="{room_w}" height="{room_h}" '
            f'rx="4" fill="{color}" stroke="{border}" stroke-width="{bw}"/>'
        )

        cx, cy = room_centre(room.id)

        # Room ID
        lines.append(f'<text x="{cx}" y="{ry + 14}" text-anchor="middle" '
                     f'fill="#fff" font-size="10" font-weight="bold">#{room.id}</text>')

        # Room name (truncated)
        name = room.name[:14] + "…" if len(room.name) > 14 else room.name
        lines.append(f'<text x="{cx}" y="{ry + 28}" text-anchor="middle" '
                     f'fill="#ddd" font-size="9">{name}</text>')

        # Enemy / loot counts
        info = []
        if room.enemies: info.append(f"⚔{len(room.enemies)}")
        if room.loot:    info.append(f"💎{len(room.loot)}")
        if info:
            lines.append(f'<text x="{cx}" y="{ry + 44}" text-anchor="middle" '
                         f'fill="#aaa" font-size="9">{" ".join(info)}</text>')

    lines.append("</svg>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SVG terrain map
# ---------------------------------------------------------------------------

def to_svg_terrain(terrain: Any, cell_px: int = 12) -> str:
    """Render a TerrainMap as a coloured SVG grid."""
    _TILE_COLORS = {
        # terrain
        "ocean": "#1a6b9a", "beach": "#e8d5a0", "plains": "#7fc97f",
        "forest": "#2d6a2d", "mountain": "#8a8a8a",
        # arctic
        "ice_ocean": "#2255aa", "ice_shelf": "#aaccee", "tundra": "#ccddcc", "glacier": "#eaf4f4",
        # desert
        "dune": "#d4a64a", "rocky": "#9c7a4e", "mesa": "#c0522a", "oasis": "#5aab5a",
        # volcanic
        "sea": "#1a4080", "shore": "#8a7055", "ash": "#888", "lava": "#cc3300", "obsidian": "#222",
    }

    w = terrain.width * cell_px
    h = terrain.height * cell_px

    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">']

    for y, row in enumerate(terrain.tile_grid):
        for x, tile in enumerate(row):
            color = _TILE_COLORS.get(tile or "", "#333")
            lines.append(
                f'<rect x="{x * cell_px}" y="{y * cell_px}" '
                f'width="{cell_px}" height="{cell_px}" fill="{color}"/>'
            )

    lines.append("</svg>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown narrative
# ---------------------------------------------------------------------------

def to_markdown(outline: Any) -> str:
    """Render a NarrativeOutline as a Markdown document."""
    lines = [
        f"# {outline.title}",
        f"",
        f"**Genre:** {outline.genre}  |  **Tone:** {outline.tone}  |  "
        f"**Protagonist:** {outline.protagonist_archetype}",
        f"",
        f"**MacGuffin:** {outline.MacGuffin}",
        f"**Antagonist:** {outline.antagonist}",
        f"",
        "---",
        "",
    ]
    for act in outline.acts:
        lines.append(f"## Act {act.number}: {act.name}")
        lines.append("")
        for beat in act.beats:
            tension_bar = "█" * int(beat.tension * 10) + "░" * (10 - int(beat.tension * 10))
            lines.append(f"### {beat.name}")
            lines.append(f"*Tension:* `{tension_bar}` {beat.tension:.0%}")
            lines.append("")
            lines.append(beat.description)
            lines.append("")
    return "\n".join(lines)
