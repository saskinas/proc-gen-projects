"""
audio_listener.channel_router — NES MIDI channel role assignment.

Maps MIDI channels to semantic roles: melody, countermelody, bass, drums.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from audio_listener.midi_parser import MidiData, RawNote


@dataclass
class ChannelAssignment:
    melody:        int | None        # MIDI channel for primary melody
    countermelody: int | None        # secondary melodic channel (may be None)
    bass:          int | None        # bass channel (lowest register)
    drums:         int | None        # percussion channel (9 or noise)
    ignored:       list[int]         # channels excluded from analysis


def assign_channels(data: MidiData) -> ChannelAssignment:
    """
    Assign NES channel roles heuristically.

    Priority:
    1. Channel 9 (GM drums) → drums.
    2. Any channel whose track name contains 'noise', 'drum', 'perc' → drums.
    3. Among remaining, any channel named 'triangle', 'bass' → bass.
    4. Remaining channels sorted by mean pitch:
       - Lowest mean → bass (if not already assigned)
       - Highest mean → melody
       - Middle → countermelody
    5. Channels with fewer than 4 notes → ignored.
    """
    channels = data.channels()
    notes_by_ch: dict[int, list[RawNote]] = {
        ch: data.notes_for_channel(ch) for ch in channels
    }

    # Filter out sparse channels
    active = {ch for ch, nts in notes_by_ch.items() if len(nts) >= 4}

    drums_ch: int | None        = None
    bass_ch:  int | None        = None
    melody_ch: int | None       = None
    counter_ch: int | None      = None
    ignored: list[int]          = [ch for ch in channels if ch not in active]

    # Step 1 & 2: find drums channel
    if 9 in active:
        drums_ch = 9
        active.discard(9)
    else:
        for ch in list(active):
            name = data.track_names.get(ch, "").lower()
            if any(k in name for k in ("noise", "drum", "perc", "dpcm", "pcm", "noi")):
                drums_ch = ch
                active.discard(ch)
                break

    # Step 3: name-based bass detection
    for ch in list(active):
        name = data.track_names.get(ch, "").lower()
        if any(k in name for k in ("triangle", "bass", "tri")):
            bass_ch = ch
            active.discard(ch)
            break

    if not active:
        return ChannelAssignment(melody_ch, counter_ch, bass_ch, drums_ch, ignored)

    # Step 4: sort remaining by mean pitch
    def _mean_pitch(ch: int) -> float:
        pitches = [n.pitch for n in notes_by_ch[ch]]
        return sum(pitches) / len(pitches) if pitches else 0.0

    def _note_count(ch: int) -> int:
        return len(notes_by_ch[ch])

    sorted_ch = sorted(active, key=_mean_pitch)  # ascending pitch

    if bass_ch is None and sorted_ch:
        bass_ch = sorted_ch.pop(0)          # lowest → bass

    # Filter sparse channels out of the melody/counter pool.
    # Channels with very few notes relative to the busiest channel should not
    # be selected as melody even if they happen to have the highest mean pitch.
    if sorted_ch:
        max_pool = max(_note_count(ch) for ch in sorted_ch)
        sparse_cutoff = max(8, int(max_pool * 0.08))
        substantial = [ch for ch in sorted_ch if _note_count(ch) >= sparse_cutoff]
        if substantial:
            # Move sparse channels to ignored
            for ch in sorted_ch:
                if ch not in substantial:
                    ignored.append(ch)
            sorted_ch = substantial

    # Score remaining channels: weight note count and pitch, but penalise pure
    # arpeggio channels (strictly alternating intervals) since they carry chord
    # tones, not the melodic theme. A melody channel should have varied contour.
    def _melodic_variety(ch: int) -> float:
        """1.0 = varied melody, <0.5 = mostly alternating arpeggio."""
        raw_notes = sorted(notes_by_ch[ch], key=lambda n: n.tick_on)
        pitches = [n.pitch for n in raw_notes]
        if len(pitches) < 4:
            return 1.0
        intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches) - 1)]
        reversals = sum(
            1 for i in range(len(intervals) - 1)
            if intervals[i] == -intervals[i + 1]
        )
        arp_fraction = reversals / max(1, len(intervals) - 1)
        return 1.0 - arp_fraction * 0.8   # max 80% penalty for pure arpeggios

    def _mean_duration_beats(ch: int, ticks_per_beat: int) -> float:
        """Mean note duration in beats for this channel's raw notes."""
        raw = notes_by_ch[ch]
        if not raw or ticks_per_beat <= 0:
            return 1.0
        durations = [max(0, n.tick_off - n.tick_on) for n in raw]
        mean_ticks = sum(durations) / len(durations)
        return mean_ticks / ticks_per_beat

    tpb = data.ticks_per_beat if data.ticks_per_beat > 0 else 480

    def _selection_score(ch: int) -> float:
        count = _note_count(ch)
        pitch = _mean_pitch(ch)
        pitch_norm = max(0.0, min(1.0, (pitch - 36) / 60.0))
        variety   = _melodic_variety(ch)
        mean_dur  = _mean_duration_beats(ch, tpb)
        # Melody notes are long (1-2 beats); arpeggio notes are short (0.25-0.5 beats).
        # Weight by mean duration so arpeggio channels don't dominate via note count.
        return count * mean_dur * (0.2 + 0.8 * pitch_norm) * variety

    if sorted_ch:
        sorted_ch = sorted(sorted_ch, key=_selection_score)  # ascending score

    if sorted_ch:
        melody_ch = sorted_ch.pop()         # highest composite score → melody

    if sorted_ch:
        counter_ch = sorted_ch.pop()        # next highest score → countermelody

    # Anything left over
    ignored.extend(sorted_ch)

    # Sanity: if bass has higher mean pitch than melody, swap
    if melody_ch is not None and bass_ch is not None:
        if _mean_pitch(bass_ch) > _mean_pitch(melody_ch):
            melody_ch, bass_ch = bass_ch, melody_ch

    # Sanity: if counter has higher mean pitch than melody, swap (melody = upper voice)
    if melody_ch is not None and counter_ch is not None:
        if _mean_pitch(counter_ch) > _mean_pitch(melody_ch):
            melody_ch, counter_ch = counter_ch, melody_ch

    # Sanity: if counter has lower mean pitch than bass, swap
    if bass_ch is not None and counter_ch is not None:
        if _mean_pitch(counter_ch) < _mean_pitch(bass_ch):
            counter_ch, bass_ch = bass_ch, counter_ch

    return ChannelAssignment(
        melody=melody_ch,
        countermelody=counter_ch,
        bass=bass_ch,
        drums=drums_ch,
        ignored=ignored,
    )
