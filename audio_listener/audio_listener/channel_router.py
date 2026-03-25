"""
audio_listener.channel_router — MIDI channel role assignment.

Maps MIDI channels to semantic roles: melody, countermelody, bass, drums.
Supports both global (whole-file) and per-section assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from audio_listener.midi_parser import MidiData, RawNote
from audio_listener.pitch_tracker import TrackedNote
from audio_listener.phrase_segmenter import SectionBoundary


@dataclass
class ChannelAssignment:
    melody:        int | None        # MIDI channel for primary melody
    countermelody: int | None        # secondary melodic channel (may be None)
    bass:          int | None        # bass channel (lowest register)
    drums:         int | None        # percussion channel (9 or noise)
    ignored:       list[int]         # channels excluded from analysis (sparse)
    inner:         list[int]         = field(default_factory=list)  # all other active channels


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
    sparse_inner: list[int] = []
    if sorted_ch:
        max_pool = max(_note_count(ch) for ch in sorted_ch)
        sparse_cutoff = max(8, int(max_pool * 0.08))
        substantial = [ch for ch in sorted_ch if _note_count(ch) >= sparse_cutoff]
        if substantial:
            # Sparse channels are excluded from melody/counter selection but still
            # captured as inner voices — they may dominate specific sections.
            sparse_inner = [ch for ch in sorted_ch if ch not in substantial]
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

    # All remaining substantial channels plus section-local sparse channels become
    # inner voices and are captured in voice_sequences.
    inner_voices = list(sorted_ch) + sparse_inner

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
        inner=inner_voices,
    )


# ── Per-section channel assignment ────────────────────────────────────────────

@dataclass
class SectionChannelAssignment:
    """Per-section channel role assignment."""
    section_idx:    int
    start_beat:     float
    end_beat:       float
    melody:         int | None
    countermelody:  int | None
    bass:           int | None
    drums:          int | None
    inner:          list[int] = field(default_factory=list)


def _score_tracked_notes(notes: list[TrackedNote]) -> float:
    """
    Score a list of TrackedNotes for melody-ness.

    Uses the same composite metric as assign_channels():
      note_count * mean_duration * pitch_norm * melodic_variety
    """
    if not notes:
        return 0.0

    count = len(notes)

    # Mean duration
    mean_dur = sum(n.duration_beats for n in notes) / count

    # Pitch normalisation (same 36-96 range as assign_channels)
    mean_pitch = sum(n.pitch for n in notes) / count
    pitch_norm = max(0.0, min(1.0, (mean_pitch - 36) / 60.0))

    # Melodic variety (penalise strict alternating arpeggios)
    sorted_notes = sorted(notes, key=lambda n: n.start_beat)
    pitches = [n.pitch for n in sorted_notes]
    if len(pitches) < 4:
        variety = 1.0
    else:
        intervals = [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]
        reversals = sum(
            1 for i in range(len(intervals) - 1)
            if intervals[i] == -intervals[i + 1]
        )
        arp_fraction = reversals / max(1, len(intervals) - 1)
        variety = 1.0 - arp_fraction * 0.8

    # Cap mean_dur so sustained drones don't dominate over rhythmic melodies.
    # Add unique-pitch bonus so channels with distinct melodic content score higher.
    capped_dur     = min(mean_dur, 2.0)
    unique_pitches = len(set(n.pitch for n in notes))
    unique_bonus   = min(1.5, max(0.7, unique_pitches / 7.0))
    return count * capped_dur * (0.2 + 0.8 * pitch_norm) * variety * unique_bonus


def assign_channels_per_section(
    channel_notes: dict[int, list[TrackedNote]],
    global_assignment: ChannelAssignment,
    sections: list[SectionBoundary],
) -> list[SectionChannelAssignment]:
    """
    Re-score channel roles for each section independently.

    Parameters
    ----------
    channel_notes
        channel number → beat-quantised TrackedNote list (all non-drum channels).
    global_assignment
        The whole-file ChannelAssignment (used for drums, which stay fixed).
    sections
        Section boundaries from phrase segmentation.

    Returns
    -------
    list[SectionChannelAssignment]
        One entry per section, in order.
    """
    drums_ch = global_assignment.drums
    result: list[SectionChannelAssignment] = []

    for sec in sections:
        s, e = sec.start_beat, sec.end_beat

        # Filter each channel's notes to this section's time window
        windowed: dict[int, list[TrackedNote]] = {}
        for ch, notes in channel_notes.items():
            w = [n for n in notes if s <= n.start_beat < e]
            if len(w) >= 2:          # require at least 2 notes in window
                windowed[ch] = w

        if not windowed:
            # No pitched notes in this section — fall back to global
            result.append(SectionChannelAssignment(
                section_idx=sec.section_idx,
                start_beat=s, end_beat=e,
                melody=global_assignment.melody,
                countermelody=global_assignment.countermelody,
                bass=global_assignment.bass,
                drums=drums_ch,
                inner=list(global_assignment.inner),
            ))
            continue

        # Score each channel for this section
        scores: dict[int, float] = {
            ch: _score_tracked_notes(notes) for ch, notes in windowed.items()
        }
        mean_pitch: dict[int, float] = {
            ch: sum(n.pitch for n in notes) / len(notes)
            for ch, notes in windowed.items()
        }

        # Sort ascending by score
        ranked = sorted(scores.keys(), key=lambda ch: scores[ch])

        # Assign bass = lowest mean pitch among active channels
        bass_candidates = sorted(windowed.keys(), key=lambda ch: mean_pitch[ch])
        bass = bass_candidates[0]
        ranked = [ch for ch in ranked if ch != bass]

        # Melody = highest score among remaining
        melody = ranked.pop() if ranked else None

        # Counter = next highest score
        counter = ranked.pop() if ranked else None

        # Sanity: melody should have higher mean pitch than bass
        if melody is not None and mean_pitch.get(melody, 0) < mean_pitch.get(bass, 0):
            melody, bass = bass, melody

        # Sanity: melody should have higher mean pitch than counter
        if melody is not None and counter is not None:
            if mean_pitch.get(counter, 0) > mean_pitch.get(melody, 0):
                melody, counter = counter, melody

        # Sanity: counter should have higher mean pitch than bass
        if counter is not None and mean_pitch.get(counter, 0) < mean_pitch.get(bass, 0):
            counter, bass = bass, counter

        inner = list(ranked)

        result.append(SectionChannelAssignment(
            section_idx=sec.section_idx,
            start_beat=s, end_beat=e,
            melody=melody,
            countermelody=counter,
            bass=bass,
            drums=drums_ch,
            inner=inner,
        ))

    return result
