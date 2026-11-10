"""Local moment proposals, before any language model is called.

Three signals, all computed from data we already have:

- **topic_shift** — adjacent ~20s blocks of the transcript stop sharing words.
- **pause** / **speaker** — a long gap, or a new speaker turn, opens the next stretch.
- **loud** — a short RMS peak in the extracted WAV (a shout, a hit, a laugh).

Each signal opens one candidate that runs from that point to the next cut, and
never longer than ``max_s``. The model then sees only these spans (plus a few
seconds of lead-in), not an 8-minute window of the whole VOD. When nothing
fires, the transcript is sliced into ``max_s`` pieces so a flat video still
gets scored.
"""

from __future__ import annotations

import logging
import re
import statistics
import struct
import wave
from dataclasses import dataclass, field
from pathlib import Path

from ..pipeline.highlights import render_window_text
from ..pipeline.prepare import Silence
from ..pipeline.transcript import Transcript
from ..providers import TranscriptWindow
from .models import MomentCandidate

log = logging.getLogger(__name__)

CANDIDATES_VERSION = "v1"

#: Topic comparison is done on blocks of about this much speech.
BLOCK_S = 20.0
#: A gap this long is a conversation cut, not a breath.
PAUSE_S = 1.8
#: Shorter gaps are only used to finish a candidate on a pause.
SNAP_GAP_S = 0.6
#: How far before a candidate the model is allowed to see.
CONTEXT_S = 12.0
#: RMS z-score, against the speech bins, that counts as a peak.
LOUD_Z = 1.7

#: Cap on how many candidates are sent to the model in one run.
MODEL_CAP = 40

REASON_ORDER = ("topic_shift", "pause", "speaker", "loud", "span")

_TOKEN = re.compile(r"[0-9a-záéíóúüñ]+", re.IGNORECASE)

#: Function words in the languages these VODs usually are. They dominate a raw
#: word count and would hide a real change of subject.
_STOP = frozenset(
    """
    el la los las un una unos unas de del al y o u pero que porque si no ya lo
    le les se su sus por para con sin sobre entre cuando donde qué cual quien
    esto esta estos estas ese esa eso muy más mas también tambien hay ser estar
    este esta hay fue son es como the a an and or but if of to in on for with
    at from this that it is are was were be as not you we they i me my your
    so just like yeah yes
    """.split()  # noqa: SIM905 — a word list reads better as prose than as a literal
)


@dataclass
class _Cue:
    time: float
    reasons: set[str] = field(default_factory=set)
    prior: float = 0.0
    similarity: float | None = None
    loudness_z: float | None = None


def build_candidates(
    transcript: Transcript,
    *,
    min_s: float,
    max_s: float,
    silences: list[Silence] | None = None,
    audio: Path | None = None,
) -> list[MomentCandidate]:
    """Propose moments. Pure aside from reading ``audio`` for loudness."""
    if not transcript.words or max_s <= 0:
        return []
    silences = silences or []
    cues = _cues(transcript, silences, audio)
    if cues:
        moments = _from_cues(transcript, cues, min_s=min_s, max_s=max_s, silences=silences)
    else:
        moments = _fallback(transcript, min_s=min_s, max_s=max_s)
    moments.sort(key=lambda m: (m.start_s, m.end_s))
    log.info("%d local candidate(s) from %d cue(s).", len(moments), len(cues))
    return moments


def for_model(moments: list[MomentCandidate], top: int) -> list[MomentCandidate]:
    """The moments worth a model call: highest prior first, capped.

    The ceiling grows with ``top`` so a run that asks for 20 clips still has
    rejects to throw away, and stops growing so a two-hour VOD cannot turn
    into hundreds of calls.
    """
    ceiling = min(MODEL_CAP, max(top * 2, 8))
    ordered = sorted(moments, key=lambda m: (-m.prior, m.start_s))
    return ordered[: min(len(ordered), ceiling)]


def note_for(moment: MomentCandidate) -> str:
    """The line appended to the user message for this moment."""
    reasons = ", ".join(moment.reasons)
    return (
        f"proposed={moment.start_s:.1f}-{moment.end_s:.1f} "
        f"A local pass marked this span (words {moment.start_word}–{moment.end_word}) "
        f"because: {reasons}. The text includes a few seconds of lead-in before the "
        "span. Return at most one clip, tightened to the moment, or an empty list "
        "if it is not worth clipping."
    )


def window_for(transcript: Transcript, moment: MomentCandidate) -> TranscriptWindow:
    """The moment plus a short lead-in, in the same ``[index]word`` form as before."""
    start = max(0.0, moment.start_s - CONTEXT_S)
    covered = _cover(transcript, start, moment.end_s)
    first, last = covered if covered is not None else (moment.start_word, moment.end_word)
    speakers = sorted({w.speaker for w in transcript.slice(first, last) if w.speaker})
    start_s, end_s = transcript.time_range(first, last)
    return TranscriptWindow(
        text=render_window_text(transcript, first, last),
        first_word=first,
        last_word=last,
        duration_s=end_s - start_s,
        speakers=speakers,
    )


# -- cues -------------------------------------------------------------------


def _cues(transcript: Transcript, silences: list[Silence], audio: Path | None) -> list[_Cue]:
    cues = [
        *_topic_cues(transcript),
        *_pause_cues(transcript, silences),
        *_speaker_cues(transcript),
    ]
    cues.extend(_loud_cues(audio))
    return _cluster(cues)


def _topic_cues(transcript: Transcript) -> list[_Cue]:
    blocks = _blocks(transcript)
    if len(blocks) < 2:
        return []
    sims: list[float | None] = []
    for left, right in zip(blocks, blocks[1:], strict=False):
        sims.append(_cosine(_counts(left[1]), _counts(right[1])))
    valid = [s for s in sims if s is not None]
    if not valid:
        return []
    median = statistics.median(valid)
    cues: list[_Cue] = []
    for index, sim in enumerate(sims):
        if not _is_shift(sim, median, len(valid), blocks[index][1], blocks[index + 1][1]):
            continue
        assert sim is not None
        cues.append(
            _Cue(
                time=blocks[index + 1][0],
                reasons={"topic_shift"},
                prior=min(0.95, 0.45 + 0.4 * (1.0 - sim)),
                similarity=round(sim, 3),
            )
        )
    return cues


def _is_shift(
    sim: float | None,
    median: float,
    n_valid: int,
    left_text: str,
    right_text: str,
) -> bool:
    """True when the two blocks are about different things.

    A single low score is not enough: a stream of distinct words (names, filler
    the tokenizer kept) scores near zero against everything. A real change is a
    drop below the video's own baseline, and each side has to be consistent
    with itself when there are too few blocks for a baseline.
    """
    if sim is None or sim > 0.28:
        return False
    if n_valid >= 3 and sim < median - 0.15:
        return True
    return sim <= 0.22 and _self_similar(left_text) and _self_similar(right_text)


def _pause_cues(transcript: Transcript, silences: list[Silence]) -> list[_Cue]:
    cues: list[_Cue] = []
    words = transcript.words
    for index in range(1, len(words)):
        gap = words[index].start - words[index - 1].end
        if gap >= PAUSE_S:
            cues.append(_Cue(time=words[index].start, reasons={"pause"}, prior=_pause_prior(gap)))
    for silence in silences:
        if silence.duration < PAUSE_S:
            continue
        cues.append(_Cue(time=silence.end, reasons={"pause"}, prior=_pause_prior(silence.duration)))
    return cues


def _pause_prior(gap: float) -> float:
    return min(0.7, 0.34 + gap / 12.0)


def _speaker_cues(transcript: Transcript) -> list[_Cue]:
    if not transcript.has_diarization:
        return []
    cues: list[_Cue] = []
    words = transcript.words
    index = 0
    while index < len(words):
        speaker = words[index].speaker
        end = index + 1
        while end < len(words) and words[end].speaker == speaker:
            end += 1
        if speaker and index > 0 and end - index >= 4:
            cues.append(_Cue(time=words[index].start, reasons={"speaker"}, prior=0.42))
        index = end
    return cues


def _loud_cues(audio: Path | None) -> list[_Cue]:
    if audio is None or not audio.exists():
        return []
    try:
        bins = _rms_bins(audio)
    except (wave.Error, EOFError, OSError) as exc:
        log.warning("Loudness pass skipped (%s).", exc)
        return []
    cues: list[_Cue] = []
    for peak_at, z in _peaks(bins):
        # Start a few seconds before the peak so the shout is inside the span,
        # not on its first frame.
        cues.append(
            _Cue(
                time=max(0.0, peak_at - 4.0),
                reasons={"loud"},
                prior=min(0.85, 0.4 + z / 8.0),
                loudness_z=round(z, 2),
            )
        )
    return cues


def _cluster(cues: list[_Cue]) -> list[_Cue]:
    """Cues within 2s are one moment; keep every reason and the stronger prior."""
    if not cues:
        return []
    cues = sorted(cues, key=lambda c: c.time)
    groups: list[_Cue] = [cues[0]]
    for cue in cues[1:]:
        current = groups[-1]
        if cue.time - current.time <= 2.0:
            current.reasons |= cue.reasons
            current.prior = max(current.prior, cue.prior)
            if cue.similarity is not None:
                current.similarity = cue.similarity
            if cue.loudness_z is not None and (
                current.loudness_z is None or cue.loudness_z > current.loudness_z
            ):
                current.loudness_z = cue.loudness_z
            continue
        groups.append(cue)
    return groups


# -- spans ------------------------------------------------------------------


def _from_cues(
    transcript: Transcript,
    cues: list[_Cue],
    *,
    min_s: float,
    max_s: float,
    silences: list[Silence],
) -> list[MomentCandidate]:
    media_end = transcript.words[-1].end
    snaps = _snap_points(transcript, silences)
    kept: list[MomentCandidate] = []
    for cue in cues:
        start = cue.time
        end = min(media_end, start + max_s)
        for later in cues:
            if start + min_s <= later.time <= end:
                end = later.time
                break
        end = _snap_end(start, end, min_s, snaps)
        _add_span(transcript, kept, start, end, cue, min_s=min_s)
    return kept


def _fallback(transcript: Transcript, *, min_s: float, max_s: float) -> list[MomentCandidate]:
    """No pause, shift or peak. Slice the speech so the model still has spans."""
    kept: list[MomentCandidate] = []
    start = transcript.words[0].start
    media_end = transcript.words[-1].end
    blank = _Cue(time=start, reasons={"span"}, prior=0.15)
    while start < media_end - (min_s - 0.5):
        end = min(start + max_s, media_end)
        if end - start < min_s - 0.5:
            break
        _add_span(transcript, kept, start, end, blank, min_s=min_s)
        start = end
    return kept


def _add_span(
    transcript: Transcript,
    kept: list[MomentCandidate],
    start: float,
    end: float,
    cue: _Cue,
    *,
    min_s: float,
) -> None:
    if end - start < min_s - 0.5:
        return
    covered = _cover(transcript, start, end)
    if covered is None:
        return
    first, last = covered
    start_s, end_s = transcript.time_range(first, last)
    if end_s - start_s < min_s - 0.5:
        return
    excerpt = transcript.text_between(first, last)
    excerpt = re.sub(r"\s+", " ", excerpt).strip()
    if len(excerpt) > 280:
        excerpt = excerpt[:277].rstrip() + "..."
    moment = MomentCandidate(
        start_s=round(start_s, 3),
        end_s=round(end_s, 3),
        start_word=first,
        end_word=last,
        reasons=_ordered(cue.reasons),
        prior=round(cue.prior, 3),
        excerpt=excerpt,
        similarity=cue.similarity,
        loudness_z=cue.loudness_z,
    )
    _absorb(kept, moment)


def _absorb(kept: list[MomentCandidate], new: MomentCandidate) -> None:
    """Drop a moment that largely repeats one we have, keeping both reasons."""
    for other in kept:
        if _overlap_ratio(other, new) <= 0.5:
            continue
        winner, loser = (new, other) if new.prior > other.prior else (other, new)
        winner.reasons = _ordered(set(winner.reasons) | set(loser.reasons))
        winner.prior = round(max(winner.prior, loser.prior), 3)
        if winner.similarity is None:
            winner.similarity = loser.similarity
        if loser.loudness_z is not None and (
            winner.loudness_z is None or loser.loudness_z > winner.loudness_z
        ):
            winner.loudness_z = loser.loudness_z
        if winner is new:
            kept.remove(other)
            kept.append(new)
        return
    kept.append(new)


def _overlap_ratio(a: MomentCandidate, b: MomentCandidate) -> float:
    inter = max(0.0, min(a.end_s, b.end_s) - max(a.start_s, b.start_s))
    shorter = min(a.end_s - a.start_s, b.end_s - b.start_s)
    if shorter <= 0:
        return 0.0
    return inter / shorter


def _snap_end(start: float, end: float, min_s: float, snaps: list[float]) -> float:
    usable = [point for point in snaps if start + min_s <= point <= end]
    return usable[-1] if usable else end


def _snap_points(transcript: Transcript, silences: list[Silence]) -> list[float]:
    points = [silence.start for silence in silences if silence.duration >= SNAP_GAP_S]
    words = transcript.words
    for index in range(1, len(words)):
        if words[index].start - words[index - 1].end >= SNAP_GAP_S:
            points.append(words[index - 1].end)
    return sorted(points)


def _cover(transcript: Transcript, start: float, end: float) -> tuple[int, int] | None:
    """Inclusive word range spoken inside ``[start, end]``."""
    first, last_excl = transcript.indices_in_range(start - 0.02, end + 0.02)
    if last_excl <= first:
        return None
    last = last_excl - 1
    while last > first and transcript.words[last].start > end + 0.05:
        last -= 1
    return first, last


def _ordered(reasons: set[str] | list[str]) -> list[str]:
    present = set(reasons)
    return [reason for reason in REASON_ORDER if reason in present]


# -- text similarity --------------------------------------------------------


def _blocks(transcript: Transcript) -> list[tuple[float, str]]:
    """``(start_time, text)`` for each non-empty block of :data:`BLOCK_S`."""
    words = transcript.words
    blocks: list[tuple[float, str]] = []
    start = words[0].start
    media_end = words[-1].end
    while start < media_end - 1.0:
        first, last_excl = transcript.indices_in_range(start, start + BLOCK_S)
        if last_excl > first:
            text = transcript.text_between(first, last_excl - 1)
            if text:
                blocks.append((words[first].start, text))
        start += BLOCK_S
    return blocks


def _counts(text: str) -> dict[str, int]:
    bag: dict[str, int] = {}
    for token in _TOKEN.findall(text.lower()):
        if len(token) <= 2 or token in _STOP:
            continue
        bag[token] = bag.get(token, 0) + 1
    return bag


def _cosine(left: dict[str, int], right: dict[str, int]) -> float | None:
    if not left or not right:
        return None
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    left_n = sum(value * value for value in left.values()) ** 0.5
    right_n = sum(value * value for value in right.values()) ** 0.5
    if left_n == 0 or right_n == 0:
        return None
    return dot / (left_n * right_n)


def _self_similar(text: str) -> bool:
    """True when the two halves of a block talk about the same thing."""
    tokens = [t for t in _TOKEN.findall(text.lower()) if len(t) > 2 and t not in _STOP]
    if len(tokens) < 8:
        return False
    mid = len(tokens) // 2
    sim = _cosine(_bag(tokens[:mid]), _bag(tokens[mid:]))
    return sim is not None and sim >= 0.4


def _bag(tokens: list[str]) -> dict[str, int]:
    bag: dict[str, int] = {}
    for token in tokens:
        bag[token] = bag.get(token, 0) + 1
    return bag


# -- loudness ---------------------------------------------------------------


def _rms_bins(path: Path, bin_s: float = 1.0) -> list[float]:
    """Per-second RMS of a PCM WAV. One bin at a time, so a long VOD stays small."""
    bins: list[float] = []
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        if width != 2 or rate <= 0 or channels <= 0:
            return []
        per_bin = max(1, int(rate * bin_s))
        while True:
            raw = handle.readframes(per_bin)
            count = len(raw) // width
            if count == 0:
                break
            samples = struct.unpack("<" + "h" * count, raw[: count * width])
            if channels > 1:
                mixed = [
                    sum(samples[i : i + channels]) / channels
                    for i in range(0, count - channels + 1, channels)
                ]
            else:
                mixed = list(samples)
            if not mixed:
                continue
            bins.append((sum(sample * sample for sample in mixed) / len(mixed)) ** 0.5)
    return bins


def _peaks(bins: list[float]) -> list[tuple[float, float]]:
    """``(time, z)`` of local RMS peaks. Empty when the level never moves."""
    if len(bins) < 8:
        return []
    speech = [value for value in bins if value > 50]
    if len(speech) < 8:
        return []
    mean = statistics.fmean(speech)
    spread = statistics.pstdev(speech)
    # A steady tone (game music, a test sine) has no peak worth clipping, even
    # when codec noise gives it a non-zero deviation.
    if spread < 1 or mean <= 0 or spread / mean < 0.35:
        return []
    baseline = statistics.median(speech)
    found: list[tuple[float, float]] = []
    for index, value in enumerate(bins):
        z = (value - mean) / spread
        if z < LOUD_Z or value < baseline * 2.2:
            continue
        previous = bins[index - 1] if index else value
        nxt = bins[index + 1] if index + 1 < len(bins) else value
        if value >= previous and value >= nxt:
            at = index + 0.5
            if found and at - found[-1][0] < 12:
                if z > found[-1][1]:
                    found[-1] = (at, z)
            else:
                found.append((at, z))
    return found
