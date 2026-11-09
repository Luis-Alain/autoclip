"""Highlight selection for the raw pipeline.

Upstream's provider and word-index contract are reused unchanged; what's
different is the prompt (stream moments, titles in the transcript's language,
``needs_context``) and that the result keeps those extra fields instead of
being flattened into a database row.

Constraints are enforced here in code, not trusted to the prompt: the length
range, overlap dedupe (higher score wins), and the top-N cut.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from ..pipeline import boundaries
from ..pipeline.highlights import build_windows, dedupe
from ..pipeline.prepare import Silence
from ..pipeline.transcript import Transcript
from ..providers import ClipCandidate, DetectionConfig, LLMProvider, TranscriptWindow
from .models import RawClip

log = logging.getLogger(__name__)

PROMPT_VERSION = "highlight_raw_v1"

#: Upper bound on clips requested per transcript window. The global top-N is
#: applied after merging all windows.
PER_WINDOW_MAX = 8

#: Two clips sharing more than this fraction of their words are the same moment.
OVERLAP_IOU = 0.4

HOSTED_CONCURRENCY = 3
LOCAL_CONCURRENCY = 1


class SelectionError(RuntimeError):
    """No usable clips came out of highlight detection."""


async def select_clips(
    transcript: Transcript,
    provider: LLMProvider,
    *,
    min_s: float,
    max_s: float,
    top: int,
    silences: list[Silence] | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> list[RawClip]:
    """Ask the provider for candidates window by window, then rank them."""
    windows = build_windows(transcript)
    if not windows:
        raise SelectionError("The transcript is empty, so there is nothing to clip.")

    config = DetectionConfig(
        min_duration_s=min_s,
        max_duration_s=max_s,
        max_clips=min(top, PER_WINDOW_MAX),
        language=transcript.language,
        prompt_version=PROMPT_VERSION,
    )
    concurrency = LOCAL_CONCURRENCY if provider.name == "ollama" else HOSTED_CONCURRENCY
    semaphore = asyncio.Semaphore(concurrency)
    done = 0

    async def run_window(window: TranscriptWindow) -> list[ClipCandidate]:
        nonlocal done
        async with semaphore:
            try:
                result = await provider.detect_highlights(window, config)
                found = result.clips
            except Exception as exc:
                # One bad window shouldn't lose all the others.
                log.warning(
                    "Window %d-%d failed (%s); continuing.",
                    window.first_word,
                    window.last_word,
                    exc,
                )
                found = []
            done += 1
            if on_progress:
                on_progress(done / len(windows))
            return found

    log.info("Detecting highlights in %d window(s) with %s.", len(windows), provider.name)
    groups = await asyncio.gather(*(run_window(w) for w in windows))
    candidates = [c for group in groups for c in group]
    log.info("Provider proposed %d raw candidates.", len(candidates))

    clips = rank_clips(
        transcript, candidates, min_s=min_s, max_s=max_s, top=top, silences=silences or []
    )
    if not clips:
        raise SelectionError(
            "No clips were found. The VOD may have no standalone moments in the "
            f"{min_s:.0f}-{max_s:.0f}s range, or the model struggled — try a larger "
            "model or another provider, or widen --min/--max."
        )
    return clips


def rank_clips(
    transcript: Transcript,
    candidates: list[ClipCandidate],
    *,
    min_s: float,
    max_s: float,
    top: int,
    silences: list[Silence] | None = None,
) -> list[RawClip]:
    """Refine, filter, dedupe and rank candidates. Pure: no I/O, no model calls."""
    refined: list[RawClip] = []
    for candidate in dedupe(candidates, iou_threshold=OVERLAP_IOU):
        boundary = boundaries.refine(
            transcript,
            candidate.start_word_index,
            candidate.end_word_index,
            silences=silences or [],
            min_duration_s=min_s,
            max_duration_s=max_s,
        )
        if boundary is None:
            continue
        # refine() aims for the range but can't always hit it exactly (e.g. no
        # sentence end nearby); the limits are hard requirements here.
        if not (min_s - 0.5 <= boundary.duration_s <= max_s + 0.5):
            log.debug(
                "Dropped %d-%d: %.1fs is outside the range.",
                boundary.start_word,
                boundary.end_word,
                boundary.duration_s,
            )
            continue

        title = candidate.title.strip()
        if not title or (title.startswith("<") and title.endswith(">")):
            # Missing, or the model echoed the schema placeholder.
            title = transcript.text_between(
                boundary.start_word, min(boundary.start_word + 8, boundary.end_word)
            )

        refined.append(
            RawClip(
                score=candidate.score,
                start_s=boundary.start_s,
                end_s=boundary.end_s,
                start_word=boundary.start_word,
                end_word=boundary.end_word,
                hook_title=title,
                hook=candidate.hook.strip(),
                reason=candidate.reason.strip(),
                needs_context=candidate.needs_context,
            )
        )

    # Refinement can move edges until two survivors overlap; dedupe again.
    kept: list[RawClip] = []
    for clip in sorted(refined, key=lambda c: c.score, reverse=True):
        if all(_iou(clip, other) <= OVERLAP_IOU for other in kept):
            kept.append(clip)

    kept = kept[:top]
    for rank, clip in enumerate(kept, start=1):
        clip.rank = rank
    return kept


def _iou(a: RawClip, b: RawClip) -> float:
    """Intersection over union of two inclusive word ranges."""
    inter = max(0, min(a.end_word, b.end_word) - max(a.start_word, b.start_word) + 1)
    if inter == 0:
        return 0.0
    union = (a.end_word - a.start_word + 1) + (b.end_word - b.start_word + 1) - inter
    return inter / union
