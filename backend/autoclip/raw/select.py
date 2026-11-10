"""Highlight selection for the raw pipeline.

Upstream's provider and word-index contract are reused unchanged; what's
different is the prompt (stream moments, titles in the transcript's language,
``needs_context``) and that the result keeps those extra fields instead of
being flattened into a database row.

The model does not see 8-minute slices of the VOD. A local pass
(:mod:`autoclip.raw.candidates`) proposes moments from topic shifts, pauses,
speaker changes and loudness peaks, and each model call is one of those
moments plus a few seconds of lead-in.

Constraints are enforced here in code, not trusted to the prompt: the length
range, overlap dedupe (higher score wins), and the top-N cut.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from ..pipeline import boundaries
from ..pipeline.highlights import dedupe
from ..pipeline.prepare import Silence
from ..pipeline.transcript import Transcript
from ..providers import ClipCandidate, DetectionConfig, LLMProvider
from .candidates import (
    BLOCK_S,
    CANDIDATES_VERSION,
    build_candidates,
    for_model,
    note_for,
    window_for,
)
from .models import CandidatesArtifact, MomentCandidate, RawClip, write_model

log = logging.getLogger(__name__)

PROMPT_VERSION = "highlight_score_v1"

#: One moment per call. The local pass already split the VOD; asking the model
#: for several clips inside one moment just re-merges what we separated.

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
    audio: Path | None = None,
    candidates_out: Path | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> list[RawClip]:
    """Score locally proposed moments, then rank whatever the model keeps."""
    moments = build_candidates(
        transcript, min_s=min_s, max_s=max_s, silences=silences, audio=audio
    )
    chosen = for_model(moments, top)
    sent = {(m.start_word, m.end_word) for m in chosen}
    for moment in moments:
        moment.sent = (moment.start_word, moment.end_word) in sent
    if candidates_out is not None:
        write_model(
            candidates_out,
            CandidatesArtifact(version=CANDIDATES_VERSION, block_s=BLOCK_S, candidates=moments),
        )
    if not chosen:
        raise SelectionError(
            "No candidate moments were found. The transcript may be shorter than "
            f"{min_s:.0f}s, or there is nothing to clip."
        )

    config = DetectionConfig(
        min_duration_s=min_s,
        max_duration_s=max_s,
        max_clips=1,
        language=transcript.language,
        prompt_version=PROMPT_VERSION,
    )
    concurrency = LOCAL_CONCURRENCY if provider.name in ("ollama", "mlx") else HOSTED_CONCURRENCY
    semaphore = asyncio.Semaphore(concurrency)
    done = 0
    total = len(chosen)

    async def run_moment(moment: MomentCandidate) -> list[ClipCandidate]:
        nonlocal done
        window = window_for(transcript, moment)
        moment_config = replace(config, extra_context=note_for(moment))
        async with semaphore:
            try:
                result = await provider.detect_highlights(window, moment_config)
                found = result.clips
            except Exception as exc:
                # One bad moment shouldn't lose all the others.
                log.warning(
                    "Moment %d-%d failed (%s); continuing.",
                    moment.start_word,
                    moment.end_word,
                    exc,
                )
                found = []
            done += 1
            if on_progress:
                on_progress(done / total)
            return found

    log.info(
        "Scoring %d of %d local moment(s) with %s.", len(chosen), len(moments), provider.name
    )
    groups = await asyncio.gather(*(run_moment(m) for m in chosen))
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
