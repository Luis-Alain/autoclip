"""Vertical 9:16 export with burned-in captions, built on upstream's stages.

Upstream already does the hard parts: MediaPipe speaker tracking produces a
crop path, and ``export.export_clip`` renders crop + animated ASS captions +
loudness normalisation in one ffmpeg pass (VideoToolbox on a Mac). This module
only adapts a :class:`RawClip` to those calls and caches the crop path.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..config import ExportSettings
from ..pipeline import export
from ..pipeline.captions import CaptionStyle
from ..pipeline.reframe import ReframeConfig, build_crop_path
from ..pipeline.reframe.croppath import CropPath
from ..pipeline.transcript import Transcript
from .models import RawClip
from .slug import slugify


def clip_filename(rank: int, score: int, title: str) -> str:
    """``01_s92_el-grito-de-xqc.mp4``"""
    return f"{rank:02d}_s{score}_{slugify(title)}.mp4"


def is_vertical_file(path: Path) -> bool:
    """True if ``path`` is a readable portrait video.

    Used when resuming, so a file left over from an older layout (e.g. a 16:9
    clip with the same name) is re-rendered instead of being taken as done.
    """
    from ..pipeline.ffmpeg import FFmpegError, probe

    try:
        info = probe(path)
    except FFmpegError:
        return False
    return bool(info.width and info.height and info.height > info.width)


def export_vertical_clip(
    source: Path,
    destination: Path,
    clip: RawClip,
    transcript: Transcript,
    *,
    style: CaptionStyle,
    settings: ExportSettings,
    work_dir: Path,
    on_progress: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Path:
    """Reframe one clip to 9:16 and burn its captions.

    ``work_dir`` holds the cached crop path and the caption workspace, so a
    resumed run skips face tracking for clips it already analysed.
    """
    crop_cache = work_dir / "crops" / f"{destination.stem}.json"
    if crop_cache.exists():
        crop = CropPath.load(crop_cache)
    else:
        crop = build_crop_path(
            source,
            start_s=clip.start_s,
            end_s=clip.end_s,
            transcript=transcript,
            config=ReframeConfig(aspect_w=9, aspect_h=16),
        )
        crop.save(crop_cache)

    # Render under a temp name, so an interrupted export never leaves a file
    # that a resumed run would mistake for a finished one.
    partial = destination.with_name(f".partial-{destination.name}")
    request = export.ExportRequest(
        source=source,
        destination=partial,
        start_s=clip.start_s,
        end_s=clip.end_s,
        crop_path=crop,
        # Absolute word timings; export_clip offsets them by start_s itself.
        words=transcript.slice(clip.start_word, clip.end_word),
        style=style,
        ratio="9:16",
    )
    try:
        export.export_clip(
            request,
            work_dir=work_dir / "captions",
            settings=settings,
            on_progress=on_progress,
            cancelled=cancelled,
        )
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(destination)
    return destination
