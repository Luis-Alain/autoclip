"""Raw 16:9 export: cut each clip from the source at full quality, with handles.

No crop, no scaling, no captions — the file is the source's own frames, so it
drops straight into a Premiere timeline at the source's size and frame rate.

The cut is re-encoded rather than stream-copied: ``-c copy`` can only cut on
keyframes, which in stream VODs are often 2+ seconds apart, so the clip would
start at the wrong moment.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..pipeline import ffmpeg
from ..system import report
from .slug import slugify

#: VideoToolbox constant quality for the raw (editing) file. Higher than the
#: 9:16 export's: this is an intermediate that gets edited and re-encoded.
RAW_VT_QUALITY = 80
#: libx264 fallback quality when hardware encoding is unavailable.
RAW_CRF = 16


@dataclass(frozen=True)
class RawCut:
    """Where a clip is cut from the source, and where its edit points sit inside the file.

    ``handle_*`` are absolute seconds in the source (the file's extent).
    ``in_point``/``out_point`` are seconds inside the file (the chosen moment).
    """

    handle_start: float
    handle_end: float
    in_point: float
    out_point: float

    @property
    def file_duration(self) -> float:
        return self.handle_end - self.handle_start


def plan_cut(start_s: float, end_s: float, *, handles_s: float, source_duration_s: float) -> RawCut:
    """Extend ``[start_s, end_s]`` by handles, clamped to the source.

    Preconditions:
        0 <= start_s < end_s <= source_duration_s; handles_s >= 0.
    """
    handle_start = max(0.0, start_s - handles_s)
    handle_end = min(source_duration_s, end_s + handles_s)
    return RawCut(
        handle_start=round(handle_start, 3),
        handle_end=round(handle_end, 3),
        in_point=round(start_s - handle_start, 3),
        out_point=round(end_s - handle_start, 3),
    )


def raw_filename(rank: int, score: int, title: str, *, prores: bool) -> str:
    """``01_s92_el-grito-de-xqc.mp4`` (``.mov`` for ProRes)."""
    extension = "mov" if prores else "mp4"
    return f"{rank:02d}_s{score}_{slugify(title)}.{extension}"


def encode_args(*, prores: bool, hardware: bool) -> list[str]:
    """Video + audio codec arguments for the raw file."""
    if prores:
        video = (
            ["-c:v", "prores_videotoolbox", "-profile:v", "standard"]
            if hardware
            else ["-c:v", "prores_ks", "-profile:v", "2"]
        )
        # PCM audio: lossless, and what ProRes .mov files conventionally carry.
        return [*video, "-c:a", "pcm_s16le"]

    video = (
        [
            "-c:v",
            "h264_videotoolbox",
            "-q:v",
            str(RAW_VT_QUALITY),
            "-profile:v",
            "high",
            "-allow_sw",
            "1",
        ]
        if hardware
        else ["-c:v", "libx264", "-preset", "medium", "-crf", str(RAW_CRF), "-profile:v", "high"]
    )
    return [*video, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "320k", "-movflags", "+faststart"]


def export_raw_clip(
    source: Path,
    destination: Path,
    cut: RawCut,
    *,
    prores: bool = False,
    on_progress: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Path:
    """Render one raw clip. Writes to a temp name first, so a partial file is never kept."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".partial-{destination.name}")

    args = [
        # Input-side seek: fast, and frame-accurate because we re-encode.
        "-ss",
        f"{cut.handle_start:.3f}",
        "-t",
        f"{cut.file_duration:.3f}",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",  # the "?" makes audio optional, for silent sources
        *encode_args(prores=prores, hardware=report().ffmpeg.videotoolbox_works),
        str(partial),
    ]
    try:
        ffmpeg.run(
            args,
            total_duration_s=cut.file_duration,
            on_progress=on_progress,
            cancelled=cancelled,
        )
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(destination)
    return destination
