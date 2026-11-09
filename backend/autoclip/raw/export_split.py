"""Composite 9:16 renders for gameplay streams, in one ffmpeg pass each.

- **split** — webcam on top (35%), gameplay below (65%), captions on the seam.
  The source is split into two crops (``layout.split_crops``), each scaled to
  its panel, then stacked.
- **blur** — the whole 16:9 game frame, centred, over a blurred, darkened copy
  of itself filling 9:16. Captions sit just below the game. Used for gameplay
  moments without a webcam on screen.

Audio is loudness-normalised and the video encoded exactly like upstream's
export (VideoToolbox on a Mac).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from ..config import ExportSettings
from ..pipeline import captions, ffmpeg
from ..pipeline.captions import CaptionStyle
from ..pipeline.export import (
    AUDIO_BITRATE,
    AUDIO_SAMPLE_RATE,
    build_audio_filtergraph,
    encoder_args,
)
from ..pipeline.transcript import Transcript
from .layout import CAM_H, GAME_H, OUT_H, OUT_W, Layout, split_crops
from .models import RawClip

CompositeMode = Literal["split", "blur"]

#: Caption baseline as a share of frame height from the bottom. 0.56 puts the
#: text just under the webcam/gameplay seam (at 65% from the bottom), where it
#: covers the least gameplay and is read first.
SEAM_MARGIN_V = 0.56
#: In blur mode the game ends ~34% from the bottom; 0.26 puts captions just below it.
BLUR_MARGIN_V = 0.26
#: The blurred background is computed small and scaled up: same look, far cheaper.
_BLUR_W, _BLUR_H = 270, 480


def split_filtergraph(layout: Layout, *, subtitle_name: str | None, fonts_name: str) -> str:
    """The filter_complex for one split render, ending in ``[vout]``."""
    cam_crop, game_crop = split_crops(layout)
    parts = [
        # `split` duplicates the decoded video so each panel crops its own copy.
        "[0:v]split=2[src_cam][src_game]",
        f"[src_cam]{cam_crop.filter()},scale={OUT_W}:{CAM_H}:flags=lanczos,setsar=1[cam]",
        f"[src_game]{game_crop.filter()},scale={OUT_W}:{GAME_H}:flags=lanczos,setsar=1[game]",
        "[cam][game]vstack=inputs=2[stack]",
    ]
    if subtitle_name:
        parts.append(f"[stack]ass=filename={subtitle_name}:fontsdir={fonts_name}[vout]")
    else:
        parts.append("[stack]null[vout]")
    return ";".join(parts)


def blur_filtergraph(
    frame_w: int, frame_h: int, *, subtitle_name: str | None, fonts_name: str
) -> str:
    """The filter_complex for one blur-background render, ending in ``[vout]``."""
    game_h = _even(OUT_W * frame_h / frame_w)  # 608 for a 16:9 source
    parts = [
        "[0:v]split=2[src_bg][src_fg]",
        # Fill 9:16 (crop the sides), blur and darken a little so the game pops.
        f"[src_bg]scale={_BLUR_W}:{_BLUR_H}:force_original_aspect_ratio=increase,"
        f"crop={_BLUR_W}:{_BLUR_H},gblur=sigma=10,eq=brightness=-0.08,"
        f"scale={OUT_W}:{OUT_H},setsar=1[bg]",
        f"[src_fg]scale={OUT_W}:{game_h}:flags=lanczos,setsar=1[fg]",
        "[bg][fg]overlay=(W-w)/2:(H-h)/2[stack]",
    ]
    if subtitle_name:
        parts.append(f"[stack]ass=filename={subtitle_name}:fontsdir={fonts_name}[vout]")
    else:
        parts.append("[stack]null[vout]")
    return ";".join(parts)


def seam_style(style: CaptionStyle) -> CaptionStyle:
    """The caption style, moved to sit on the webcam/gameplay seam."""
    return dataclasses.replace(style, margin_v_ratio=SEAM_MARGIN_V)


def below_game_style(style: CaptionStyle) -> CaptionStyle:
    """The caption style, moved just below the game in blur mode."""
    return dataclasses.replace(style, margin_v_ratio=BLUR_MARGIN_V)


def composite_filtergraph(
    mode: CompositeMode, layout: Layout, *, subtitle_name: str | None, fonts_name: str
) -> str:
    if mode == "split":
        return split_filtergraph(layout, subtitle_name=subtitle_name, fonts_name=fonts_name)
    return blur_filtergraph(
        layout.frame_w, layout.frame_h, subtitle_name=subtitle_name, fonts_name=fonts_name
    )


def export_composite_clip(
    source: Path,
    destination: Path,
    clip: RawClip,
    transcript: Transcript,
    layout: Layout,
    *,
    mode: CompositeMode,
    style: CaptionStyle,
    settings: ExportSettings,
    work_dir: Path,
    on_progress: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Path:
    """Render one split or blur clip. Writes via a temp name, renamed when complete."""
    words = transcript.slice(clip.start_word, clip.end_word)
    workspace = work_dir / "captions" / destination.stem
    workspace.mkdir(parents=True, exist_ok=True)

    subtitle_name: str | None = None
    fonts_name = "fonts"
    render_cwd: Path | None = None
    if words:
        ass_path = captions.write_ass(
            workspace / "captions.ass",
            words,
            seam_style(style) if mode == "split" else below_game_style(style),
            width=OUT_W,
            height=OUT_H,
            time_offset_s=clip.start_s,
        )
        # The ass filter is run from the caption folder with relative names,
        # which sidesteps ffmpeg's filter-path escaping rules entirely.
        render_cwd, subtitle_name, fonts_name = ffmpeg.relative_filter_workspace(
            ass_path, captions.FONT_DIR
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".partial-{destination.name}")
    duration = clip.end_s - clip.start_s
    args = [
        "-ss", f"{clip.start_s:.4f}", "-t", f"{duration:.4f}", "-i", str(source),
        "-filter_complex", composite_filtergraph(mode, layout, subtitle_name=subtitle_name,
                                                 fonts_name=fonts_name),
        "-map", "[vout]", "-map", "0:a?",
        "-af", build_audio_filtergraph(settings),
        *encoder_args(settings),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ar", str(AUDIO_SAMPLE_RATE),
        "-movflags", "+faststart",
        str(partial.resolve()),
    ]  # fmt: skip
    try:
        ffmpeg.run(
            args,
            total_duration_s=duration,
            on_progress=on_progress,
            cancelled=cancelled,
            cwd=render_cwd,
        )
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(destination)
    return destination


def render_preview(
    source: Path,
    layout: Layout,
    at_s: float,
    destination: Path,
    *,
    mode: CompositeMode = "split",
) -> Path:
    """A single 9:16 frame of a layout (no captions), to check it by eye."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    graph = composite_filtergraph(mode, layout, subtitle_name=None, fonts_name="fonts")
    ffmpeg.run(
        [
            "-ss", f"{at_s:.3f}", "-i", str(source),
            "-filter_complex", graph,
            "-map", "[vout]", "-frames:v", "1", "-q:v", "3", str(destination.resolve()),
        ]
    )  # fmt: skip
    return destination


def render_detection(source: Path, layout: Layout, at_s: float, destination: Path) -> Path:
    """The source frame with the detected webcam (green) and gameplay crop (blue) drawn on."""
    cam_crop, game_crop = split_crops(layout)
    assert layout.cam is not None
    cam = layout.cam
    boxes = ",".join(
        [
            f"drawbox=x={cam.x}:y={cam.y}:w={cam.w}:h={cam.h}:color=lime@0.9:t=4",
            f"drawbox=x={game_crop.x}:y={game_crop.y}:w={game_crop.w}:h={game_crop.h}"
            ":color=deepskyblue@0.9:t=4",
            f"drawbox=x={cam_crop.x}:y={cam_crop.y}:w={cam_crop.w}:h={cam_crop.h}"
            ":color=yellow@0.9:t=2",
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg.run(
        ["-ss", f"{at_s:.3f}", "-i", str(source), "-vf", boxes, "-frames:v", "1",
         "-q:v", "3", str(destination.resolve())]
    )  # fmt: skip
    return destination


def _even(value: float) -> int:
    return int(round(value / 2) * 2)
