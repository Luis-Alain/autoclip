"""Find the streamer's facecam overlay, and plan the 9:16 split-screen crop.

Gameplay streams put the webcam in a fixed rectangle over the game. Two signals
find it, from ~40 frames sampled across the whole VOD:

1. **Where** — MediaPipe faces. The streamer's face sits in the same small
   area in most samples; faces in the game (cutscenes) move around, so a vote
   over a coarse grid picks the webcam's face and ignores the rest.
2. **How big** — temporal variation. Game pixels change a lot between samples
   minutes apart; the streamer's room barely changes. Scanning outward from the
   face, the overlay's edge is where variation jumps to the game's level. Where
   no jump is found the overlay runs to the frame edge (corner cams).

If the edges can't be found (e.g. a green-screened cutout with no rectangle),
a box proportional to the face is used instead. The result is cached as
``layout.json`` and can be overridden with ``--cam x,y,w,h``.

Split output (1080x1920): webcam on top (35%), gameplay below (65%).
"""

from __future__ import annotations

import logging
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

from ..pipeline.ffmpeg import ffmpeg_path

log = logging.getLogger(__name__)

OUT_W = 1080
OUT_H = 1920
#: Share of the 9:16 frame given to the webcam (top); the rest is gameplay.
CAM_SHARE = 0.35
CAM_H = int(round(OUT_H * CAM_SHARE / 2) * 2)  # 672, kept even for the encoder
GAME_H = OUT_H - CAM_H  # 1248

#: Frames sampled across the VOD for detection.
SAMPLES = 40
#: Width the variation map is computed at. Small is enough to find a border.
MAP_W = 320
#: Grid used to vote on where the recurring face is.
GRID = 8
#: The webcam face must appear in at least this share of samples.
MIN_FACE_SHARE = 0.3
#: A face wider than this share of the frame means a full-camera scene, not an overlay.
MAX_OVERLAY_FACE_W = 0.3
#: Border test: outside the overlay looks like game (>= this share of the game's
#: typical variation), inside it doesn't (<= this share).
GAME_LEVEL = 0.75
CAM_LEVEL = 0.6
#: Clip-level check: the face must be inside the webcam rect in this share of
#: sampled frames for a clip to use the split layout.
CLIP_FACE_SHARE = 0.3

Method = Literal["variation", "face-box", "manual", "none"]


class Rect(BaseModel):
    """A rectangle in source pixels."""

    x: int
    y: int
    w: int
    h: int

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    @classmethod
    def parse(cls, text: str) -> Rect:
        """From ``"x,y,w,h"`` (the --cam option)."""
        try:
            x, y, w, h = (int(float(v)) for v in text.split(","))
        except ValueError as exc:
            raise ValueError(f"Expected x,y,w,h in pixels, got {text!r}.") from exc
        if w <= 0 or h <= 0:
            raise ValueError("Width and height must be positive.")
        return cls(x=x, y=y, w=w, h=h)


class Layout(BaseModel):
    """``layout.json``: where the webcam is in this VOD, if anywhere."""

    frame_w: int
    frame_h: int
    cam: Rect | None = None
    #: The webcam face's centre, used to keep the face centred when cropping.
    face_cx: float | None = None
    face_cy: float | None = None
    method: Method = "none"
    #: How many samples showed the face in the cam area, out of how many.
    face_hits: int = 0
    samples: int = 0


# -- detection ------------------------------------------------------------------


def detect_layout(video: Path, *, duration_s: float, frame_w: int, frame_h: int) -> Layout:
    """Find the webcam overlay. Returns a Layout with ``cam=None`` when there isn't one."""
    times = list(np.linspace(duration_s * 0.02, duration_s * 0.98, SAMPLES))
    faces_per_frame = [_faces_at(video, t, frame_w, frame_h) for t in times]

    # Vote: in how many samples does a (small) face sit in each grid cell?
    votes: Counter[tuple[int, int]] = Counter()
    for faces in faces_per_frame:
        cells = {_cell(f, frame_w, frame_h) for f in faces if f[2] < MAX_OVERLAY_FACE_W * frame_w}
        votes.update(cells)

    base = Layout(frame_w=frame_w, frame_h=frame_h, samples=len(times))
    if not votes:
        log.info("No small recurring face found; no webcam overlay.")
        return base
    ((cell, hits),) = votes.most_common(1)
    base.face_hits = hits
    if hits < MIN_FACE_SHARE * len(times):
        log.info(
            "Best face cell %s seen in only %d/%d samples; no overlay.", cell, hits, len(times)
        )
        return base

    cluster = [f for faces in faces_per_frame for f in faces if _cell(f, frame_w, frame_h) == cell]
    cx, cy, fw, fh = (float(np.median([f[i] for f in cluster])) for i in range(4))
    base.face_cx, base.face_cy = cx, cy

    variation = _variation_map(video, times, frame_w, frame_h)
    rect = find_overlay(variation, (cx, cy, fw, fh), frame_w, frame_h)
    if rect is not None:
        base.cam, base.method = rect, "variation"
    else:
        base.cam, base.method = face_box(cx, cy, fw, fh, frame_w, frame_h), "face-box"
    log.info("Webcam overlay: %s via %s (face in %d/%d samples).",
             base.cam, base.method, hits, len(times))  # fmt: skip
    return base


def find_overlay(
    variation: np.ndarray,
    face: tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
) -> Rect | None:
    """Locate the overlay rectangle around ``face`` in a variation map.

    ``variation`` is a (rows, cols) map of per-pixel standard deviation over
    time, at any scale; ``face`` is (cx, cy, w, h) in source pixels. Returns
    None when no side shows a game-like border (nothing separates the face from
    the game, so the face-box fallback should be used).
    """
    rows, cols = variation.shape
    sx, sy = cols / frame_w, rows / frame_h
    cx, cy, fw, fh = face
    left, right = int((cx - fw / 2) * sx), int((cx + fw / 2) * sx)
    top, bottom = int((cy - fh / 2) * sy), int((cy + fh / 2) * sy)
    left, top = max(0, left), max(0, top)
    right, bottom = min(cols - 1, right), min(rows - 1, bottom)

    game = float(np.median(variation))
    if game <= 0:
        return None

    col_profile = variation[top : bottom + 1, :].mean(axis=0)  # one value per column
    row_profile = variation[:, left : right + 1].mean(axis=1)  # one value per row

    x0 = _border(col_profile, start=left, step=-1, game=game)
    x1 = _border(col_profile, start=right, step=+1, game=game)
    y0 = _border(row_profile, start=top, step=-1, game=game)
    y1 = _border(row_profile, start=bottom, step=+1, game=game)

    found = [b for b in (x0, x1, y0, y1) if b is not None]
    if not found:
        return None
    # A side with no border runs to the frame edge.
    x0 = 0 if x0 is None else x0
    x1 = cols if x1 is None else x1
    y0 = 0 if y0 is None else y0
    y1 = rows if y1 is None else y1

    rect = Rect(
        x=_even(x0 / sx),
        y=_even(y0 / sy),
        w=_even((x1 - x0) / sx),
        h=_even((y1 - y0) / sy),
    )
    # An "overlay" covering most of the frame is a full-camera scene, not a cam box.
    if rect.w * rect.h > 0.5 * frame_w * frame_h:
        return None
    return rect


def _border(profile: np.ndarray, *, start: int, step: int, game: float, k: int = 3) -> int | None:
    """Scan from ``start`` in direction ``step`` for the overlay's edge.

    The edge is the first position where the next ``k`` cells look like game
    (high variation) and the previous ``k`` don't. Returns the index of the
    first outside cell's boundary (so it can be used as a rect edge), or None.
    """
    n = len(profile)
    p = start
    while 0 <= p < n:
        if step > 0:
            inside, outside = profile[max(0, p - k) : p], profile[p : p + k]
        else:
            inside, outside = profile[p + 1 : p + 1 + k], profile[max(0, p - k + 1) : p + 1]
        if (
            len(inside)
            and len(outside) == k
            and outside.mean() >= GAME_LEVEL * game
            and inside.mean() <= CAM_LEVEL * game
        ):
            return p if step > 0 else p + 1
        p += step
    return None


def face_box(cx: float, cy: float, fw: float, fh: float, frame_w: int, frame_h: int) -> Rect:
    """Fallback overlay: a head-and-shoulders box around the face."""
    w, h = fw * 2.6, fh * 2.4
    x = min(max(0.0, cx - w / 2), frame_w - w)
    y = min(max(0.0, cy - h * 0.45), frame_h - h)
    return Rect(x=_even(max(0, x)), y=_even(max(0, y)), w=_even(min(w, frame_w)),
                h=_even(min(h, frame_h)))  # fmt: skip


# -- per-clip decision --------------------------------------------------------


def clip_shows_cam(video: Path, layout: Layout, start_s: float, end_s: float) -> bool:
    """True when the webcam face is inside the overlay for enough of the clip.

    Streams switch scenes (a "just chatting" full-cam scene, a pause screen), so
    the split layout is only used where the overlay is actually on screen.
    """
    if layout.cam is None:
        return False
    times = np.linspace(start_s, end_s, max(3, int((end_s - start_s) / 3)))
    hits = sum(
        any(layout.cam.contains(f[0], f[1]) for f in _faces_at(video, t, layout.frame_w,
                                                                 layout.frame_h))
        for t in times
    )  # fmt: skip
    return hits >= CLIP_FACE_SHARE * len(times)


#: A face this wide (share of frame width) means a talking-head shot.
TALKING_FACE_W = 0.08


def clip_shows_face(video: Path, layout: Layout, start_s: float, end_s: float) -> bool:
    """True when a talking-head-sized face is on screen for enough of the clip.

    Used for VODs without a webcam overlay, to tell a camera scene (face-tracked
    crop) from pure gameplay (blur layout).
    """
    times = np.linspace(start_s, end_s, max(3, int((end_s - start_s) / 3)))
    min_w = TALKING_FACE_W * layout.frame_w
    hits = sum(
        any(f[2] >= min_w for f in _faces_at(video, t, layout.frame_w, layout.frame_h))
        for t in times
    )
    return hits >= CLIP_FACE_SHARE * len(times)


# -- crop geometry ------------------------------------------------------------


@dataclass(frozen=True)
class Crop:
    x: int
    y: int
    w: int
    h: int

    def filter(self) -> str:
        return f"crop={self.w}:{self.h}:{self.x}:{self.y}"


def split_crops(layout: Layout) -> tuple[Crop, Crop]:
    """Source crops for the webcam (top) and gameplay (bottom) panels.

    Each crop has its panel's aspect ratio, so scaling to the panel never
    distorts. The webcam crop keeps the face centred; the gameplay crop is the
    centre of the frame, nudged sideways if it would include the webcam.

    Preconditions:
        layout.cam is not None.
    """
    assert layout.cam is not None
    cam, fw, fh = layout.cam, layout.frame_w, layout.frame_h
    face_x = layout.face_cx if layout.face_cx is not None else cam.x + cam.w / 2
    face_y = layout.face_cy if layout.face_cy is not None else cam.y + cam.h / 2

    # Webcam: crop the overlay down to the panel's aspect (1080:672).
    panel = OUT_W / CAM_H
    if cam.w / cam.h > panel:
        cw, ch = cam.h * panel, float(cam.h)
        cx = _clamp(face_x - cw / 2, cam.x, cam.x + cam.w - cw)
        cy = float(cam.y)
    else:
        cw, ch = float(cam.w), cam.w / panel
        cx = float(cam.x)
        # Face centre at 55% of the panel height: headroom above, chin room below.
        cy = _clamp(face_y - ch * 0.55, cam.y, cam.y + cam.h - ch)
    cam_crop = Crop(_even(cx), _even(cy), _even(cw), _even(ch))

    # Gameplay: full height, the panel's aspect (1080:1248), centred.
    gh = float(fh)
    gw = min(float(fw), gh * OUT_W / GAME_H)
    gx = (fw - gw) / 2
    overlaps = gx < cam.x + cam.w and cam.x < gx + gw
    if overlaps and cam.h < fh * 0.6:  # a corner overlay, not a full-height one
        if cam.x + cam.w / 2 > fw / 2:  # webcam on the right: move left of it
            gx = max(0.0, min(gx, cam.x - gw))
        else:  # webcam on the left: move right of it
            gx = min(fw - gw, max(gx, cam.x + cam.w))
    game_crop = Crop(_even(gx), 0, _even(gw), _even(gh))
    return cam_crop, game_crop


# -- frame sampling -------------------------------------------------------------


def _faces_at(video: Path, t: float, frame_w: int, frame_h: int) -> list[tuple[float, ...]]:
    """Faces at time ``t`` as (cx, cy, w, h) in source pixels."""
    frame = _grab_rgb(video, t, frame_w, frame_h)
    if frame is None:
        return []
    return [(f.cx, f.cy, f.width, f.height) for f in _detect(frame, frame_w, frame_h)]


def _grab_rgb(video: Path, t: float, w: int, h: int) -> np.ndarray | None:
    """One RGB frame at ``t`` via ffmpeg (handles AV1 and anything else ffmpeg reads)."""
    out = subprocess.run(
        [ffmpeg_path(), "-v", "error", "-ss", f"{t:.3f}", "-i", str(video),
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=False,
    ).stdout  # fmt: skip
    if len(out) != w * h * 3:
        return None
    return np.frombuffer(out, np.uint8).reshape(h, w, 3)


_LANDMARKER = None


def _detect(frame: np.ndarray, w: int, h: int):  # type: ignore[no-untyped-def]
    """MediaPipe face landmarks on one still frame (image mode, model reused)."""
    global _LANDMARKER  # one model per process; creating it costs ~100 ms
    import os

    os.environ.setdefault("GLOG_minloglevel", "2")
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

    from .. import models
    from ..pipeline.reframe.faces import _to_observation

    if _LANDMARKER is None:
        _LANDMARKER = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(models.ensure("face_landmarker"))),
                running_mode=RunningMode.IMAGE,
                num_faces=4,
                min_face_detection_confidence=0.4,
            )
        )
    result = _LANDMARKER.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame))
    observations = [_to_observation(lm, 0.0, w, h) for lm in result.face_landmarks or []]
    return [o for o in observations if o is not None]


def _variation_map(video: Path, times: list[float], frame_w: int, frame_h: int) -> np.ndarray:
    """Per-pixel standard deviation over the sampled frames, at MAP_W wide."""
    map_h = max(1, round(MAP_W * frame_h / frame_w))
    frames = []
    for t in times:
        out = subprocess.run(
            [ffmpeg_path(), "-v", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
             "-vf", f"scale={MAP_W}:{map_h}", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
            capture_output=True, check=False,
        ).stdout  # fmt: skip
        if len(out) == MAP_W * map_h:
            frames.append(np.frombuffer(out, np.uint8).reshape(map_h, MAP_W))
    return np.stack(frames).astype(np.float32).std(axis=0)


# -- small helpers ----------------------------------------------------------------


def _cell(face: tuple[float, ...], frame_w: int, frame_h: int) -> tuple[int, int]:
    return (
        min(GRID - 1, int(face[0] / frame_w * GRID)),
        min(GRID - 1, int(face[1] / frame_h * GRID)),
    )


def _even(value: float) -> int:
    """Round to an even integer (H.264 with 4:2:0 needs even crop sizes)."""
    return int(round(value / 2) * 2)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))
