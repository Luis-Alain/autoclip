"""Webcam overlay detection and split-screen geometry (autoclip/raw/layout.py)."""

from __future__ import annotations

import numpy as np
import pytest
from autoclip.raw import layout as L
from autoclip.raw.export_split import (
    BLUR_MARGIN_V,
    SEAM_MARGIN_V,
    below_game_style,
    blur_filtergraph,
    seam_style,
    split_filtergraph,
)
from autoclip.raw.layout import Layout, Rect, find_overlay, split_crops

FRAME_W, FRAME_H = 1280, 720


def _variation_map(cam: tuple[int, int, int, int] | None, *, cols: int = 320) -> np.ndarray:
    """A synthetic per-pixel variation map: busy game (60) with a calm webcam box (20)."""
    rows = cols * FRAME_H // FRAME_W
    scale = cols / FRAME_W
    vmap = np.full((rows, cols), 60.0, dtype=np.float32)
    if cam:
        x, y, w, h = (int(v * scale) for v in cam)
        vmap[y : y + h, x : x + w] = 20.0
    return vmap


# -- overlay detection ----------------------------------------------------------


def test_corner_overlay_is_found_and_runs_to_the_frame_edges() -> None:
    # JuegaGerman's layout: a webcam in the top-right corner.
    vmap = _variation_map((1000, 0, 280, 208))
    face = (1136.0, 111.0, 90.0, 110.0)  # cx, cy, w, h in source pixels
    rect = find_overlay(vmap, face, FRAME_W, FRAME_H)
    assert rect is not None
    assert abs(rect.x - 1000) <= 8 and rect.y == 0
    assert rect.x + rect.w == FRAME_W
    assert abs(rect.h - 208) <= 8


def test_floating_overlay_has_all_four_borders() -> None:
    vmap = _variation_map((80, 400, 320, 240))
    rect = find_overlay(vmap, (240.0, 520.0, 90.0, 110.0), FRAME_W, FRAME_H)
    assert rect is not None
    assert abs(rect.x - 80) <= 8 and abs(rect.y - 400) <= 8
    assert abs(rect.w - 320) <= 10 and abs(rect.h - 240) <= 10


def test_no_border_means_no_overlay() -> None:
    # Uniformly busy around the face: nothing separates it from the game.
    vmap = _variation_map(None)
    assert find_overlay(vmap, (640.0, 360.0, 90.0, 110.0), FRAME_W, FRAME_H) is None


def test_face_box_fallback_stays_in_frame() -> None:
    rect = L.face_box(1250.0, 40.0, 90.0, 110.0, FRAME_W, FRAME_H)
    assert rect.x >= 0 and rect.y >= 0
    assert rect.x + rect.w <= FRAME_W and rect.y + rect.h <= FRAME_H


def test_rect_parse() -> None:
    assert Rect.parse("1000,0,280,208") == Rect(x=1000, y=0, w=280, h=208)
    with pytest.raises(ValueError):
        Rect.parse("1,2,3")
    with pytest.raises(ValueError):
        Rect.parse("0,0,0,10")


# -- split geometry -------------------------------------------------------------


def _layout(cam: Rect, face: tuple[float, float] | None = None) -> Layout:
    cx, cy = face or (cam.x + cam.w / 2, cam.y + cam.h / 2)
    return Layout(frame_w=FRAME_W, frame_h=FRAME_H, cam=cam, face_cx=cx, face_cy=cy)


def test_crops_have_the_panel_aspect_ratios() -> None:
    cam_crop, game_crop = split_crops(_layout(Rect(x=1000, y=0, w=280, h=208), (1136, 111)))
    assert cam_crop.w / cam_crop.h == pytest.approx(L.OUT_W / L.CAM_H, rel=0.03)
    assert game_crop.w / game_crop.h == pytest.approx(L.OUT_W / L.GAME_H, rel=0.03)
    for crop in (cam_crop, game_crop):
        assert crop.w % 2 == 0 and crop.h % 2 == 0  # yuv420p needs even sizes


def test_cam_crop_stays_inside_the_overlay() -> None:
    cam = Rect(x=1000, y=0, w=280, h=208)
    cam_crop, _ = split_crops(_layout(cam, (1136, 111)))
    assert cam.x <= cam_crop.x and cam_crop.x + cam_crop.w <= cam.x + cam.w
    assert cam.y <= cam_crop.y and cam_crop.y + cam_crop.h <= cam.y + cam.h


def test_gameplay_crop_avoids_a_corner_webcam() -> None:
    # A wide webcam reaching into the centre: the gameplay crop moves left of it.
    cam = Rect(x=700, y=0, w=580, h=240)
    _, game = split_crops(_layout(cam))
    assert game.x + game.w <= cam.x
    assert game.x >= 0


def test_panels_fill_the_9x16_frame() -> None:
    assert L.CAM_H + L.GAME_H == L.OUT_H == 1920
    assert pytest.approx(0.35 * 1920, abs=2) == L.CAM_H


def test_filtergraph_stacks_two_panels_and_captions() -> None:
    graph = split_filtergraph(
        _layout(Rect(x=1000, y=0, w=280, h=208)), subtitle_name="captions.ass", fonts_name="fonts"
    )
    assert "split=2" in graph
    assert f"scale={L.OUT_W}:{L.CAM_H}" in graph
    assert f"scale={L.OUT_W}:{L.GAME_H}" in graph
    assert "vstack=inputs=2" in graph
    assert graph.endswith("ass=filename=captions.ass:fontsdir=fonts[vout]")


def test_captions_move_to_the_seam() -> None:
    from autoclip.pipeline.captions import get_style

    assert seam_style(get_style("bold_pop")).margin_v_ratio == SEAM_MARGIN_V


def test_blur_filtergraph_centres_the_whole_game_over_a_blurred_fill() -> None:
    graph = blur_filtergraph(1280, 720, subtitle_name=None, fonts_name="fonts")
    assert "gblur" in graph
    assert f"scale={L.OUT_W}:608" in graph  # whole 16:9 frame at full width
    assert "overlay=(W-w)/2:(H-h)/2" in graph
    assert graph.endswith("null[vout]")


def test_blur_captions_sit_below_the_game() -> None:
    from autoclip.pipeline.captions import get_style

    assert below_game_style(get_style("bold_pop")).margin_v_ratio == BLUR_MARGIN_V
    # The game ends at (1920 + 608) / 2 = 1264 px from the top: captions go under it.
    assert BLUR_MARGIN_V * L.OUT_H < L.OUT_H - 1264


# -- per-clip layout choice ---------------------------------------------------------


@pytest.mark.parametrize(
    ("has_overlay", "cam_shown", "face_shown", "expected"),
    [
        (True, True, False, "split"),
        (True, False, False, "blur"),
        (False, False, True, "face"),
        (False, False, False, "blur"),
    ],
)
def test_auto_layout_choice(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,  # type: ignore[no-untyped-def]
    has_overlay: bool,
    cam_shown: bool,
    face_shown: bool,
    expected: str,
) -> None:
    from autoclip.config import Settings
    from autoclip.raw import runner as raw_runner
    from autoclip.raw.models import RawClip, SourceInfo

    monkeypatch.setattr(raw_runner, "clip_shows_cam", lambda *a: cam_shown)
    monkeypatch.setattr(raw_runner, "clip_shows_face", lambda *a: face_shown)
    source = SourceInfo(path="/v.mp4", duration_s=100, width=FRAME_W, height=FRAME_H)
    runner = raw_runner.RawRunner(source, tmp_path, raw_runner.RawOptions(), Settings())
    cam = Rect(x=1000, y=0, w=280, h=208) if has_overlay else None
    layout = Layout(frame_w=FRAME_W, frame_h=FRAME_H, cam=cam)
    clip = RawClip(score=80, start_s=0, end_s=20, start_word=0, end_word=10, hook_title="x")
    assert runner._clip_mode(clip, layout) == expected


def test_forced_layout_wins(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from autoclip.config import Settings
    from autoclip.raw import runner as raw_runner
    from autoclip.raw.models import RawClip, SourceInfo

    source = SourceInfo(path="/v.mp4", duration_s=100, width=FRAME_W, height=FRAME_H)
    options = raw_runner.RawOptions(layout="blur")
    runner = raw_runner.RawRunner(source, tmp_path, options, Settings())
    clip = RawClip(score=80, start_s=0, end_s=20, start_word=0, end_word=10, hook_title="x")
    assert runner._clip_mode(clip, Layout(frame_w=FRAME_W, frame_h=FRAME_H)) == "blur"
