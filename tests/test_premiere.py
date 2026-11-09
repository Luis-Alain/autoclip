"""Premiere timeline (FCP7 xmeml) generation."""

from __future__ import annotations

import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from autoclip.raw.export_premiere import (
    AudioFormat,
    Rate,
    build_timeline,
    path_url,
    rate_for,
    write_timeline,
)
from autoclip.raw.models import ManifestRow

FIXTURE = Path(__file__).parent / "fixtures" / "premiere" / "timeline_30fps.xml"


def _row(rank: int, start: float, dur: float, title: str = "t", reason: str = "r") -> ManifestRow:
    return ManifestRow(
        rank=rank,
        file=f"clips/{rank:02d}_s90_x.mp4",
        abs_start=start,
        abs_end=start + dur,
        handle_start=start - 4,
        handle_end=start + dur + 4,
        in_point=4.0,
        out_point=4.0 + dur,
        duration=dur,
        score=90,
        hook_title=title,
        reason=reason,
        needs_context=False,
        transcript_excerpt="",
        language="es",
        source_path="/v.mp4",
    )


def _fixture_rows() -> list[ManifestRow]:
    return [
        _row(1, 100.0, 20.0, "¡Qué jugada!", "Hook fuerte & payoff <claro>"),
        _row(2, 300.0, 10.0, "Segundo", "r2"),
    ]


def _render(tree: ET.ElementTree) -> str:
    body = ET.tostring(tree.getroot(), encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n' + body + "\n"


# -- rates ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fps", "expected"),
    [
        (30.0, Rate(30, False)),
        (29.97, Rate(30, True)),
        (30000 / 1001, Rate(30, True)),
        (59.94, Rate(60, True)),
        (60.0, Rate(60, False)),
        (23.976, Rate(24, True)),
        (25.0, Rate(25, False)),
        (None, Rate(30, False)),
    ],
)
def test_rate_for(fps: float | None, expected: Rate) -> None:
    assert rate_for(fps) == expected


def test_ntsc_frames_use_the_real_rate() -> None:
    # One hour at 29.97 is 107892 frames, not 108000.
    assert Rate(30, True).frames(3600) == 107892
    assert Rate(30, False).frames(3600) == 108000


# -- structure --------------------------------------------------------------


def test_matches_fixture() -> None:
    tree = build_timeline(
        _fixture_rows(),
        Path("/clips/out"),
        name="Stream (autoclip)",
        width=1920,
        height=1080,
        fps=30.0,
        audio=AudioFormat(48_000, 2),
    )
    assert _render(tree) == FIXTURE.read_text(encoding="utf-8")


def test_clips_are_trimmed_to_in_out_with_handles_outside() -> None:
    tree = build_timeline(
        _fixture_rows(), Path("/c"), name="s", width=1920, height=1080, fps=30.0,
        audio=AudioFormat(),
    )  # fmt: skip
    items = tree.getroot().findall("./sequence/media/video/track/clipitem")
    first, second = items

    def num(el: ET.Element, tag: str) -> int:
        return int(el.findtext(tag) or "0")

    # Clip 1: 20 s chosen, 4 s handles each side, at 30 fps.
    assert (num(first, "start"), num(first, "end")) == (0, 600)
    assert (num(first, "in"), num(first, "out")) == (120, 720)
    assert num(first, "duration") == 840  # handles are in the media
    # Clip 2 starts after a one-second gap.
    assert num(second, "start") == 600 + 30
    assert num(tree.getroot().find("sequence"), "duration") == num(second, "end")  # type: ignore[arg-type]


def test_markers_on_clips_and_sequence() -> None:
    tree = build_timeline(
        _fixture_rows(), Path("/c"), name="s", width=1920, height=1080, fps=30.0,
        audio=AudioFormat(),
    )  # fmt: skip
    root = tree.getroot()
    clip_marker = root.find("./sequence/media/video/track/clipitem/marker")
    assert clip_marker is not None
    assert clip_marker.findtext("name") == "¡Qué jugada!"
    assert clip_marker.findtext("comment") == "Hook fuerte & payoff <claro>"
    seq_markers = root.findall("./sequence/marker")
    assert [m.findtext("name") for m in seq_markers] == ["01 ¡Qué jugada!", "02 Segundo"]
    assert [m.findtext("in") for m in seq_markers] == ["0", "630"]


def test_audio_items_reference_the_video_file_and_are_linked() -> None:
    tree = build_timeline(
        _fixture_rows(), Path("/c"), name="s", width=1920, height=1080, fps=30.0,
        audio=AudioFormat(44_100, 2),
    )  # fmt: skip
    root = tree.getroot()
    audio_items = root.findall("./sequence/media/audio/track/clipitem")
    assert len(audio_items) == 2
    file_ref = audio_items[0].find("file")
    assert file_ref is not None and file_ref.get("id") == "file-1" and len(file_ref) == 0
    links = [link.findtext("linkclipref") for link in audio_items[0].findall("link")]
    assert links == ["clipitem-v1", "clipitem-a1"]
    assert root.findtext(".//audio/format/samplecharacteristics/samplerate") == "44100"


def test_silent_sources_get_no_audio_track() -> None:
    tree = build_timeline(
        _fixture_rows(), Path("/c"), name="s", width=1920, height=1080, fps=30.0, audio=None
    )
    assert tree.getroot().find("./sequence/media/audio") is None


def test_ntsc_sequence_is_flagged() -> None:
    tree = build_timeline(
        _fixture_rows(), Path("/c"), name="s", width=1920, height=1080, fps=29.97,
        audio=None,
    )  # fmt: skip
    assert tree.getroot().findtext("./sequence/rate/ntsc") == "TRUE"


def test_path_url_encodes_spaces_and_accents() -> None:
    url = path_url(Path("/Users/luis/out/Mi Stream/01_s9_ñ.mp4"))
    assert url == "file://localhost/Users/luis/out/Mi%20Stream/01_s9_%C3%B1.mp4"


# -- real file ----------------------------------------------------------------


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_write_timeline_probes_audio_and_is_well_formed(tmp_path: Path) -> None:
    clip = tmp_path / "clips" / "01_s90_x.mp4"
    clip.parent.mkdir()
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc2=s=320x180:r=30:d=28", "-f", "lavfi",
         "-i", "sine=duration=28:sample_rate=44100", "-ac", "2", "-shortest", str(clip)],
        check=True,
    )  # fmt: skip
    path = write_timeline(
        [_row(1, 100.0, 20.0)], tmp_path, name="s", width=320, height=180, fps=30.0
    )
    text = path.read_text(encoding="utf-8")
    assert text.startswith('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>')
    root = ET.fromstring(text.split("\n", 2)[2])
    assert root.findtext(".//audio/format/samplecharacteristics/samplerate") == "44100"
    assert root.findtext(".//file/pathurl") == path_url(clip)
