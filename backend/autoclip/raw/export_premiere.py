"""``timeline.xml``: a Final Cut Pro 7 XML (xmeml) sequence that Premiere imports.

Premiere Pro reads xmeml through File → Import. The sequence holds every raw
clip in rank order with a one-second gap between them. Each clip item is
trimmed to the chosen moment (``in``/``out``), while the handles stay in the
media outside the edit points, so a clip can be extended by dragging its edge.

Each clip carries a marker named after ``hook_title`` with ``reason`` as the
comment; the same marker is placed on the sequence ruler, where Premiere shows
sequence markers most reliably.

All times in xmeml are integer frame counts at the sequence's ``timebase``.
NTSC rates (29.97, 59.94, 23.976) are written as timebase 30/60/24 with
``<ntsc>TRUE</ntsc>``; frame counts then follow the real (1000/1001) rate.
"""

from __future__ import annotations

import json
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from ..pipeline.ffmpeg import ffprobe_path
from .models import ManifestRow

#: Gap between consecutive clips on the timeline.
GAP_S = 1.0


@dataclass(frozen=True)
class Rate:
    """A frame rate as xmeml expresses it."""

    timebase: int
    ntsc: bool

    @property
    def fps(self) -> float:
        return self.timebase * 1000 / 1001 if self.ntsc else float(self.timebase)

    def frames(self, seconds: float) -> int:
        return round(seconds * self.fps)


def rate_for(fps: float | None) -> Rate:
    """Map a measured fps to xmeml's (timebase, ntsc). Defaults to 30."""
    if not fps or fps <= 0:
        return Rate(30, False)
    timebase = round(fps)
    # 29.97 rounds to 30 but is ~0.03 slower: that difference is the NTSC flag.
    ntsc = abs(fps - timebase * 1000 / 1001) < abs(fps - timebase)
    return Rate(timebase, ntsc)


@dataclass(frozen=True)
class AudioFormat:
    sample_rate: int = 48_000
    channels: int = 2


def probe_audio(path: Path) -> AudioFormat | None:
    """Sample rate and channel count of the first audio stream, or None if silent."""
    proc = subprocess.run(
        [
            ffprobe_path(),
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    streams = (json.loads(proc.stdout or "{}") or {}).get("streams") or []
    if not streams:
        return None
    return AudioFormat(
        sample_rate=int(streams[0].get("sample_rate") or 48_000),
        channels=int(streams[0].get("channels") or 2),
    )


def path_url(path: Path) -> str:
    """``file://localhost/Users/luis/My%20Clip.mp4`` — the form Premiere resolves."""
    return "file://localhost" + quote(str(path.resolve()), safe="/")


def build_timeline(
    rows: list[ManifestRow],
    folder: Path,
    *,
    name: str,
    width: int,
    height: int,
    fps: float | None,
    audio: AudioFormat | None,
) -> ET.ElementTree:
    """Build the xmeml tree. Pure apart from resolving clip paths under ``folder``.

    Preconditions:
        every row's ``file`` exists under ``folder``; rows are in rank order.
    """
    rate = rate_for(fps)
    gap = rate.frames(GAP_S)

    root = ET.Element("xmeml", version="4")
    sequence = ET.SubElement(root, "sequence", id="sequence-1")
    _text(sequence, "name", name)
    duration_el = ET.SubElement(sequence, "duration")
    _rate(sequence, rate)

    media = ET.SubElement(sequence, "media")
    video = ET.SubElement(media, "video")
    video_format = ET.SubElement(video, "format")
    _video_characteristics(video_format, rate, width, height)
    video_track = ET.SubElement(video, "track")

    audio_track = None
    if audio is not None:
        audio_el = ET.SubElement(media, "audio")
        _text(audio_el, "numOutputChannels", str(audio.channels))
        audio_format = ET.SubElement(audio_el, "format")
        _audio_characteristics(audio_format, audio)
        audio_track = ET.SubElement(audio_el, "track")

    cursor = 0
    for index, row in enumerate(rows, start=1):
        file_path = folder / row.file
        file_frames = rate.frames(row.handle_end - row.handle_start)
        in_frame = min(rate.frames(row.in_point), file_frames - 1)
        out_frame = min(max(rate.frames(row.out_point), in_frame + 1), file_frames)
        start, end = cursor, cursor + (out_frame - in_frame)

        video_id, audio_id, file_id = f"clipitem-v{index}", f"clipitem-a{index}", f"file-{index}"

        item = _clipitem(video_track, video_id, file_path.name, rate, file_frames,
                         start, end, in_frame, out_frame)  # fmt: skip
        file_el = ET.SubElement(item, "file", id=file_id)
        _text(file_el, "name", file_path.name)
        _text(file_el, "pathurl", path_url(file_path))
        _rate(file_el, rate)
        _text(file_el, "duration", str(file_frames))
        file_media = ET.SubElement(file_el, "media")
        _video_characteristics(ET.SubElement(file_media, "video"), rate, width, height)
        if audio is not None:
            file_audio = ET.SubElement(file_media, "audio")
            _audio_characteristics(file_audio, audio)
            _text(file_audio, "channelcount", str(audio.channels))

        # Clip marker, in source-media frames (inside the handles).
        _marker(item, row.hook_title, row.reason, in_frame)

        if audio_track is not None:
            audio_item = _clipitem(audio_track, audio_id, file_path.name, rate, file_frames,
                                   start, end, in_frame, out_frame)  # fmt: skip
            ET.SubElement(audio_item, "file", id=file_id)  # reference to the file above
            source = ET.SubElement(audio_item, "sourcetrack")
            _text(source, "mediatype", "audio")
            _text(source, "trackindex", "1")
            # Link video and audio so they select and move together.
            for parent in (item, audio_item):
                _link(parent, video_id, "video", index)
                _link(parent, audio_id, "audio", index)

        # Sequence marker on the ruler at the clip's start.
        _marker(sequence, f"{row.rank:02d} {row.hook_title}", row.reason, start)
        cursor = end + gap

    duration_el.text = str(max(0, cursor - gap))
    ET.indent(root, space="  ")
    return ET.ElementTree(root)


def write_timeline(
    rows: list[ManifestRow],
    folder: Path,
    *,
    name: str,
    width: int,
    height: int,
    fps: float | None,
) -> Path:
    """Write ``timeline.xml`` next to the clips and return its path."""
    audio = probe_audio(folder / rows[0].file) if rows else None
    tree = build_timeline(rows, folder, name=name, width=width, height=height, fps=fps, audio=audio)
    path = folder / "timeline.xml"
    body = ET.tostring(tree.getroot(), encoding="unicode")
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n' + body + "\n",
        encoding="utf-8",
    )
    return path


# -- element helpers ----------------------------------------------------------


def _text(parent: ET.Element, tag: str, value: str) -> ET.Element:
    element = ET.SubElement(parent, tag)
    element.text = value
    return element


def _rate(parent: ET.Element, rate: Rate) -> None:
    element = ET.SubElement(parent, "rate")
    _text(element, "timebase", str(rate.timebase))
    _text(element, "ntsc", "TRUE" if rate.ntsc else "FALSE")


def _video_characteristics(parent: ET.Element, rate: Rate, width: int, height: int) -> None:
    sc = ET.SubElement(parent, "samplecharacteristics")
    _rate(sc, rate)
    _text(sc, "width", str(width))
    _text(sc, "height", str(height))
    _text(sc, "anamorphic", "FALSE")
    _text(sc, "pixelaspectratio", "square")
    _text(sc, "fielddominance", "none")


def _audio_characteristics(parent: ET.Element, audio: AudioFormat) -> None:
    sc = ET.SubElement(parent, "samplecharacteristics")
    _text(sc, "depth", "16")
    _text(sc, "samplerate", str(audio.sample_rate))


def _clipitem(
    track: ET.Element,
    item_id: str,
    name: str,
    rate: Rate,
    file_frames: int,
    start: int,
    end: int,
    in_frame: int,
    out_frame: int,
) -> ET.Element:
    item = ET.SubElement(track, "clipitem", id=item_id)
    _text(item, "name", name)
    _text(item, "enabled", "TRUE")
    _text(item, "duration", str(file_frames))
    _rate(item, rate)
    _text(item, "start", str(start))
    _text(item, "end", str(end))
    _text(item, "in", str(in_frame))
    _text(item, "out", str(out_frame))
    return item


def _marker(parent: ET.Element, name: str, comment: str, frame: int) -> None:
    marker = ET.SubElement(parent, "marker")
    _text(marker, "name", name)
    _text(marker, "comment", comment)
    _text(marker, "in", str(frame))
    _text(marker, "out", "-1")


def _link(parent: ET.Element, item_id: str, mediatype: str, clip_index: int) -> None:
    link = ET.SubElement(parent, "link")
    _text(link, "linkclipref", item_id)
    _text(link, "mediatype", mediatype)
    _text(link, "trackindex", "1")
    _text(link, "clipindex", str(clip_index))
