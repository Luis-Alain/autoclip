"""Resolve the input (local file or URL) into a :class:`SourceInfo` and an output folder.

Local files are read where they are — never copied, since a stream VOD can be
many gigabytes. A URL is downloaded once (upstream's yt-dlp ingest); later runs
find the existing output folder for that VOD and reuse its download.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..config import Settings
from ..pipeline import ffmpeg, ingest
from .models import SourceInfo, read_model
from .slug import slugify


class SourceError(RuntimeError):
    """The input can't be used."""


def is_url(target: str) -> bool:
    return urlparse(target).scheme in ("http", "https")


def youtube_id(url: str) -> str:
    """The video id from youtube.com/watch?v=ID, youtu.be/ID, or /live/ID URLs."""
    parsed = urlparse(url)
    if parsed.hostname and parsed.hostname.endswith("youtu.be"):
        return parsed.path.strip("/")
    if query_id := parse_qs(parsed.query).get("v"):
        return query_id[0]
    return parsed.path.rstrip("/").rsplit("/", 1)[-1]


def folder_name(source: SourceInfo, date: str) -> str:
    """``<date>_<channel>_<vod-id>``, ASCII-safe."""
    channel = slugify(source.channel, max_length=30, fallback="local")
    vod = slugify(source.vod_id, max_length=40, fallback="vod")
    return f"{date}_{channel}_{vod}"


def find_existing(out_dir: Path, vod_id: str) -> Path | None:
    """The most recent output folder for ``vod_id`` whose source is still on disk."""
    suffix = "_" + slugify(vod_id, max_length=40, fallback="vod")
    matches = sorted(
        (p for p in out_dir.glob(f"*{suffix}") if (p / "source.json").exists()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for folder in matches:
        info: SourceInfo = read_model(folder / "source.json", SourceInfo)
        if Path(info.path).exists():
            return folder
    return None


def resolve(target: str, out_dir: Path, settings: Settings) -> tuple[SourceInfo, Path]:
    """Return the source and the output folder for it (existing or new)."""
    if is_url(target):
        return _from_url(target, out_dir, settings)
    return _from_file(Path(target).expanduser(), out_dir)


def _from_file(path: Path, out_dir: Path) -> tuple[SourceInfo, Path]:
    if not path.is_file():
        raise SourceError(f"{path} does not exist or is not a file.")

    vod_id = path.stem
    existing = find_existing(out_dir, vod_id)
    if existing:
        info: SourceInfo = read_model(existing / "source.json", SourceInfo)
        if Path(info.path).resolve() == path.resolve():
            return info, existing

    try:
        media = ffmpeg.probe(path)
    except ffmpeg.FFmpegError as exc:
        raise SourceError(f"ffprobe could not read {path.name}: {exc}") from exc
    if not media.has_video:
        raise SourceError(f"{path.name} has no video stream.")

    info = SourceInfo(
        path=str(path.resolve()),
        title=media.title or path.stem,
        channel="local",
        vod_id=vod_id,
        duration_s=media.duration_s,
        width=media.width,
        height=media.height,
        fps=media.fps,
        has_audio=media.has_audio,
        has_video=media.has_video,
    )
    # The file's date, not today's, so re-running another day lands in the same folder.
    date = dt.date.fromtimestamp(path.stat().st_mtime).isoformat()
    return info, out_dir / folder_name(info, date)


def _from_url(url: str, out_dir: Path, settings: Settings) -> tuple[SourceInfo, Path]:
    if not ingest.is_youtube_url(url):
        raise SourceError(
            "Only YouTube links are supported so far; Twitch and Kick links arrive in M5. "
            "For now, download the VOD and pass the file path."
        )
    vod_id = youtube_id(url)
    if existing := find_existing(out_dir, vod_id):
        return read_model(existing / "source.json", SourceInfo), existing

    try:
        source = ingest.ingest_youtube(url, settings.ingest)
    except ingest.IngestError as exc:
        raise SourceError(str(exc)) from exc

    info = SourceInfo(
        path=source.path,
        url=url,
        title=source.title,
        channel=source.channel or "youtube",
        vod_id=vod_id,
        duration_s=source.duration_s,
        width=source.width,
        height=source.height,
        fps=source.fps,
        has_audio=source.has_audio,
        has_video=source.has_video,
    )
    return info, out_dir / folder_name(info, dt.date.today().isoformat())
