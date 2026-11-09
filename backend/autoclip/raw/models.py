"""Schemas for the raw pipeline's on-disk artifacts.

Pydantic models rather than dicts so every JSON file is validated when it is
read back on a resumed run — a hand-edited or truncated artifact fails loudly
instead of producing wrong timestamps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SourceInfo(BaseModel):
    """The VOD being clipped. Written to ``source.json``."""

    path: str
    url: str | None = None
    title: str = ""
    channel: str = "local"
    vod_id: str = ""
    duration_s: float
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    has_audio: bool = True
    has_video: bool = True


class RawClip(BaseModel):
    """One selected moment, in absolute VOD seconds and transcript word indices."""

    rank: int = 0
    score: int
    start_s: float
    end_s: float
    start_word: int
    end_word: int
    hook_title: str
    hook: str = ""
    reason: str = ""
    needs_context: bool = False

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class HighlightsArtifact(BaseModel):
    """``highlights.json``: the selection plus the parameters that produced it.

    The parameters are stored so a re-run with different ``--top/--min/--max``
    or provider recomputes the selection instead of reusing a stale one.
    """

    params: dict[str, Any]
    language: str = ""
    clips: list[RawClip] = Field(default_factory=list)


class ManifestRow(BaseModel):
    """One line of ``manifest.json`` / ``manifest.csv``.

    ``file`` is the 9:16 clip, relative to the output folder. ``abs_*`` are
    seconds in the source VOD; the clip file spans exactly that range.
    """

    rank: int
    file: str
    abs_start: float
    abs_end: float
    duration: float
    score: int
    hook_title: str
    reason: str
    needs_context: bool
    chat_score: float | None = None
    transcript_excerpt: str
    language: str
    source_url: str | None = None
    source_path: str
    campaign: str | None = None
    credit_text: str | None = None


def read_model(path: Path, model: type[BaseModel]) -> Any:
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def write_model(path: Path, value: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)  # atomic: a crash never leaves a half-written artifact
