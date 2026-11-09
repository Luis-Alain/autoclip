"""The stream-clips pipeline, stage by stage, with every artifact in one output folder.

    <out>/<date>_<channel>_<vod-id>/
      source.json          the VOD's metadata
      layout_detect.jpg    the detected webcam (green) and crops, on a source frame
      layout_preview.jpg   one 9:16 split frame, to check the crops by eye
      work/                audio.wav, silences.json, layout.json, crops, captions
      transcript/full.json word-level transcript (absolute seconds)
      highlights.json      the ranked selection + the params that produced it
      clips/               9:16, speaker-tracked, burned-in captions
      manifest.json/.csv   one row per clip
      timeline.xml         Premiere sequence of the clips (FCP7 xmeml)
      logs/                run log, including every ffmpeg command

Resuming: each stage is skipped when its artifact exists. Changing the
selection params (--top/--min/--max/provider/model) recomputes highlights;
clips are skipped individually when already rendered.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ..config import Settings
from ..pipeline import captions, prepare, transcribe
from ..pipeline.prepare import Silence
from ..pipeline.transcript import Transcript
from ..providers import build_provider
from ..system import report
from .export_premiere import write_timeline
from .export_split import export_composite_clip, render_detection, render_preview
from .export_vertical import clip_filename, export_vertical_clip, is_vertical_file
from .layout import Layout, Rect, clip_shows_cam, clip_shows_face, detect_layout
from .manifest import excerpt, write_manifest
from .models import HighlightsArtifact, ManifestRow, RawClip, SourceInfo, read_model, write_model
from .select import PROMPT_VERSION, select_clips

log = logging.getLogger(__name__)

#: Share of the progress bar per stage — rough wall-time proportions on an M4.
WEIGHTS = {
    "prepare": 0.05,
    "transcribe": 0.35,
    "highlights": 0.20,
    "layout": 0.05,
    "clips": 0.35,
}

LayoutMode = Literal["auto", "split", "blur", "face"]

#: (overall 0..1, message)
ProgressFn = Callable[[float, str], None]


class RunError(RuntimeError):
    """The run can't proceed; the message says why and what to do."""


@dataclass
class RawOptions:
    lang: str = "es"
    top: int = 20
    min_s: float = 15.0
    max_s: float = 60.0
    provider: str | None = None
    style: str = ""
    #: auto: split where the webcam overlay is on screen, blur for gameplay
    #: without it, face crop for talking-head scenes. Or force one of them.
    layout: LayoutMode = "auto"
    #: Manual webcam rectangle in source pixels; skips detection.
    cam: Rect | None = None


@dataclass
class RawResult:
    folder: Path
    clips: list[RawClip]
    rows: list[ManifestRow]
    warnings: list[str] = field(default_factory=list)


class RawRunner:
    def __init__(
        self,
        source: SourceInfo,
        folder: Path,
        options: RawOptions,
        settings: Settings,
        *,
        on_progress: ProgressFn | None = None,
    ) -> None:
        self.source = source
        self.folder = folder.resolve()
        self.options = options
        self.settings = settings
        self.on_progress = on_progress
        self._done = 0.0
        self.warnings: list[str] = []

        # "auto" means let Whisper detect; the language check still guards a fixed one.
        settings.whisper.language = "" if options.lang == "auto" else options.lang

    # -- paths ---------------------------------------------------------------

    @property
    def work(self) -> Path:
        return self.folder / "work"

    @property
    def source_path(self) -> Path:
        return Path(self.source.path)

    # -- progress ------------------------------------------------------------

    def _report(self, stage: str, fraction: float, message: str) -> None:
        if self.on_progress:
            overall = self._done + WEIGHTS[stage] * max(0.0, min(1.0, fraction))
            self.on_progress(min(1.0, overall), message)

    def _finish(self, stage: str) -> None:
        self._done += WEIGHTS[stage]

    # -- run -----------------------------------------------------------------

    async def run(self) -> RawResult:
        if not report().ffmpeg.has_libass:
            # Checked up front: captions are part of every clip, so without
            # libass there is no output — better to fail before transcribing.
            raise RunError(
                "This ffmpeg has no libass, so captions can't be burned in. "
                "Install ffmpeg-full (see `autoclip doctor`)."
            )

        self.folder.mkdir(parents=True, exist_ok=True)
        write_model(self.folder / "source.json", self.source)
        log.info("Output folder: %s", self.folder)

        audio = self._prepare()
        transcript = self._transcribe(audio)
        silences = self._silences(audio)
        clips = await self._highlights(transcript, silences)
        layout = self._layout()
        rows = self._export(clips, transcript, layout)
        write_manifest(rows, self.folder)
        self._timeline(rows)
        self._report("clips", 1.0, "Done")
        return RawResult(folder=self.folder, clips=clips, rows=rows, warnings=self.warnings)

    # -- stages --------------------------------------------------------------

    def _prepare(self) -> Path:
        audio = self.work / "audio.wav"
        if audio.exists() and audio.stat().st_size > 0:
            log.info("Reusing %s", audio)
        else:
            self._report("prepare", 0.0, "Extracting audio")
            partial = audio.with_name(".partial-audio.wav")
            prepare.extract_audio(
                self.source_path,
                partial,
                duration_s=self.source.duration_s,
                on_progress=lambda f: self._report("prepare", f, "Extracting audio"),
            )
            partial.replace(audio)
        self._finish("prepare")
        return audio

    def _transcribe(self, audio: Path) -> Transcript:
        path = self.folder / "transcript" / "full.json"
        if path.exists():
            log.info("Reusing %s", path)
            transcript = Transcript.load(path)
        else:
            self._report("transcribe", 0.0, "Transcribing")
            transcript = transcribe.transcribe(
                audio,
                self.settings.whisper,
                duration_s=self.source.duration_s,
                on_progress=lambda f: self._report("transcribe", f, "Transcribing"),
            )
            transcript.save(path)
        expected = self.settings.whisper.language
        if expected and transcript.language and transcript.language != expected:
            self.warnings.append(
                f"The audio is in '{transcript.language}', not '{expected}'. It was "
                f"transcribed as '{transcript.language}', and titles are written in it."
            )
        self._finish("transcribe")
        return transcript

    def _silences(self, audio: Path) -> list[Silence]:
        path = self.work / "silences.json"
        if path.exists():
            return [Silence(**s) for s in json.loads(path.read_text(encoding="utf-8"))]
        silences = prepare.detect_silences(audio)
        path.write_text(json.dumps([{"start": s.start, "end": s.end} for s in silences]))
        return silences

    async def _highlights(self, transcript: Transcript, silences: list[Silence]) -> list[RawClip]:
        path = self.folder / "highlights.json"
        provider = build_provider(self.options.provider or None, self.settings)
        params = {
            "provider": provider.name,
            "model": provider.model,
            "prompt": PROMPT_VERSION,
            "top": self.options.top,
            "min_s": self.options.min_s,
            "max_s": self.options.max_s,
        }

        if path.exists():
            saved: HighlightsArtifact = read_model(path, HighlightsArtifact)
            if saved.params == params:
                log.info("Reusing %s", path)
                self._finish("highlights")
                return saved.clips
            log.info("Selection params changed; recomputing highlights.")

        self._report("highlights", 0.0, f"Finding moments with {provider.name}")
        clips = await select_clips(
            transcript,
            provider,
            min_s=self.options.min_s,
            max_s=self.options.max_s,
            top=self.options.top,
            silences=silences,
            on_progress=lambda f: self._report(
                "highlights", f, f"Finding moments with {provider.name}"
            ),
        )
        write_model(
            path, HighlightsArtifact(params=params, language=transcript.language, clips=clips)
        )
        self._finish("highlights")
        return clips

    def _layout(self) -> Layout:
        """Where the webcam overlay is (if any). Cached in work/layout.json.

        Precedence: --cam (saved, and reused by later runs) > cached result >
        fresh detection. --layout face skips all of it without touching the cache.
        """
        path = self.work / "layout.json"
        info = self.source
        width, height = info.width or 1920, info.height or 1080

        if self.options.layout in ("face", "blur"):
            # Neither needs the webcam position.
            self._finish("layout")
            return Layout(frame_w=width, frame_h=height)

        if self.options.cam is not None:
            cam = self.options.cam
            layout = Layout(
                frame_w=width,
                frame_h=height,
                cam=cam,
                face_cx=cam.x + cam.w / 2,
                face_cy=cam.y + cam.h / 2,
                method="manual",
            )
            write_model(path, layout)
        elif path.exists():
            log.info("Reusing %s", path)
            layout = read_model(path, Layout)
        else:
            self._report("layout", 0.0, "Finding the webcam")
            layout = detect_layout(
                self.source_path, duration_s=info.duration_s, frame_w=width, frame_h=height
            )
            write_model(path, layout)

        if layout.cam is not None:
            at = info.duration_s * 0.3
            render_detection(self.source_path, layout, at, self.folder / "layout_detect.jpg")
            render_preview(self.source_path, layout, at, self.folder / "layout_preview.jpg")
        elif self.options.layout == "split":
            raise RunError(
                "--layout split needs a webcam overlay, but none was found. Pass its "
                "position with --cam x,y,w,h (source pixels), or use --layout face."
            )
        self._finish("layout")
        return layout

    def _clip_mode(self, clip: RawClip, layout: Layout) -> str:
        """Pick split, blur or face for one clip (see RawOptions.layout)."""
        if self.options.layout != "auto":
            return self.options.layout
        if layout.cam is not None:
            # A gameplay stream with a webcam overlay: split while the webcam is
            # on screen, blur for moments where it isn't (scene switches).
            shown = clip_shows_cam(self.source_path, layout, clip.start_s, clip.end_s)
            return "split" if shown else "blur"
        # No overlay: a camera scene gets the face crop, pure gameplay the blur.
        talking = clip_shows_face(self.source_path, layout, clip.start_s, clip.end_s)
        return "face" if talking else "blur"

    def _export(
        self, clips: list[RawClip], transcript: Transcript, layout: Layout
    ) -> list[ManifestRow]:
        style = captions.get_style(self.options.style or self.settings.export.caption_style)
        export_settings = self.settings.export.model_copy(update={"ratio": "9:16"})

        # Which layout each rendered file was made with, so a file is re-rendered
        # when its layout changes between runs (e.g. --layout face -> auto).
        rendered_path = self.work / "rendered.json"
        rendered: dict[str, str] = (
            json.loads(rendered_path.read_text(encoding="utf-8")) if rendered_path.exists() else {}
        )

        rows: list[ManifestRow] = []
        for index, clip in enumerate(clips):
            name = clip_filename(clip.rank, clip.score, clip.hook_title)
            destination = self.folder / "clips" / name
            mode = self._clip_mode(clip, layout)
            message = f"Rendering clip {index + 1}/{len(clips)} ({mode})"

            def progress(f: float, i: int = index, m: str = message) -> None:
                self._report("clips", (i + f) / len(clips), m)

            if (
                destination.exists()
                and rendered.get(name, "face") == mode
                and is_vertical_file(destination)
            ):
                log.info("Reusing %s", destination)
            elif mode in ("split", "blur"):
                export_composite_clip(
                    self.source_path, destination, clip, transcript, layout,
                    mode=mode,  # type: ignore[arg-type]
                    style=style, settings=export_settings, work_dir=self.work,
                    on_progress=progress,
                )  # fmt: skip
            else:
                export_vertical_clip(
                    self.source_path, destination, clip, transcript,
                    style=style, settings=export_settings, work_dir=self.work,
                    on_progress=progress,
                )  # fmt: skip
            rendered[name] = mode
            rendered_path.write_text(json.dumps(rendered, indent=2), encoding="utf-8")

            rows.append(
                ManifestRow(
                    rank=clip.rank,
                    file=f"clips/{name}",
                    abs_start=round(clip.start_s, 3),
                    abs_end=round(clip.end_s, 3),
                    duration=round(clip.duration_s, 3),
                    score=clip.score,
                    hook_title=clip.hook_title,
                    reason=clip.reason,
                    needs_context=clip.needs_context,
                    layout=mode,
                    transcript_excerpt=excerpt(
                        transcript.text_between(clip.start_word, clip.end_word)
                    ),
                    language=transcript.language,
                    source_url=self.source.url,
                    source_path=self.source.path,
                )
            )
        self._finish("clips")
        return rows

    def _timeline(self, rows: list[ManifestRow]) -> None:
        """Premiere timeline of the clips. Cheap, so always regenerated."""
        if rows:
            write_timeline(
                rows, self.folder, name=f"{self.source.title or self.source.vod_id} (autoclip)"
            )
