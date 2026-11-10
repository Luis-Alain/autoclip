"""Start and watch an ``autoclip raw`` run from the web UI.

The run is the same pipeline as the CLI: local candidates, a language model
scoring each one, then 9:16 clips and a Premiere timeline. One run at a time,
because transcription and Gemma already fill the GPU.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import platform
import sys
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .. import paths
from ..config import load as load_settings
from ..pipeline import ingest
from ..raw.manifest import read_manifest
from ..raw.models import SourceInfo, read_model
from ..raw.runner import RawOptions, RawRunner, RunError
from ..raw.source import SourceError, resolve

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/raw", tags=["raw"])

UPLOAD_CHUNK = 4 * 1024 * 1024
Status = Literal["running", "done", "failed"]


class RawClipOut(BaseModel):
    rank: int
    file: str
    score: int
    hook_title: str
    reason: str
    layout: str
    duration: float
    url: str


class RawRunOut(BaseModel):
    id: str
    title: str
    created_at: str = ""
    status: Status
    progress: float = 0
    message: str = ""
    error: str | None = None
    provider: str = ""
    layout: str = "auto"
    folder: str = ""
    clips: list[RawClipOut] = Field(default_factory=list)
    timeline_url: str | None = None


class _Run:
    def __init__(self, run_id: str, title: str, provider: str, layout: str) -> None:
        self.id = run_id
        self.title = title
        self.status: Status = "running"
        self.progress = 0.0
        self.message = "En cola"
        self.error: str | None = None
        self.provider = provider
        self.layout = layout
        self.created_at = dt.datetime.now(dt.UTC).isoformat()
        self.folder: Path | None = None

    def view(self) -> RawRunOut:
        clips: list[RawClipOut] = []
        timeline: str | None = None
        folder_name = ""
        if self.folder is not None:
            folder_name = str(self.folder)
            manifest = self.folder / "manifest.json"
            if manifest.exists():
                for row in read_manifest(manifest):
                    clips.append(
                        RawClipOut(
                            rank=row.rank,
                            file=row.file,
                            score=row.score,
                            hook_title=row.hook_title,
                            reason=row.reason,
                            layout=row.layout,
                            duration=row.duration,
                            url=f"/api/raw/runs/{self.id}/file/{row.file}",
                        )
                    )
            if (self.folder / "timeline.xml").exists():
                timeline = f"/api/raw/runs/{self.id}/timeline"
        return RawRunOut(
            id=self.id,
            title=self.title,
            created_at=self.created_at,
            status=self.status,
            progress=self.progress,
            message=self.message,
            error=self.error,
            provider=self.provider,
            layout=self.layout,
            folder=folder_name,
            clips=clips,
            timeline_url=timeline,
        )

    def save(self) -> None:
        payload = self.view().model_dump_json(indent=2)
        path = _record_path(self.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)


_runs: dict[str, _Run] = {}
_active: str | None = None
_gate = asyncio.Lock()


def _runs_dir() -> Path:
    path = paths.root() / "raw" / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _output_dir() -> Path:
    path = paths.root() / "raw" / "out"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _uploads_dir() -> Path:
    path = paths.root() / "raw" / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _record_path(run_id: str) -> Path:
    return _runs_dir() / f"{run_id}.json"


def _default_provider() -> str:
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "mlx"
    settings = load_settings()
    return settings.active_provider


def _load(run_id: str) -> _Run | None:
    if run_id in _runs:
        return _runs[run_id]
    path = _record_path(run_id)
    if not path.exists():
        return None
    data = RawRunOut.model_validate_json(path.read_text(encoding="utf-8"))
    run = _Run(data.id, data.title, data.provider, data.layout)
    run.created_at = data.created_at or run.created_at
    run.status = data.status
    run.progress = data.progress
    run.message = data.message
    run.error = data.error
    if data.folder:
        run.folder = Path(data.folder)
    _runs[run_id] = run
    return run


def _list() -> list[RawRunOut]:
    found: dict[str, RawRunOut] = {}
    if _runs_dir().exists():
        for path in _runs_dir().glob("*.json"):
            try:
                found[path.stem] = RawRunOut.model_validate_json(path.read_text(encoding="utf-8"))
            except ValueError:
                log.warning("Skipping unreadable run record %s", path.name)
    for run_id, run in _runs.items():
        found[run_id] = run.view()
    return sorted(found.values(), key=lambda item: item.created_at, reverse=True)


@router.get("/runs", response_model=list[RawRunOut])
async def list_runs() -> list[RawRunOut]:
    return _list()


@router.get("/runs/{run_id}", response_model=RawRunOut)
async def get_run(run_id: str) -> RawRunOut:
    run = _load(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Esa corrida no existe.")
    return run.view()


@router.post("/runs", response_model=RawRunOut, status_code=201)
async def start_run(
    url: str = Form(""),
    lang: str = Form("es"),
    provider: str = Form(""),
    model: str = Form(""),
    layout: str = Form("auto"),
    top: int = Form(20),
    min_s: float = Form(15),
    max_s: float = Form(60),
    file: UploadFile | None = File(None),
) -> RawRunOut:
    global _active
    if layout not in ("auto", "split", "blur", "face"):
        raise HTTPException(
            status_code=400, detail="El encuadre debe ser auto, split, blur o face."
        )
    if min_s >= max_s:
        raise HTTPException(
            status_code=400,
            detail="La duración mínima tiene que ser menor que la máxima.",
        )
    if top < 1:
        raise HTTPException(status_code=400, detail="Pide al menos un clip.")

    target = url.strip()
    upload_path: Path | None = None
    if file is not None and file.filename:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in ingest.ACCEPTED_SUFFIXES:
            raise HTTPException(
                status_code=415,
                detail=f"Formato no soportado. Usa {', '.join(sorted(ingest.ACCEPTED_SUFFIXES))}.",
            )
        upload_path = _uploads_dir() / f"{uuid.uuid4().hex}{suffix}"
        with upload_path.open("wb") as handle:
            while chunk := await file.read(UPLOAD_CHUNK):
                handle.write(chunk)
        target = str(upload_path)
    if not target:
        raise HTTPException(status_code=400, detail="Pega un enlace o suelta un archivo.")

    async with _gate:
        if _active is not None:
            raise HTTPException(
                status_code=409,
                detail="Ya hay un video en proceso. Espera a que termine: la GPU va de uno en uno.",
            )
        chosen = provider.strip() or _default_provider()
        title = Path(file.filename).stem if file and file.filename else (target or "Video")
        run = _Run(uuid.uuid4().hex[:12], title, chosen, layout)
        _runs[run.id] = run
        _active = run.id
        run.save()

    asyncio.create_task(
        _execute(run, target, lang, chosen, model.strip(), layout, top, min_s, max_s),
        name=f"raw-run-{run.id}",
    )
    return run.view()


async def _execute(
    run: _Run,
    target: str,
    lang: str,
    provider: str,
    model: str,
    layout: str,
    top: int,
    min_s: float,
    max_s: float,
) -> None:
    global _active
    try:
        settings = load_settings()
        run.message = "Leyendo el video"
        run.save()
        try:
            source, folder = await asyncio.to_thread(resolve, target, _output_dir(), settings)
        except SourceError as exc:
            raise RunError(str(exc)) from exc
        run.folder = folder
        run.title = source.title or run.title
        run.message = "Transcribiendo"
        run.save()
        options = RawOptions(
            lang=lang or "es",
            top=top,
            min_s=min_s,
            max_s=max_s,
            provider=provider or None,
            model=model,
            layout=layout,  # type: ignore[arg-type]
        )

        def on_progress(fraction: float, message: str) -> None:
            run.progress = fraction
            run.message = message
            # The record is polled; writing every frame of progress is unnecessary.
            if int(fraction * 20) != int(max(0, fraction - 0.001) * 20):
                run.save()

        await RawRunner(source, folder, options, settings, on_progress=on_progress).run()
        run.status = "done"
        run.progress = 1
        run.message = "Listo"
        if (folder / "source.json").exists():
            info = read_model(folder / "source.json", SourceInfo)
            run.title = info.title or run.title
    except Exception as exc:
        log.exception("Raw run %s failed", run.id)
        run.status = "failed"
        run.error = str(exc)
        run.message = "Falló"
    finally:
        run.save()
        if _active == run.id:
            _active = None


def _safe_file(run: _Run, relative: str) -> Path:
    if run.folder is None:
        raise HTTPException(status_code=404, detail="Esta corrida todavía no tiene archivos.")
    folder = run.folder.resolve()
    path = (folder / relative).resolve()
    if path != folder and folder not in path.parents:
        raise HTTPException(status_code=404, detail="Archivo no encontrado.")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Archivo no encontrado.")
    return path


@router.get("/runs/{run_id}/file/{file_path:path}")
async def run_file(run_id: str, file_path: str) -> FileResponse:
    run = _load(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Esa corrida no existe.")
    # Only the rendered clips and the timeline are part of the UI. Source
    # video and work files stay on disk.
    if not (file_path.startswith("clips/") or file_path == "timeline.xml"):
        raise HTTPException(status_code=404, detail="Archivo no encontrado.")
    return FileResponse(_safe_file(run, file_path))


@router.get("/runs/{run_id}/timeline")
async def run_timeline(run_id: str) -> FileResponse:
    run = _load(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Esa corrida no existe.")
    return FileResponse(_safe_file(run, "timeline.xml"), filename="timeline.xml")
