"""Transcribe stage — audio to word-level timestamps.

Two interchangeable engines produce the same :class:`Transcript`:

- **mlx** — mlx-whisper on the Apple Silicon GPU. The default on arm64 Macs.
- **faster-whisper** — CTranslate2 on CUDA or CPU. The default everywhere else.

WhisperX (optional) adds speaker labels. Word-level timing is non-negotiable:
clip boundaries, caption animation, and trim-handle snapping all derive from it.
"""

from __future__ import annotations

import logging
import platform
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np

from ..config import WhisperSettings
from ..system import GPUInfo, report
from .transcript import Segment, Transcript, Word

log = logging.getLogger(__name__)


class TranscriptionError(RuntimeError):
    """Transcription could not be completed."""


def resolve_compute(settings: WhisperSettings, gpu: GPUInfo | None = None) -> tuple[str, str]:
    """Return the ``(device, compute_type)`` to run Whisper with.

    An explicit ``settings.compute_type`` wins, but only if the device actually
    supports it. An override the backend will reject fails at model load with a
    message that points at cuDNN rather than at the setting, so an unsupported
    override is warned about and ignored rather than honoured into a crash.
    """
    gpu = gpu if gpu is not None else report().gpu
    automatic = gpu.compute_type

    override = settings.compute_type
    if not override:
        return gpu.device, automatic

    supported = gpu.supported_compute_types
    if supported and override not in supported:
        log.warning(
            "whisper.compute_type is set to %r, which %s does not support "
            "(supported: %s). Using %r instead.",
            override,
            gpu.device,
            ", ".join(sorted(supported)),
            automatic,
        )
        return gpu.device, automatic

    return gpu.device, override


def _model_load_error(exc: Exception, device: str, compute_type: str) -> TranscriptionError:
    """Turn a Whisper model-load failure into something actionable.

    The two causes look identical from the outside and need opposite fixes, so
    they're distinguished rather than lumped under one guess.
    """
    message = str(exc)
    lowered = message.lower()

    if "compute type" in lowered or "not support" in lowered:
        supported = report().gpu.supported_compute_types
        options = ", ".join(sorted(supported)) if supported else "unknown"
        return TranscriptionError(
            f"This device does not support the '{compute_type}' compute type.\n"
            f"Supported on {device}: {options}.\n\n"
            "Clear whisper.compute_type in config.json to let AutoClip choose, "
            "or set it to one of the supported values."
        )

    if device == "cuda" and ("cudnn" in lowered or "library" in lowered or "dll" in lowered):
        return TranscriptionError(
            f"Could not load Whisper on the GPU: {message}\n\n"
            "This is a missing CUDA runtime library. Install it with "
            "`uv pip install 'autoclip[gpu]'`, then retry — the job resumes "
            "from this stage."
        )

    return TranscriptionError(f"Could not load the Whisper model: {message}")


Backend = Literal["faster-whisper", "mlx"]


def mlx_supported() -> bool:
    """True on an Apple Silicon Mac with mlx-whisper installed."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    from ..system import _module_available

    return _module_available("mlx_whisper")


def resolve_backend(settings: WhisperSettings) -> Backend:
    """Pick the transcription engine for these settings on this machine."""
    if settings.backend == "mlx":
        if not mlx_supported():
            raise TranscriptionError(
                "whisper.backend is 'mlx', but mlx-whisper is not available here. "
                "It needs an Apple Silicon Mac: run `uv pip install mlx-whisper`, or set "
                "`autoclip config set whisper.backend auto`."
            )
        return "mlx"
    if settings.backend == "faster-whisper":
        return "faster-whisper"
    return "mlx" if mlx_supported() else "faster-whisper"


def transcribe(
    audio: Path,
    settings: WhisperSettings | None = None,
    *,
    duration_s: float | None = None,
    on_progress: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Transcript:
    """Transcribe an audio file into a :class:`Transcript` with word timings.

    Preconditions:
        audio is a decodable audio file, ideally 16 kHz mono as produced by
        :func:`autoclip.pipeline.prepare.extract_audio`.
    """
    settings = settings or WhisperSettings()
    backend = resolve_backend(settings)

    language = settings.language
    if language and settings.verify_language:
        language = verify_language(
            audio, language, backend=backend, settings=settings, duration_s=duration_s
        )

    if backend == "mlx":
        return _transcribe_mlx(
            audio,
            settings,
            language=language,
            duration_s=duration_s,
            on_progress=on_progress,
            cancelled=cancelled,
        )
    return _transcribe_faster_whisper(
        audio,
        settings,
        language=language,
        duration_s=duration_s,
        on_progress=on_progress,
        cancelled=cancelled,
    )


# --------------------------------------------------------------------------
# Language verification
# --------------------------------------------------------------------------

#: Seconds of audio per language sample, and how many samples to take.
LANGUAGE_SAMPLE_S = 30
LANGUAGE_SAMPLES = 3


def verify_language(
    audio: Path,
    expected: str,
    *,
    backend: Backend,
    settings: WhisperSettings,
    duration_s: float | None = None,
) -> str:
    """Return the language to transcribe in, after checking the audio.

    Forcing Whisper into the wrong language doesn't fail: it quietly
    *translates*, producing fluent subtitles that don't match what is said. So
    a few samples spread across the file are detected first, and if most of
    them agree on a different language, that one is used and a warning logged.
    """
    total = duration_s or _audio_duration(audio)
    if not total:
        return expected

    starts = _sample_starts(total, LANGUAGE_SAMPLES, LANGUAGE_SAMPLE_S)
    try:
        detected = [
            _detect_language(_read_pcm(audio, start, LANGUAGE_SAMPLE_S), backend, settings)
            for start in starts
        ]
    except Exception as exc:  # detection is a safety net, never a blocker
        log.warning("Language check failed (%s); transcribing as %r.", exc, expected)
        return expected

    detected = [code for code in detected if code]
    if not detected:
        return expected

    winner, votes = Counter(detected).most_common(1)[0]
    if winner != expected and votes > len(detected) / 2:
        log.warning(
            "whisper.language is %r but the audio is %r (%d of %d samples). "
            "Transcribing as %r — forcing %r would translate instead of transcribe.",
            expected,
            winner,
            votes,
            len(detected),
            winner,
            expected,
        )
        return winner
    return expected


def _sample_starts(total_s: float, count: int, length_s: float) -> list[float]:
    """Evenly spaced sample start times, skipping the first and last 10%."""
    if total_s <= length_s:
        return [0.0]
    lo, hi = total_s * 0.1, max(total_s * 0.9 - length_s, total_s * 0.1)
    if count == 1:
        return [lo]
    step = (hi - lo) / (count - 1)
    return [lo + i * step for i in range(count)]


def _read_pcm(audio: Path, start_s: float, length_s: float) -> np.ndarray:
    """Decode a slice of ``audio`` to 16 kHz mono float32, as Whisper expects."""
    from .ffmpeg import ffmpeg_path

    proc = subprocess.run(
        [
            ffmpeg_path(),
            "-nostdin",
            "-loglevel",
            "error",
            "-ss",
            f"{start_s:.3f}",
            "-t",
            f"{length_s:.3f}",
            "-i",
            str(audio),
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-",
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def _audio_duration(audio: Path) -> float:
    from .ffmpeg import probe

    try:
        return probe(audio).duration_s or 0.0
    except Exception:
        return 0.0


def _detect_language(samples: np.ndarray, backend: Backend, settings: WhisperSettings) -> str:
    if samples.size == 0:
        return ""
    if backend == "mlx":
        import mlx_whisper

        # Decoding 30 s with language=None runs Whisper's own language ID first.
        result = mlx_whisper.transcribe(
            samples,
            path_or_hf_repo=settings.mlx_model,
            verbose=None,
            condition_on_previous_text=False,
        )
        return str(result.get("language") or "")

    from faster_whisper import WhisperModel

    device, compute_type = resolve_compute(settings)
    model = WhisperModel(settings.model, device=device, compute_type=compute_type)
    language, _probability, _ = model.detect_language(samples)
    return language


# --------------------------------------------------------------------------
# MLX (Apple Silicon GPU)
# --------------------------------------------------------------------------


def _transcribe_mlx(
    audio: Path,
    settings: WhisperSettings,
    *,
    language: str,
    duration_s: float | None = None,
    on_progress: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Transcript:
    """Transcribe on the Apple Silicon GPU with mlx-whisper.

    The model is downloaded from Hugging Face on first use and cached.
    """
    import mlx_whisper

    log.info("Transcribing with mlx-whisper model=%s (Apple GPU)", settings.mlx_model)

    try:
        with _mlx_progress(on_progress, cancelled):
            result = mlx_whisper.transcribe(
                str(audio),
                path_or_hf_repo=settings.mlx_model,
                language=language or None,
                word_timestamps=True,
                # Stream audio (game sound, music) makes Whisper loop on its own
                # previous output; not conditioning on it trades a little
                # consistency for far fewer repeated-phrase hallucinations.
                condition_on_previous_text=False,
                # With word timestamps, skip text hallucinated into long silences.
                hallucination_silence_threshold=2.0,
                verbose=False,
            )
    except _MlxCancelled as exc:
        raise TranscriptionError("Transcription cancelled.") from exc
    except Exception as exc:
        raise TranscriptionError(f"mlx-whisper failed: {exc}") from exc

    transcript = _transcript_from_whisper_dict(result, model=settings.mlx_model)
    transcript.language = transcript.language or language

    if not transcript.words:
        raise TranscriptionError(
            "No speech was detected in this audio. Check that the correct audio track "
            "was extracted."
        )
    if on_progress:
        on_progress(1.0)

    log.info(
        "Transcribed %d words in %d segments.", len(transcript.words), len(transcript.segments)
    )
    return transcript


def _transcript_from_whisper_dict(result: dict[str, Any], *, model: str) -> Transcript:
    """Convert openai-whisper-style output (which mlx-whisper returns) to a Transcript."""
    transcript = Transcript(language=str(result.get("language") or ""), model=model)

    for segment in result.get("segments") or []:
        first_word = len(transcript.words)
        for word in segment.get("words") or []:
            text = str(word.get("word") or "").strip()
            if not text:
                continue
            transcript.words.append(
                Word(text=text, start=float(word["start"]), end=float(word["end"]))
            )
        last_word = max(first_word, len(transcript.words) - 1)
        transcript.segments.append(
            Segment(
                text=str(segment.get("text") or "").strip(),
                start=float(segment.get("start", 0.0)),
                end=float(segment.get("end", 0.0)),
                first_word=first_word,
                last_word=last_word,
            )
        )
    return transcript


class _MlxCancelled(RuntimeError):
    pass


class _mlx_progress:  # noqa: N801 - used as a context manager, reads like one
    """Route mlx-whisper's tqdm progress bar into our progress callback.

    mlx-whisper has no progress or cancellation hooks, but it advances a tqdm
    bar (in audio frames) as it decodes. Swapping that bar for a tiny stand-in
    gives us both: the fraction done, and a place to raise when cancelled.
    """

    def __init__(
        self,
        on_progress: Callable[[float], None] | None,
        cancelled: Callable[[], bool] | None,
    ) -> None:
        self.on_progress = on_progress
        self.cancelled = cancelled
        self._module: Any = None
        self._original: Any = None

    def __enter__(self) -> _mlx_progress:
        import importlib

        # Not `import mlx_whisper.transcribe as module`: the package re-exports a
        # *function* named transcribe that shadows the submodule, so that form
        # binds the function. import_module always returns the module itself.
        module = importlib.import_module("mlx_whisper.transcribe")

        outer = self

        class _Bar:
            def __init__(self, *args: Any, total: int | None = None, **kwargs: Any) -> None:
                self.total = total or 0
                self.n = 0

            def __enter__(self) -> _Bar:
                return self

            def __exit__(self, *exc: object) -> None:
                return None

            def update(self, n: int = 1) -> None:
                self.n += n
                if outer.cancelled is not None and outer.cancelled():
                    raise _MlxCancelled()
                if outer.on_progress and self.total:
                    outer.on_progress(min(0.99, self.n / self.total))

        # Replace mlx_whisper.transcribe's *reference* to the tqdm module, not
        # tqdm.tqdm itself, so other progress bars (model downloads) are untouched.
        self._module = module
        self._original = module.tqdm
        module.tqdm = SimpleNamespace(tqdm=_Bar)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._module is not None:
            self._module.tqdm = self._original


# --------------------------------------------------------------------------
# faster-whisper (CUDA / CPU)
# --------------------------------------------------------------------------


def _transcribe_faster_whisper(
    audio: Path,
    settings: WhisperSettings,
    *,
    language: str,
    duration_s: float | None = None,
    on_progress: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Transcript:

    # Must happen before faster-whisper pulls in CTranslate2, which resolves its
    # CUDA dependencies at import.
    from ..cuda import ensure_cuda_libraries

    ensure_cuda_libraries()

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - faster-whisper is a core dep
        raise TranscriptionError(
            "faster-whisper is not installed. Run `autoclip doctor` for details."
        ) from exc

    device, compute_type = resolve_compute(settings)
    log.info(
        "Transcribing with model=%s device=%s compute_type=%s",
        settings.model,
        device,
        compute_type,
    )

    try:
        model = WhisperModel(settings.model, device=device, compute_type=compute_type)
    except Exception as exc:
        raise _model_load_error(exc, device, compute_type) from exc

    segments_iter, info = model.transcribe(
        str(audio),
        language=language or None,
        word_timestamps=True,
        vad_filter=True,
        # Trims long silences before decoding, which both speeds things up and
        # stops Whisper hallucinating text into empty audio.
        vad_parameters={"min_silence_duration_ms": 500},
    )

    total = duration_s or getattr(info, "duration", 0.0) or 0.0
    transcript = Transcript(
        language=getattr(info, "language", "") or language,
        model=settings.model,
        source="whisper",
    )

    for segment in segments_iter:
        if cancelled is not None and cancelled():
            raise TranscriptionError("Transcription cancelled.")

        first_word = len(transcript.words)
        for word in getattr(segment, "words", None) or []:
            text = (word.word or "").strip()
            if not text:
                continue
            transcript.words.append(Word(text=text, start=float(word.start), end=float(word.end)))

        # A segment with no word timings still carries text worth keeping for
        # display, but it can't contribute to word-indexed clip boundaries.
        last_word = max(first_word, len(transcript.words) - 1)
        transcript.segments.append(
            Segment(
                text=(segment.text or "").strip(),
                start=float(segment.start),
                end=float(segment.end),
                first_word=first_word,
                last_word=last_word,
            )
        )

        if on_progress and total:
            on_progress(min(1.0, float(segment.end) / total))

    if not transcript.words:
        raise TranscriptionError(
            "No speech was detected in this audio. If the file really does contain "
            "speech, try a larger Whisper model or check that the correct audio track "
            "was extracted."
        )

    if on_progress:
        on_progress(1.0)

    log.info(
        "Transcribed %d words in %d segments.", len(transcript.words), len(transcript.segments)
    )
    return transcript


# --------------------------------------------------------------------------
# Diarization
# --------------------------------------------------------------------------


def diarize(
    audio: Path,
    transcript: Transcript,
    *,
    hf_token: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> Transcript:
    """Label each word with a speaker, in place.

    Uses pyannote via WhisperX. Returns the transcript unchanged (with a logged
    warning) if diarization isn't available — speaker labels improve reframing
    but are never required for the pipeline to complete.

    Preconditions:
        transcript already has word-level timings.
    """
    if not transcript.words:
        return transcript

    pipeline_cls = _load_diarization_pipeline()
    if pipeline_cls is None:
        log.warning(
            "WhisperX is not installed; skipping diarization. "
            "Install it with `uv pip install 'autoclip[diarization]'`."
        )
        return transcript

    if not hf_token:
        log.warning(
            "Diarization needs a HuggingFace token. Set one with "
            "`autoclip config set-secret huggingface_token`, and accept the pyannote "
            "model licences on huggingface.co. Continuing without speaker labels."
        )
        return transcript

    device = report().gpu.device
    try:
        pipeline = pipeline_cls(use_auth_token=hf_token, device=device)
        diarization = pipeline(str(audio), min_speakers=min_speakers, max_speakers=max_speakers)
    except Exception as exc:
        log.warning("Diarization failed (%s); continuing without speaker labels.", exc)
        return transcript

    turns = _diarization_turns(diarization)
    if not turns:
        return transcript

    _assign_speakers(transcript, turns)
    log.info("Diarization labelled %d speakers.", len(transcript.speakers))
    return transcript


def _load_diarization_pipeline():
    """Import WhisperX's diarization pipeline across its API reshuffles."""
    try:
        from whisperx.diarize import DiarizationPipeline

        return DiarizationPipeline
    except ImportError:
        pass
    try:
        from whisperx import DiarizationPipeline  # type: ignore[attr-defined]

        return DiarizationPipeline
    except (ImportError, AttributeError):
        return None


def _diarization_turns(diarization) -> list[tuple[str, float, float]]:
    """Normalise WhisperX output into ``(speaker, start, end)`` tuples.

    Recent versions return a pandas DataFrame; older ones return a pyannote
    ``Annotation``. Both are handled so the extra can be upgraded independently.
    """
    turns: list[tuple[str, float, float]] = []

    if hasattr(diarization, "itertuples"):  # DataFrame
        for row in diarization.itertuples():
            speaker = getattr(row, "speaker", None)
            start = getattr(row, "start", None)
            end = getattr(row, "end", None)
            if speaker is not None and start is not None and end is not None:
                turns.append((str(speaker), float(start), float(end)))
        return turns

    if hasattr(diarization, "itertracks"):  # pyannote Annotation
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            turns.append((str(speaker), float(segment.start), float(segment.end)))

    return turns


def _assign_speakers(transcript: Transcript, turns: list[tuple[str, float, float]]) -> None:
    """Attach a speaker to every word by maximum temporal overlap.

    Overlap rather than midpoint containment: words straddling a turn boundary
    are common, and the speaker who covers more of the word is the better guess.
    """
    turns = sorted(turns, key=lambda t: t[1])

    for word in transcript.words:
        best_speaker: str | None = None
        best_overlap = 0.0
        for speaker, start, end in turns:
            if start > word.end:
                break
            overlap = min(word.end, end) - max(word.start, start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        word.speaker = best_speaker

    for segment in transcript.segments:
        segment.speaker = transcript.dominant_speaker(segment.first_word, segment.last_word)
