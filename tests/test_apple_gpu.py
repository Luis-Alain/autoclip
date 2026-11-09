"""Apple Silicon GPU path, language verification, and the first-run fixes."""

from __future__ import annotations

from pathlib import Path

import pytest
from autoclip.config import ExportSettings, WhisperSettings
from autoclip.pipeline import export, highlights, transcribe
from autoclip.providers import ClipCandidate, DetectionConfig, TranscriptWindow
from autoclip.providers.base import load_prompt, render_window_prompt

# -- backend selection ------------------------------------------------------


def test_auto_backend_uses_mlx_when_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transcribe, "mlx_supported", lambda: True)
    assert transcribe.resolve_backend(WhisperSettings()) == "mlx"


def test_auto_backend_falls_back_to_faster_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transcribe, "mlx_supported", lambda: False)
    assert transcribe.resolve_backend(WhisperSettings()) == "faster-whisper"


def test_explicit_backend_is_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transcribe, "mlx_supported", lambda: True)
    settings = WhisperSettings(backend="faster-whisper")
    assert transcribe.resolve_backend(settings) == "faster-whisper"


def test_forcing_mlx_where_unavailable_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transcribe, "mlx_supported", lambda: False)
    with pytest.raises(transcribe.TranscriptionError, match="mlx-whisper"):
        transcribe.resolve_backend(WhisperSettings(backend="mlx"))


# -- whisper output conversion ---------------------------------------------


def test_whisper_dict_becomes_transcript() -> None:
    result = {
        "language": "es",
        "segments": [
            {
                "start": 0.0,
                "end": 1.2,
                "text": " Hola a todos.",
                "words": [
                    {"word": " Hola", "start": 0.0, "end": 0.4},
                    {"word": " a", "start": 0.4, "end": 0.5},
                    {"word": " todos.", "start": 0.5, "end": 1.2},
                ],
            },
            {
                "start": 2.0,
                "end": 2.6,
                "text": " Vamos.",
                "words": [{"word": " Vamos.", "start": 2.0, "end": 2.6}, {"word": " "}],
            },
        ],
    }
    t = transcribe._transcript_from_whisper_dict(result, model="m")
    assert t.language == "es"
    assert [w.text for w in t.words] == ["Hola", "a", "todos.", "Vamos."]
    assert (t.segments[1].first_word, t.segments[1].last_word) == (3, 3)
    assert t.words[2].end == 1.2


# -- language verification --------------------------------------------------


def _fake_detection(monkeypatch: pytest.MonkeyPatch, codes: list[str]) -> None:
    import numpy as np

    answers = iter(codes)
    monkeypatch.setattr(transcribe, "_read_pcm", lambda *a, **k: np.zeros(10, dtype=np.float32))
    monkeypatch.setattr(transcribe, "_detect_language", lambda *a, **k: next(answers))


def test_language_mismatch_switches_to_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_detection(monkeypatch, ["ar", "ar", "es"])
    chosen = transcribe.verify_language(
        Path("x.wav"), "es", backend="mlx", settings=WhisperSettings(), duration_s=900
    )
    assert chosen == "ar"


def test_language_match_keeps_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_detection(monkeypatch, ["es", "en", "es"])
    chosen = transcribe.verify_language(
        Path("x.wav"), "es", backend="mlx", settings=WhisperSettings(), duration_s=900
    )
    assert chosen == "es"


def test_language_check_failure_never_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("no ffmpeg")

    monkeypatch.setattr(transcribe, "_read_pcm", boom)
    chosen = transcribe.verify_language(
        Path("x.wav"), "es", backend="mlx", settings=WhisperSettings(), duration_s=900
    )
    assert chosen == "es"


def test_sample_starts_stay_inside_the_audio() -> None:
    starts = transcribe._sample_starts(900, 3, 30)
    assert len(starts) == 3
    assert all(s >= 0 and s + 30 <= 900 for s in starts)
    assert transcribe._sample_starts(20, 3, 30) == [0.0]


# -- encoder ----------------------------------------------------------------


def _report(system):  # type: ignore[no-untyped-def]
    return system.SystemReport(
        python_version="3.11",
        python_ok=True,
        platform="test",
        ffmpeg=system.FFmpegInfo(),
        gpu=system.GPUInfo(),
        deps=system.OptionalDeps(),
    )


def test_videotoolbox_is_preferred_when_it_works(monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip import system

    rep = _report(system)
    rep.ffmpeg.videotoolbox_works = True
    monkeypatch.setattr(export, "report", lambda: rep)
    args = export.encoder_args(ExportSettings(crf=18))
    assert args[args.index("-c:v") + 1] == "h264_videotoolbox"
    assert args[args.index("-q:v") + 1] == "64"


def test_software_encoder_without_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip import system

    rep = _report(system)
    monkeypatch.setattr(export, "report", lambda: rep)
    args = export.encoder_args(ExportSettings())
    assert args[args.index("-c:v") + 1] == "libx264"


# -- prompt and titles ------------------------------------------------------


def test_v2_prompt_has_no_copyable_example_title() -> None:
    prompt = load_prompt(DetectionConfig().prompt_version)
    assert "Why most agencies die at 10 clients" not in prompt
    assert "SAME LANGUAGE AS THE TRANSCRIPT" in prompt


def test_window_prompt_names_the_language() -> None:
    window = TranscriptWindow(text="[0]hola", first_word=0, last_word=0)
    text = render_window_prompt(window, DetectionConfig(language="es"))
    assert "The transcript is in Spanish" in text


def test_placeholder_titles_fall_back_to_transcript_words() -> None:
    from autoclip.pipeline.transcript import Transcript, Word

    words = [Word(text=f"palabra{i}.", start=i * 1.0, end=i * 1.0 + 0.8) for i in range(60)]
    t = Transcript(words=words)
    candidate = ClipCandidate(
        start_word_index=0, end_word_index=30, title="<specific label for this clip>", score=80
    )
    clips = highlights.build_clips(
        t, [candidate], DetectionConfig(min_duration_s=5, max_duration_s=60), job_id="j"
    )
    assert clips
    assert not clips[0].title.startswith("<")
    assert clips[0].title.startswith("palabra")
