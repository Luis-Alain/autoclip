"""Local moment proposals: topic shifts, pauses, loudness — not 8-minute windows."""

from __future__ import annotations

import asyncio
import math
import struct
import wave
from pathlib import Path

import pytest
from autoclip.pipeline.transcript import Transcript, Word
from autoclip.providers import ClipCandidate, ClipCandidates, LLMProvider, ProviderStatus
from autoclip.raw import select
from autoclip.raw.candidates import CONTEXT_S, build_candidates
from autoclip.raw.models import CandidatesArtifact, read_model

WORD_S = 0.4


def _say(text: str, *, start: float, speaker: str | None = None) -> tuple[list[Word], float]:
    """Lay ``text`` down as back-to-back words. Returns the words and the time after them."""
    words: list[Word] = []
    t = start
    parts = text.split()
    for index, part in enumerate(parts):
        token = part if index < len(parts) - 1 or part.endswith(".") else f"{part}."
        words.append(Word(text=token, start=t, end=t + WORD_S - 0.02, speaker=speaker))
        t += WORD_S
    return words, t


def _repeat(
    sentence: str, seconds: float, *, start: float = 0.0, speaker: str | None = None
) -> list[Word]:
    words: list[Word] = []
    t = start
    while t < start + seconds - WORD_S:
        chunk, t = _say(sentence, start=t, speaker=speaker)
        words.extend(chunk)
    return words


def _tone(amplitude: int, n: int, rate: int) -> list[int]:
    return [int(amplitude * math.sin(2 * math.pi * 440 * i / rate)) for i in range(n)]


def _wav(path: Path, amplitudes: list[tuple[float, int]], rate: int = 16_000) -> None:
    """``(seconds, amplitude)`` sections of a 440 Hz tone, written as 16-bit mono."""
    samples: list[int] = []
    for seconds, amplitude in amplitudes:
        samples.extend(_tone(amplitude, int(rate * seconds), rate))
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack("<" + "h" * len(samples), *samples))


def test_topic_shift_opens_a_candidate_at_the_change() -> None:
    words = _repeat("hornear queso receta horno", 40) + _repeat(
        "partida ranked clutch rifle", 40, start=40
    )
    moments = build_candidates(Transcript(words=words, language="es"), min_s=15, max_s=45)
    shifts = [m for m in moments if "topic_shift" in m.reasons]
    assert shifts
    assert any(abs(m.start_s - 40) < 1.5 for m in shifts)
    assert all(m.end_s - m.start_s <= 45 + 0.5 for m in moments)


def test_a_long_pause_and_a_topic_change_at_the_same_spot_merge() -> None:
    before = _repeat("hornear queso receta horno", 40)
    after = _repeat("partida ranked clutch rifle", 30, start=before[-1].end + 2.0)
    moments = build_candidates(
        Transcript(words=before + after, language="es"), min_s=15, max_s=45
    )
    merged = [m for m in moments if "topic_shift" in m.reasons and "pause" in m.reasons]
    assert merged
    assert merged[0].start_s == pytest.approx(after[0].start)


def test_pause_alone_starts_at_the_word_after_the_gap() -> None:
    before = _repeat("hornear queso receta horno", 20)
    after = _repeat("hornear queso receta horno", 20, start=before[-1].end + 3.0)
    moments = build_candidates(
        Transcript(words=before + after, language="es"), min_s=15, max_s=40
    )
    pauses = [m for m in moments if "pause" in m.reasons]
    assert pauses
    assert pauses[0].start_s == pytest.approx(after[0].start)
    assert all("topic_shift" not in m.reasons for m in moments)


def test_speaker_change_is_a_candidate_even_without_a_pause() -> None:
    words = _repeat("hornear queso receta horno", 20, speaker="SPEAKER_00")
    words += _repeat(
        "hornear queso receta horno", 20, start=words[-1].end + WORD_S, speaker="SPEAKER_01"
    )
    moments = build_candidates(Transcript(words=words, language="es"), min_s=15, max_s=40)
    speakers = [m for m in moments if "speaker" in m.reasons]
    assert speakers
    assert abs(speakers[0].start_s - 20) < 1.5


def test_loud_peak_marks_the_span_that_contains_it(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    _wav(audio, [(10, 1500), (1, 20000), (9, 1500)])
    words = _repeat("hornear queso receta horno", 20)
    moments = build_candidates(
        Transcript(words=words, language="es"), min_s=8, max_s=18, audio=audio
    )
    loud = [m for m in moments if "loud" in m.reasons]
    assert len(loud) == 1
    assert loud[0].start_s <= 10.5 <= loud[0].end_s
    assert loud[0].loudness_z is not None and loud[0].loudness_z >= 1.7


def test_flat_speech_is_sliced_no_longer_than_the_maximum() -> None:
    words = _repeat("hornear queso receta horno", 90)
    moments = build_candidates(Transcript(words=words, language="es"), min_s=10, max_s=30)
    assert len(moments) == 3
    assert all(m.reasons == ["span"] for m in moments)
    assert all(m.end_s - m.start_s <= 30.5 for m in moments)


def test_select_sends_moments_not_eight_minute_windows(tmp_path: Path) -> None:
    words = _repeat("hornear queso receta horno", 40) + _repeat(
        "partida ranked clutch rifle", 40, start=40
    )
    transcript = Transcript(words=words, language="es")
    seen: list[tuple[float, str]] = []

    class _Provider(LLMProvider):
        name = "scripted"
        requires_key = False

        def __init__(self) -> None:
            super().__init__("scripted")

        async def _complete(self, system: str, user: str, config: object) -> str:
            return ""

        async def health_check(self) -> ProviderStatus:
            return ProviderStatus(name=self.name, available=True)

        async def detect_highlights(self, window, config):  # type: ignore[no-untyped-def]
            seen.append((window.duration_s, config.extra_context))
            import re

            match = re.search(r"words (\d+)–(\d+)", config.extra_context)
            assert match is not None
            return ClipCandidates(
                clips=[
                    ClipCandidate(
                        start_word_index=int(match.group(1)),
                        end_word_index=int(match.group(2)),
                        title="Cambio de tema",
                        score=80,
                    )
                ]
            )

    out = tmp_path / "candidates.json"
    clips = asyncio.run(
        select.select_clips(
            transcript,
            _Provider(),
            min_s=15,
            max_s=45,
            top=5,
            candidates_out=out,
        )
    )
    assert seen
    assert all(duration <= 45 + CONTEXT_S + 1 for duration, _ in seen)
    assert all("topic_shift" in note for _, note in seen)
    saved = read_model(out, CandidatesArtifact)
    assert any("topic_shift" in m.reasons and m.sent for m in saved.candidates)
    assert clips and clips[0].hook_title == "Cambio de tema"
