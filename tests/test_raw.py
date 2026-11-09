"""Raw-clips pipeline (autoclip/raw)."""

from __future__ import annotations

import asyncio
import csv
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from autoclip.config import Settings
from autoclip.pipeline.transcript import Transcript, Word
from autoclip.providers import ClipCandidate, ClipCandidates, LLMProvider, ProviderStatus
from autoclip.raw import export_raw, select, source
from autoclip.raw.manifest import read_manifest, write_manifest
from autoclip.raw.models import ManifestRow, SourceInfo
from autoclip.raw.slug import slugify

# -- slug -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("¡El GRITO de Ñandú!", "el-grito-de-nandu"),
        ("Qué pasó aquí 😂💀", "que-paso-aqui"),
        ("  --Hola__mundo--  ", "hola-mundo"),
        ("عادات التحقق من الهواتف", "clip"),  # no ASCII form -> fallback
        ("😂😂😂", "clip"),
    ],
)
def test_slugify_is_ascii_safe(text: str, expected: str) -> None:
    assert slugify(text) == expected


def test_slugify_truncates_without_trailing_dash() -> None:
    slug = slugify("palabra " * 20, max_length=20)
    assert len(slug) <= 20
    assert not slug.endswith("-")


# -- cut planning -----------------------------------------------------------


def test_plan_cut_adds_handles_and_maps_in_out() -> None:
    cut = export_raw.plan_cut(100.0, 130.0, handles_s=4.0, source_duration_s=3600.0)
    assert (cut.handle_start, cut.handle_end) == (96.0, 134.0)
    assert (cut.in_point, cut.out_point) == (4.0, 34.0)
    assert cut.file_duration == 38.0


def test_plan_cut_clamps_to_source_edges() -> None:
    start = export_raw.plan_cut(1.5, 20.0, handles_s=4.0, source_duration_s=3600.0)
    assert start.handle_start == 0.0
    assert start.in_point == 1.5
    end = export_raw.plan_cut(3590.0, 3598.0, handles_s=4.0, source_duration_s=3600.0)
    assert end.handle_end == 3600.0
    assert (end.in_point, end.out_point) == (4.0, 12.0)


def test_raw_filename() -> None:
    assert export_raw.raw_filename(1, 92, "¡El grito!", prores=False) == "01_s92_el-grito.mp4"
    assert export_raw.raw_filename(12, 70, "x", prores=True) == "12_s70_x.mov"


def test_encode_args_choose_codecs() -> None:
    def codec(args: list[str]) -> str:
        return args[args.index("-c:v") + 1]

    assert codec(export_raw.encode_args(prores=False, hardware=True)) == "h264_videotoolbox"
    assert codec(export_raw.encode_args(prores=False, hardware=False)) == "libx264"
    assert codec(export_raw.encode_args(prores=True, hardware=True)) == "prores_videotoolbox"
    assert codec(export_raw.encode_args(prores=True, hardware=False)) == "prores_ks"
    assert "pcm_s16le" in export_raw.encode_args(prores=True, hardware=True)


# -- selection --------------------------------------------------------------


def _transcript(n_words: int = 400, step: float = 0.5) -> Transcript:
    """Words every ``step`` seconds; every 10th word ends a sentence."""
    words = [
        Word(text=f"w{i}" + ("." if i % 10 == 9 else ""), start=i * step, end=i * step + 0.4)
        for i in range(n_words)
    ]
    return Transcript(words=words, language="es")


def _candidate(start: int, end: int, score: int, **kw: object) -> ClipCandidate:
    return ClipCandidate(start_word_index=start, end_word_index=end, score=score, **kw)


def test_rank_clips_dedupes_ranks_and_keeps_fields() -> None:
    t = _transcript()
    clips = select.rank_clips(
        t,
        [
            _candidate(10, 69, 70, title="Segundo"),
            _candidate(12, 70, 90, title="Duplicado mejor", needs_context=True),
            _candidate(200, 259, 80, title="Tercero"),
        ],
        min_s=15,
        max_s=60,
        top=10,
    )
    assert [c.hook_title for c in clips] == ["Duplicado mejor", "Tercero"]
    assert [c.rank for c in clips] == [1, 2]
    assert clips[0].needs_context is True
    assert all(15 - 0.5 <= c.duration_s <= 60 + 0.5 for c in clips)


def test_rank_clips_applies_top_n() -> None:
    t = _transcript(1000)
    candidates = [_candidate(i, i + 59, 50 + i // 100) for i in range(0, 900, 100)]
    clips = select.rank_clips(t, candidates, min_s=15, max_s=60, top=3)
    assert len(clips) == 3
    assert [c.score for c in clips] == sorted((c.score for c in clips), reverse=True)


def test_rank_clips_extends_short_candidates_to_the_minimum() -> None:
    clips = select.rank_clips(_transcript(), [_candidate(0, 3, 90)], min_s=15, max_s=60, top=10)
    assert len(clips) == 1
    assert clips[0].duration_s >= 15 - 0.5


def test_rank_clips_drops_what_cannot_reach_the_minimum() -> None:
    short = _transcript(n_words=20)  # 10 s of speech in total
    assert select.rank_clips(short, [_candidate(0, 19, 90)], min_s=15, max_s=60, top=10) == []


def test_rank_clips_replaces_placeholder_titles() -> None:
    clips = select.rank_clips(
        _transcript(),
        [_candidate(100, 159, 80, title="<specific hook title for this clip>")],
        min_s=15,
        max_s=60,
        top=10,
    )
    assert clips[0].hook_title.startswith("w1")


# -- manifest ---------------------------------------------------------------


def _row(rank: int) -> ManifestRow:
    return ManifestRow(
        rank=rank,
        file=f"clips/{rank:02d}_s80_x.mp4",
        abs_start=10.0,
        abs_end=40.0,
        handle_start=6.0,
        handle_end=44.0,
        in_point=4.0,
        out_point=34.0,
        duration=30.0,
        score=80,
        hook_title="¡Qué jugada! ñ",
        reason="r",
        needs_context=False,
        transcript_excerpt="hola",
        language="es",
        source_path="/x.mp4",
    )


def test_manifest_round_trip(tmp_path: Path) -> None:
    json_path, csv_path = write_manifest([_row(1), _row(2)], tmp_path)
    assert [r.rank for r in read_manifest(json_path)] == [1, 2]
    with csv_path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["hook_title"] == "¡Qué jugada! ñ"
    assert rows[1]["in_point"] == "4.0"


# -- source -----------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=1Vd0adEBOuU",
        "https://youtu.be/1Vd0adEBOuU",
        "https://www.youtube.com/live/1Vd0adEBOuU",
    ],
)
def test_youtube_id(url: str) -> None:
    assert source.youtube_id(url) == "1Vd0adEBOuU"


def test_folder_name_is_ascii() -> None:
    info = SourceInfo(path="/x", channel="ElCañón TV", vod_id="Stream #12 😂", duration_s=1)
    assert source.folder_name(info, "2026-11-08") == "2026-11-08_elcanon-tv_stream-12"


def test_twitch_links_are_rejected_clearly(tmp_path: Path) -> None:
    with pytest.raises(source.SourceError, match="M5"):
        source.resolve("https://www.twitch.tv/videos/123", tmp_path, Settings())


# -- end to end (real ffmpeg, scripted transcript and model) -----------------


class _ScriptedProvider(LLMProvider):
    name = "scripted"
    requires_key = False

    def __init__(self) -> None:
        super().__init__("scripted-model")

    async def _complete(self, system: str, user: str, config: object) -> str:
        return ""

    async def health_check(self) -> ProviderStatus:
        return ProviderStatus(name=self.name, available=True)

    async def detect_highlights(self, window, config):  # type: ignore[no-untyped-def]
        return ClipCandidates(
            clips=[
                ClipCandidate(
                    start_word_index=4,
                    end_word_index=13,
                    title="La jugada del año",
                    score=88,
                    reason="r",
                    needs_context=True,
                ),
                ClipCandidate(
                    start_word_index=24, end_word_index=33, title="Segundo momento", score=75
                ),
            ]
        )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_raw_pipeline_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.raw import runner as raw_runner

    video = tmp_path / "Mi Stream Épico.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=40",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=40",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(video),
        ],
        check=True,
    )  # fmt: skip

    words = [
        Word(text=f"palabra{i}" + ("." if i % 5 == 4 else ""), start=i * 1.0, end=i * 1.0 + 0.8)
        for i in range(40)
    ]
    monkeypatch.setattr(
        raw_runner.transcribe,
        "transcribe",
        lambda *a, **k: Transcript(words=words, language="es", model="fake"),
    )
    monkeypatch.setattr(raw_runner, "build_provider", lambda *a, **k: _ScriptedProvider())

    settings = Settings()
    info, folder = source.resolve(str(video), tmp_path / "out", settings)
    options = raw_runner.RawOptions(min_s=5, max_s=12, handles_s=2, vertical=False)
    monkeypatch.chdir(tmp_path)
    folder = folder.relative_to(tmp_path)  # a relative folder must still work
    result = asyncio.run(raw_runner.RawRunner(info, folder, options, settings).run())

    assert [c.rank for c in result.clips] == [1, 2]
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    first = manifest[0]
    assert first["file"].startswith("clips/01_s88_la-jugada-del-ano")
    assert first["needs_context"] is True
    assert first["in_point"] == pytest.approx(first["abs_start"] - first["handle_start"])

    # The file really contains handle_start..handle_end of the source.
    clip_file = folder / first["file"]
    duration = float(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(clip_file)],
            capture_output=True, text=True, check=True,
        ).stdout
    )  # fmt: skip
    assert duration == pytest.approx(first["handle_end"] - first["handle_start"], abs=0.1)

    # Second run resumes: nothing is re-rendered.
    mtime = clip_file.stat().st_mtime
    asyncio.run(raw_runner.RawRunner(info, folder, options, settings).run())
    assert clip_file.stat().st_mtime == mtime
    assert (folder / "source.json").exists()
    assert (folder / "highlights.json").exists()
    assert (folder / "transcript" / "full.json").exists()
