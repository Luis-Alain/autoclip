# Architecture notes: upstream autoclip, as verified in code

Written for the raw-clips fork (see `CLAUDE.md`). Every claim below was checked
against the source at upstream commit `5d0eac3`. File references are relative to
`backend/autoclip/`.

## 1. Entry points and how a job runs

### CLI (`cli.py`, Typer, installed as the `autoclip` script)

| Command | What it does |
|---|---|
| `doctor` | Probes Python, ffmpeg (filters, encoders), GPU, deps, providers, keyring |
| `clip <url-or-file>` | Runs the full pipeline synchronously with a progress bar |
| `serve` | Starts the FastAPI app + built React UI on `127.0.0.1:8000` |
| `config show / path / set-secret / delete-secret` | Settings and secrets |
| `init`, `jobs`, `providers`, `fetch-models`, `update-ytdlp`, `styles`, `version` | Utilities |

`autoclip clip` flow (`cli.py:405`):

1. `db.init()` and `config.load()`. CLI flags override settings **in memory only**.
2. **Ingest happens in the CLI, not in the runner.** If `ingest.is_youtube_url(target)` is true,
   it calls `ingest_youtube()`. Otherwise it treats the target as a local path and calls `ingest_file()`.
   There is no Twitch/Kick support: a Twitch URL fails as "does not exist".
3. It creates `Source` and `Job` rows in SQLite.
4. `asyncio.run(PipelineRunner(job, source, settings).run())`.

### API (`app.py`, `api/`)

The API serves the same pipeline through a FIFO `JobQueue` (`jobs/queue.py`), which runs one job
at a time in a worker thread (`runner.run_job` → `run_in_executor`) and reports progress over SSE.

- Sources: `POST /api/sources/youtube`, `POST /api/sources/upload` (multipart file upload)
- Jobs: `POST /api/jobs`, `GET /api/jobs/{id}/events` (SSE), `/cancel`, `/retry`, `/clips`
- Clips: `PATCH /api/clips/{id}`, `PATCH /api/clips/{id}/captions`, `POST /api/clips/{id}/export`
- Settings: `GET/PUT /api/settings`, `PUT /api/settings/secrets`, `/providers/status`, `/system`

For v1 the fork is **CLI-only**. The API and UI stay untouched.

### Stage order (`pipeline/__init__.py`, `pipeline/runner.py:171`)

```
(ingest, in cli/api) → prepare → transcribe → [silences] → highlights → reframe → captions → export
```

`silences` is not a `Stage` enum member. It runs as a side step before highlights.

## 2. Stages: where they live and what they read and write

Work dir: `paths.job_work_dir(job_id)` = `$AUTOCLIP_HOME/work/{job_id}/`, where `AUTOCLIP_HOME`
defaults to `~/.autoclip`. Other dirs: `media/{source_id}/`, `exports/{job_id}/`, `autoclip.db`, `config.json`.

| Stage | Code | Reads | Writes | Resumes? |
|---|---|---|---|---|
| ingest (YouTube) | `pipeline/ingest.py:78` | URL | `media/{source_id}/<file>` via yt-dlp, format capped at 1080p | No |
| ingest (file) | `pipeline/ingest.py:217` | local file | **copies** the file into `media/{source_id}/` | No |
| prepare | `pipeline/prepare.py:43,71` | source | `audio.wav` (mono, 16 kHz, `pcm_s16le`), `thumbnails/` | Yes, skipped if `audio.wav` is non-empty |
| transcribe | `pipeline/transcribe.py:86` | `audio.wav` | `transcript.json` + a `transcripts` DB row | Yes, skipped if the file exists |
| silences | `pipeline/prepare.py:110` | `audio.wav` | `silences.json` = `[{start, end}]` (silencedetect at −32 dB, ≥0.2 s) | Yes |
| highlights | `pipeline/highlights.py:123` | transcript, silences | **SQLite `clips` table** (no file) | Yes, skipped if the DB already has clips for the job |
| reframe | `pipeline/reframe/` | source, clip ranges | `crops/{clip_id}.json` | Yes, per clip |
| captions | `pipeline/captions.py` | none | none (validates the style name) | n/a |
| export | `pipeline/export.py:221` | source, crop path, words | `exports/{job_id}/<slug>.mp4` + `exports` rows, ASS files in `captions/` | **No**, re-renders every clip |

All ffmpeg calls go through `pipeline/ffmpeg.py:268` `run()`, which logs the full command at
**DEBUG** level. `ffmpeg.probe()` returns `MediaInfo` (duration, width, height, fps, codecs).

## 3. Transcript schema (`pipeline/transcript.py`)

`transcript.json`:

```json
{
  "language": "es",
  "model": "small",
  "source": "whisper",          // or "youtube"
  "words":    [{"text": "Hola", "start": 12.34, "end": 12.61, "speaker": null}],
  "segments": [{"text": "Hola a todos.", "start": 12.34, "end": 13.9,
                "first_word": 0, "last_word": 2, "speaker": null}]
}
```

- Times are seconds, relative to the start of the audio file. **Today that is always the full
  source, so they're effectively absolute.**
- `Transcript` methods include `index_at_time`, `indices_in_range`, `time_range`, `slice`, `text_between`, `sentence_end_indices` and `pause_before`.
- `Segment.first_word/last_word` index into `words`. This is the only coupling between segments and words.

**Fork implication:** when we transcribe a window cut out of the VOD, every `start/end` has to be
shifted by the window's absolute offset before saving. Then everything downstream keeps working on
absolute time.

## 4. Highlights: schema, prompt, model config

### Flow (`pipeline/highlights.py`)

1. `build_windows()`: 8-minute windows (`WINDOW_S = 480`) with a 60 s overlap, cut at word boundaries.
2. Each window is rendered as `[index]word [index]word ...`, using **global** word indices (`render_window_text`).
3. `provider.detect_highlights(window, config)` runs 3 windows at a time on hosted providers and 1 on Ollama.
   If one window fails, the error is logged and the other windows continue.
4. `build_clips()`:
   - `dedupe()`: greedy by score, dropping a candidate whose word-range IoU with a kept one is > 0.4
   - `boundaries.refine()`: snaps to sentence ends (±12 words), clamps to `[min, max]` duration, aligns to silences
   - drops candidates that can't fit the duration range
   - a second dedupe, then sort by score, keep the top `max_clips`, assign `rank`

**Word indices confirmed.** The model never outputs seconds. `LLMProvider._parse()` clamps
indices into the window and drops ranges where `end <= start`.

### Candidate schema the model must return (`providers/base.py:55`)

```json
{"clips": [{"start_word_index": 1042, "end_word_index": 1385,
            "title": "...", "hook": "verbatim opening words", "score": 87, "reason": "..."}]}
```

Validation is lenient. `null` text becomes `""`. A score given as a 0–1 fraction is scaled to 0–100.
A bare array is accepted. Fenced or prose-wrapped JSON is extracted.
If the JSON is invalid, there is **one retry** with the validation error appended.

### Output (`db/models.py:129` `Clip` row)

`id, job_id, start_s, end_s, rank, start_word, end_word, title, hook, score, reason, status, user_trimmed`

### Prompt and config

- System prompt: `prompts/highlight_v1.txt`. It's English and aimed at TikTok/Shorts: hook, self-containment,
  payoff, quotability. Scores below 50 are not returned. Selected by `DetectionConfig.prompt_version`.
- User message: `render_window_prompt()`. Contains the word range, min/max seconds, max clip count, and the tagged text.
- `DetectionConfig`: `min_duration_s=20`, `max_duration_s=90`, `max_clips=10`, `temperature=0.3`.
  Built from `settings.clips.*` by `providers.detection_config()`.

**Gaps vs CLAUDE.md:**
- No `needs_context` field.
- No Spanish instruction for titles.
- No extra context such as chat messages.
- The provider entry point is hardwired to `load_prompt(config.prompt_version)` + `ClipCandidates`.

The fork needs a new prompt file and a way to pass extra context and a richer schema.
Plan: a flag-gated extension to `providers/base.py`, or a raw-specific subclass of the candidate models.

## 5. LLM providers (`providers/`, `config.py`)

| Name | Class | Default model | Notes |
|---|---|---|---|
| `anthropic` | `AnthropicProvider` | `claude-sonnet-5` | `max_tokens=4096` |
| `openai` | `OpenAIProvider` | `gpt-4o` | `base_url` configurable, so it also covers OpenRouter, Groq and LM Studio. `response_format=json_object` |
| `gemini` | `GeminiProvider` | `gemini-2.0-flash` (outdated) | `response_mime_type=application/json` |
| `ollama` | `OllamaProvider` | **none, must be set** | `http://localhost:11434/api/generate`, `format: json` |

- Selection: `settings.active_provider`, which defaults to **`anthropic`**. Can be overridden with `autoclip clip -p <name>`.
- Model and base URL: `settings.providers[<name>]` in `~/.autoclip/config.json`. No CLI command sets the model.
  You edit the JSON or use the web UI.
- Keys: OS keyring (macOS Keychain) via `autoclip config set-secret <name>`. Can be overridden with the env var
  `AUTOCLIP_<NAME>_KEY` (for example `AUTOCLIP_GEMINI_KEY`). If no keyring is available, keys are stored in plaintext in config.json with a warning.
- `build_provider(name, settings)` is the factory. `PROVIDERS` is the registry.

## 6. Transcription backend

- **There is no backend interface.** `transcribe.transcribe()` imports and calls `faster_whisper.WhisperModel`
  directly with `word_timestamps=True`, `vad_filter=True` and `min_silence_duration_ms=500`.
- Device and compute type come from `system.probe_gpu()` + `resolve_compute()`. On Apple Silicon,
  `accel="mps"` but `device="cpu"` (`system.py:150`). CTranslate2 has no Metal backend, so
  **Whisper runs on CPU on the M4**, with int8 preferred.
- Settings: `whisper.model` defaults to `small`. `whisper.language` defaults to `""` (auto-detect). `compute_type` can be overridden.
- Optional WhisperX diarization adds `speaker` per word. Not needed for the fork.
- There is a YouTube-captions fast path (`ingest.prefer_youtube_captions`) that sets `source="youtube"`.

**Where to plug in mlx-whisper:** add a small `TranscriberBackend` protocol
(`transcribe(audio, settings, ...) -> Transcript`). Move the current body into a `FasterWhisperBackend`,
add an `MlxWhisperBackend`, and pick between them in `transcribe.transcribe()` based on a new
`whisper.backend` setting (`auto|faster-whisper|mlx`). `auto` picks mlx on darwin/arm64 when it can be imported.
Both must produce the schema in section 3. `mlx_whisper.transcribe()` returns segments with `words[{word,start,end}]`, so the mapping is direct.

## 7. Tests

- ~287 `def test…` functions in `tests/` across 12 files: captions, boundaries, export_render, ffmpeg, config, api, providers, reframe, db, paths, system, pipeline_e2e.
- `conftest.py` points `AUTOCLIP_HOME` at a temp dir.
- Markers: `slow`, `golden`, `e2e`. The e2e test runs the real pipeline with a scripted provider and needs `AUTOCLIP_E2E_MEDIA`.

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e '.[dev]'
pytest -m "not slow" -q                                   # fast suite
AUTOCLIP_E2E_MEDIA=$PWD/samples/test.mp4 pytest -m e2e -q # real media
```

- CI (`.github/workflows/ci.yml`) runs Ubuntu 3.11/3.12, macOS 3.12 and Windows 3.12. It installs `ffmpeg-full` on macOS.

## 8. Apple Silicon and Python 3.11/3.12 caveats

- **Python:** `requires-python = ">=3.11,<3.13"`, and `.python-version` is `3.11`. This machine only has 3.14,
  so run `uv venv --python 3.11`, which downloads a managed interpreter.
- **ffmpeg:** you need `brew install ffmpeg-full`, because Homebrew's plain `ffmpeg` has no libass.
  `doctor` marks the pipeline "not ready" without the `ass` filter, **even though the raw pipeline never burns captions.**
  The raw command should do its own lighter check: ffmpeg, ffprobe, `ebur128`, `silencedetect`, `h264_videotoolbox`.
- **Whisper is CPU-only** on Apple Silicon through faster-whisper (see section 6). This is the main reason for M2.
- **Encoders:** export only knows `h264_nvenc` and `libx264`. It has no `h264_videotoolbox` or ProRes, so those are new for export_raw.
- **Copying on ingest:** `ingest_file()` copies the source into `~/.autoclip/media/`. For multi-GB VODs the raw
  pipeline should reference the file in place.
- **Slugs:** `ingest.slugify` and `export.slugify_title` use Unicode `\w`, so accents survive and the slugs are not ASCII-safe.
- **Core deps are heavy:** mediapipe, scenedetect and opencv are installed even though we skip reframe. They're harmless, and they're why Python can't be 3.13+.
- Torch/WhisperX (the `diarization` extra) is not needed. The `gpu` extra installs nothing on macOS (by design).

## 9. Reuse vs new code for the fork

**Reused as-is:**
- `ffmpeg.run/probe`, `prepare.extract_audio/detect_silences`, `ingest.ingest_youtube` (metadata patterns), the `Transcript` model,
  `boundaries.refine`, `highlights.dedupe/build_windows/render_window_text`, the providers and `extract_json_object`, config and secrets.

**Small, flag-gated upstream edits:**
- `providers/base.py`: allow a custom prompt version, extra context, and a candidate schema.
- `transcribe.py`: backend selection.
- `cli.py`: register the `raw` command.

**New `autoclip/raw/` package:**
- `cli.py` (Typer sub-command), `runner.py` (file-based, resumable stages), `models.py` (pydantic)
- `slug.py`, `timemap.py`, `windows.py`, `spikes.py`, `chat.py`, `export_raw.py`, `export_premiere.py`, `campaigns.py`
- New prompt `prompts/highlight_raw_v1.txt`. Profiles in `campaigns/*.yaml`.

The raw pipeline **does not use SQLite or the job queue.** Its state is the output folder:

```
out/<date>_<channel>_<vod-id>/
  source.json  spikes.json  windows.json  transcript/<window>.json
  highlights.json  clips/  manifest.json  manifest.csv  timeline.xml  logs/
```

A stage is skipped when its artifact exists, the same resume rule upstream uses.

## 10. Plan for M1 (next change)

1. Add `raw/` scaffolding with models, `slug.py` and `timemap.py`, plus tests (accents, ñ, emoji; local↔absolute time).
2. Add `prompts/highlight_raw_v1.txt`: English instructions, Spanish transcript, `hook_title` in Spanish, `needs_context`.
3. Extend the provider call (flag-gated) to take the prompt version, extra context and the raw candidate schema.
4. Add `RawRunner` for a local file with `--chat none`. Steps:
   - probe the source in place
   - extract audio
   - transcribe with upstream faster-whisper (mlx comes in M2)
   - detect silences
   - run highlights per ~10-minute chunk
   - refine, dedupe and take the top N
   - write `highlights.json`
5. `export_raw.py`: frame-accurate re-encode with handles (`h264_videotoolbox` high bitrate; `--prores` → `prores_ks`).
   Files are named `{rank:02d}_s{score}_{slug}.mp4`. Write `manifest.json` and `manifest.csv`.
6. Add the `autoclip raw` CLI with `--lang --chat --top --min --max --handles --provider --prores --out`.
   Defaults: `--lang es`, 15–60 s, 4 s handles.
7. Acceptance: `autoclip raw samples/test.mp4 --chat none` produces ranked 16:9 clips and a manifest.

## Open questions (pending)

1. Typical VOD fps and resolution? Sequence settings will be read from each source anyway.
2. H.264 or ProRes by default? Proposed: H.264 videotoolbox, with ProRes behind `--prores`.
3. Default LLM: Gemini Flash-Lite or Ollama? (The only local Ollama model right now is `qwen2.5-coder:7b`, which is a weak fit.)
4. Kick chat in v1? Proposed: no, accept a manual chat file instead.
