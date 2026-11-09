# CLAUDE.md: autoclip fork for raw stream clips

## What this project is

This is a fork of [artbyjazi/autoclip](https://github.com/artbyjazi/autoclip) (MIT). It is an open-source, local-first AI clipper.

We are repurposing it. The goal is **not** finished 9:16 shorts. The goal is to:

> Take a long stream VOD (1–6 h, Twitch / Kick / YouTube), find the best moments automatically, and export them as **vertical 9:16 clips with burned-in captions**, ready to post on social media (reuses upstream's speaker-tracked reframe and animated captions). A **Premiere Pro XML timeline** of those clips and a manifest come with them.

*Decisions (2026-11-08): Luis first asked for 9:16 captioned clips in addition to the raw 16:9 export, then dropped the 16:9 export entirely: only 9:16 exists. The Premiere timeline stays, pointing at the 9:16 clips.*

The time we save is the slow part: watching hours of VOD, picking moments, and cutting them. Platforms suppress clips that add nothing original, so the timeline lets Luis add his own edit (hooks, VFX) on top of the generated clips in Premiere.

## Owner context

- Luis is a full-stack developer (TypeScript, Next.js, NestJS, Django, PostgreSQL/Supabase, Docker).
- His Python is **basic**. Use clear, idiomatic Python and briefly explain non-obvious Python patterns when you introduce them.
- Machine: **MacBook Pro M4** (Apple Silicon). Optimize for running natively on macOS, not in Docker. Docker on Mac has no access to the Apple GPU.
- Content language: mostly **Spanish** (LatAm streamers), sometimes English.
- Editor: Adobe Premiere Pro.
- Prefers direct, concise answers.

## Non-goals (do not build)

- B-roll, music, auto-generated VFX.
- Auto-posting to TikTok, YouTube or Instagram, and anything that games views or engagement.
- Live-stream processing. VODs only.

## First task: map the codebase before changing anything

The upstream README describes stages `ingest → prepare → transcribe → highlights → reframe → captions → export`. It says per-job artifacts go to `~/.autoclip/work/{job_id}/` so failed jobs can resume. It says the highlight LLM returns **word indices**, which are mapped to measured word timings. Verify all of this in the code. Don't assume it.

Before writing code, produce `docs/ARCHITECTURE_NOTES.md` covering:

1. The CLI and API entry points, and how a job runs stage by stage
2. Where each stage lives, and what files it reads and writes (exact formats)
3. The transcript schema (segments, words, timestamps)
4. The highlights output schema, plus the prompt and model config
5. How LLM providers are configured (Ollama, Gemini, OpenAI-compatible, Anthropic)
6. How the transcription backend is chosen, and where to plug in a new one
7. Existing tests and how to run them
8. Anything that breaks on Apple Silicon or with Python 3.11/3.12

Show Luis the notes and a short plan before the first big change.

## Target pipeline

```
fetch → spikes → windows → transcribe → highlights → export_vertical → export_premiere
        (new)    (new)     (mlx, done)   (reuse)     (reuse upstream)  (new)
```

`export_vertical` runs upstream's `reframe` (MediaPipe face tracking) and caption burn-in (libass, `--style`) on the ranked clips, at 9:16, encoded with VideoToolbox. Captions use the clip's word timings from the transcript, in the transcript's language.

### 0. fetch
- Input: a URL (Twitch/Kick/YouTube VOD) or a local video file.
- Get metadata with `yt-dlp` (duration, title, channel, fps, resolution).
- Get chat when available:
  - Twitch: `TwitchDownloaderCLI chatdownload` → JSON
  - YouTube live replay: `yt-dlp --write-subs --sub-langs live_chat`
  - Kick: optional and later. Accept a manual chat file path.
- Chat is optional. With no chat, the pipeline still works (see `windows`).

### 1. spikes (new, the main differentiator)
- Bin chat messages into 10-second windows. Compute a rolling median baseline over about 5 minutes.
- Flag bins where `count / baseline >= 3.0`. Make this configurable.
- Weight hype tokens: `KEKW, LUL, OMEGALUL, jaja, jajaja, xd, clip, CLIP IT, WTF, 💀, 😂`. Keep the list configurable per streamer.
- Add audio loudness peaks with ffmpeg `ebur128` as a second signal.
- **Backdate** each spike about 30 seconds (chat reacts after the moment). Window = `[spike - 45s, spike + 45s]`, configurable.
- Merge windows less than 90 seconds apart.
- Output `spikes.json`: `[{start, end, chat_score, audio_score, top_messages[]}]`.

### 2. windows (new)
- With spikes: process only the spike windows (typically 20–40 min out of 4 h). Download them with `yt-dlp --download-sections "*HH:MM:SS-HH:MM:SS"`, or cut from a local file with ffmpeg.
- Without chat: fall back to the whole VOD, processed in overlapping chunks of about 10 minutes.
- Keep a mapping from window-local time to absolute VOD time. All later timestamps must be absolute.

### 3. transcribe (new backend)
- Add an **mlx-whisper** backend (`mlx-community/whisper-large-v3-turbo`, `word_timestamps=True`) behind upstream's backend interface. Make it the default on Apple Silicon.
- Keep upstream's backend as the fallback (faster-whisper on CPU/CUDA).
- Language: `--lang es|en|auto`, default `es`.
- Output must match upstream's transcript schema, so the highlights stage works unchanged.

### 4. highlights (reuse upstream)
- Reuse the word-index approach.
- Run it **per window**, not on a full 4-hour transcript. Small local models can't handle that much context.
- Add the window's `top_messages` from chat to the prompt as context.
- Prompt in English, but tell the model the transcript is in Spanish and that hook titles must be written in Spanish.
- Required JSON per candidate: `start_word, end_word, hook_title, reason, score (0–100), needs_context (bool)`.
- Enforce constraints **in code**, not just in the prompt: min/max duration (default 15–60 s, from the campaign profile), dedupe overlapping candidates (keep the higher score), keep the top N.
- Default provider: whatever upstream supports. Recommend Gemini Flash-Lite (cheap) or Ollama (free, local).

### 5. export_vertical (reuse upstream)
- For each candidate, reframe to 9:16 around the speaker and burn in animated captions in the transcript's language. Cut accurately (re-encode), `h264_videotoolbox`.
- Filename: `{rank:02d}_s{score}_{slug(hook_title)}.mp4`, ASCII-safe slug, in `clips/`.
- Write `manifest.json` and `manifest.csv` with: rank, file, abs_start, abs_end, duration, score, hook_title, reason, needs_context, chat_score, transcript_excerpt, source_url, campaign.

### 5b. layouts for gameplay streams (new, M7)
- Detect the webcam overlay once per VOD (`raw/layout.py`): MediaPipe faces vote on where the recurring face is; per-pixel temporal variation (game busy, room calm) finds the overlay's edges. Fallback: a box around the face. Cached in `work/layout.json`; override with `--cam x,y,w,h`. Previews: `layout_detect.jpg`, `layout_preview.jpg`.
- Per clip, `--layout auto` picks: **split** (webcam top 35% / gameplay bottom 65%, captions on the seam) when the overlay is on screen; **blur** (whole 16:9 game centred over a blurred fill, captions below it) for gameplay without the webcam; **face** (upstream speaker-tracked crop) for talking-head scenes in VODs without an overlay. `--layout split|blur|face` forces one. Decisions are Luis's (2026-11-08).

### 6. export_premiere (new)
- Write `timeline.xml` in **FCP7 XML (xmeml)**, which Premiere Pro imports via File → Import.
- One sequence matching the clips' frame rate and resolution (1080×1920). Clips go in rank order, whole, with a 1-second gap between them.
- Add a **marker** on each clip with `hook_title` as the name and `reason` as the comment.
- Acceptance: Luis imports `timeline.xml` into Premiere on his Mac, media links without relinking, and the markers show up.

## CLI we want

```bash
autoclip raw <url-or-file> \
  --lang es \
  --chat auto|<path>|none \
  --campaign <profile-name> \
  --top 20 \
  --min 15 --max 60 \
  --provider ollama|gemini|openai \
  --style <caption-style> \
  --layout auto|split|blur|face \
  --cam x,y,w,h \
  --out ./out
```

Output layout:

```
out/<date>_<channel>_<vod-id>/
  clips/01_s92_el-grito-de-xqc.mp4 ...   (9:16, burned-in captions, ready to post)
  timeline.xml
  manifest.json
  manifest.csv
  spikes.json
  transcript/   (per-window transcripts)
  logs/
```

Campaign profiles live in `campaigns/<name>.yaml`: `min_len, max_len, allowed_platforms, credit_text, hashtags, notes`. The credit text is copied into the manifest so Luis can paste it exactly. Paraphrased credits get rejected.

## Milestones and acceptance criteria

| # | Milestone | Done when |
|---|---|---|
| M0 | Architecture notes | `docs/ARCHITECTURE_NOTES.md` exists and Luis has approved the plan |
| M1 | 9:16 export on a local file | ✅ Done: `autoclip raw video.mp4 --chat none` produces ranked 9:16 captioned clips and a manifest (the 16:9 raw export was built, then removed at Luis's request) |
| M2 | mlx-whisper backend | ✅ Done early (commit `ffcde3a`): mlx-whisper large-v3-turbo on the Apple GPU, VideoToolbox encode, language verification |
| M3 | Premiere XML | 🟡 Implemented for the 9:16 clips (hand-written xmeml: no extra dependency, full control of markers). Awaiting acceptance: `timeline.xml` imports cleanly into Premiere, with markers |
| M4 | Chat spikes | On a Twitch VOD with chat, only spike windows are transcribed. Output is logged as minutes processed vs total minutes |
| M5 | URL ingest + partial download | A Twitch or YouTube URL works end to end without downloading the full VOD |
| M6 | Campaign profiles | Length limits and credit text come from YAML |
| M7 | Gameplay layouts | ✅ Done: webcam overlay auto-detected; split / blur / face chosen per clip. Verified on a GTA V VOD (JuegaGerman, webcam top-right) |

Work milestone by milestone. Commit after each one with a clear message.

## Engineering conventions

- Python 3.11 or 3.12. **Not 3.13**, because upstream says MediaPipe has no 3.13 wheels. Use `uv` as upstream does.
- Put new code in its own package, e.g. `autoclip/raw/` (`spikes.py`, `windows.py`, `export_vertical.py`, `export_premiere.py`, `campaigns.py`). Keep edits to upstream files small and behind flags, so pulling upstream changes stays easy.
- Type hints everywhere. Use dataclasses or pydantic models for JSON schemas.
- Every new stage writes its artifact into the job work dir and is **resumable**, as upstream stages are.
- Tests with pytest:
  - Spike detection with a synthetic chat fixture
  - Time mapping from window-local to absolute time
  - XML generation, checked against a fixture
  - Filename slugging, including Spanish accents and emoji
- Use the system ffmpeg (`brew install ffmpeg-full`, per the upstream README). Call it through one helper that logs the exact command.
- Keep secrets in `.env` (gitignored). Never commit API keys.
- Use a small test VOD (5–10 min) in `samples/` (gitignored) for fast iteration.

## Guardrails

- Only process footage from campaigns or creators that authorize clipping.
- No features for auto-posting, fake engagement, mass account management, or tricks to evade duplicate detection (mirroring, speed changes and so on). Those break campaign and platform rules and get accounts banned.
- If a request conflicts with this file, ask Luis first.

## Open questions to raise with Luis

- What frame rate and resolution are the typical VODs? This affects the XML sequence settings.
- H.264 or ProRes by default? It's a trade-off between disk space and how smoothly Premiere edits.
- Which LLM provider by default: Gemini API (cents per VOD) or Ollama (free, slower, needs a model that fits in RAM)?
- Is a Kick chat source needed in v1?