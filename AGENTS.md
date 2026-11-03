# AGENTS.md

Guidance for AI coding agents (Claude Code, Codex, Gemini CLI, Cursor, ...) working in this repository.
Human contributors may find it useful too.

## What this project is

**dubarr** is a self-hosted service that adds AI-dubbed audio tracks ("AI - Polish", "AI - German", ...) to existing
video files. It is meant to sit next to the `*arr` stack (Radarr/Sonarr): a webhook queues a file, a background worker
runs the dubbing pipeline, and the result is muxed back into the original container in place (the original is kept as
`<file>.bak`).

Everything runs locally on the GPU(s) inside one Docker container. No cloud APIs.

## Technology stack

Python 3.12, FastAPI + SQLite + FFmpeg; AI: faster-whisper (ASR), pyannote (diarization), Demucs (separation),
OmniVoice (speech synthesis), llama.cpp / any OpenAI-compatible server (LLM). Everything runs in Docker Compose,
configured only through environment variables.

## Workflow for behaviour changes

1. Read the matching spec in `specification/specs/` (and `specification/ARCHITECTURE.md` for layering).
2. Update the spec first when a request changes behaviour, and get it accepted before writing the code. The spec
   change goes in the same PR as the code.
3. Implement, with tests, following the conventions below.
4. Run the checks (below) and, for pipeline changes, the E2E script.

Small bug fixes that do not change intended behaviour (typos, crashes) need no spec change, but must not regress one.

| Spec | Covers |
| --- | --- |
| `001_video_processing_pipeline.md` | Stages 1-8: separation, ASR, dialogue units, metadata, analysis, voice references, drafts, mix, mux. |
| `002_synchronization_engine.md` | Stage 5 sync loop: length variants, duration control, take judge, voice audition, reference order. |
| `003_task_management.md` | SQLite queue, step caching, resumption. |
| `004_api_interface.md` | HTTP endpoints, auth, dashboard. |
| `005_deployment_and_configuration.md` | Docker, volumes, GPU handling, **the full env var table**. |
| `006_series_memory.md` | Cross-episode glossary and voice bank. |

## Repository map

```
src/
  api/server.py            FastAPI app: dashboard, /webhook, /upload, /health, /ws/logs. Starts the JobWorker in lifespan.
  server.py                Container entrypoint (uvicorn on :8000). Docker CMD runs this from /app/src.
  main.py                  AIDubber: builds the shared engine managers; also a standalone "process ./videos" mode.
  config.py                Env-var config, paths, LANG_MAP. Has import-time side effects (see Gotchas).
  prompts.py               LLM prompt templates (analysis, ASR editor, translation, duration refinement).
  utils.py                 JSON parsing, SRT cleaning, ZCR, syllable counting, output cleaning.
  duration_policy.py       OmniVoice duration control (TTS_FILL_RATIO / TTS_MIN_SPEED / TTS_MAX_SPEEDUP), shared with the server.
  omnivoice_server.py      Flask OmniVoice TTS server. Separate process on the app's interpreter, 127.0.0.1:5051.
  verify_output.py         Manual QA helper (faster-whisper re-transcription of generated segments).
  core/
    pipeline.py            DubbingPipeline: stages 1-7, step caching, Stage 5 checkpointing, final mix/mux.
    synchronizer.py        Per-segment loop: pick length variant by estimate -> TTS -> measure -> other variant / LLM -> retry.
    llm_engine.py          LLMManager (llama-cpp-python Gemma 3 12B GGUF, or an external server): analysis, ASR fix, drafts, refinement.
    tts_manager.py         TTSManager: reference order per line (line clip, speaker refs, last good, designed voice) + TTS calls.
    whisper_engine.py      WhisperManager (faster-whisper, large-v3-turbo).
    diarization_engine.py  DiarizationManager (pyannote speaker-diarization-community-1, needs HF_TOKEN).
    audio.py               Demucs separation, analysis orchestration, ffmpeg mixing (ducking, ambient ghost).
    gpu_services.py        LLMService / TTSService: priority-queue worker threads that serialize GPU access.
    gpu_manager.py         Picks a CUDA device by free VRAM (maps nvidia-smi -> torch index by name+memory).
    media_context.py       Parses Sonarr/Radarr/scene file names, gathers show/episode or movie info, formats it for prompts.
    worker.py              JobWorker: polls SQLite for QUEUED tasks and runs a fresh pipeline per task.
    voice_audition.py      Before Stage 5: ranks each speaker's reference clips by the take judge's score of their clones.
    take_judge.py          Judges takes without Whisper: signal artifacts, voiceprint match, CTC text match (Spec 002).
    series_memory.py       Between episodes of a series: settled name spellings, terms and the voice bank (Spec 006).
    subtitles.py           Sidecar/embedded subtitles: cues, alignment to lines, missing speech, SDH names, gender from grammar.
    script_check.py        Stage 2c: unsure lines heard again, voiceprint speaker check; energy classes, shout/calm merge.
    report.py              Per-line production report: summary in the log, report_<lang>.tsv in the debug folder.
  infrastructure/
    database.py            SQLite: `tasks` queue + `job_steps` (cached JSON results per stage).
    ffmpeg.py              FFmpegWrapper: ffprobe metadata, extract/convert, mux.
    tts_client.py          OmniVoice server client: start/restart the server subprocess, HTTP calls.
    llm_client.py          OpenAI-compatible chat completions client (used when LLM_API_URL is set).
    metadata_client.py     .nfo parser, TVmaze (no key) and TMDB (optional key) clients for show/movie metadata.
    monitor.py             Periodic CPU/RAM/GPU/queue logging thread.
  templates/index.html     Dashboard.
tests/                     pytest unit tests (everything heavy is mocked).
scripts/                   smoke_test.py (CI, MOCK_MODE container), e2e_test.py (CPU E2E), demo/LLM helpers.
specification/             ARCHITECTURE.md + numbered feature specs (the intended behaviour).
```

## Pipeline at a glance

1. **Separation**: ffmpeg extracts stereo, Demucs `htdemucs_ft` isolates vocals.
2. **Analysis**: pyannote diarization on the vocals, then faster-whisper transcription of the original mix (models are loaded and unloaded one at a time).
   A second Whisper pass without VAD fills long gaps (speech under noise). With series memory on, speakers are matched
   against the series voice bank. `DubbingPipeline._create_script` assigns speakers to Whisper segments and joins them
   into dialogue units (whole sentences, each with a time budget).
2b. **Media metadata**: `core/media_context.py` identifies the show/episode or movie from the path (Sonarr/Radarr naming)
   and fetches synopsis and cast (character, actor, gender) from `.nfo` files, TVmaze or TMDB. Failures only log.
3. **Global analysis**: the LLM writes a summary, a phonetic glossary (reading-tested, fixed across a series) and
   speaker profiles, then corrects ASR errors in chunks (reference subtitles from `<video>.srt` or the first embedded
   subtitle stream are used as ground truth).
4. **Voice references**: up to `REF_AUDITION_CANDIDATES` scored, ZCR-checked clips per speaker, plus voice-bank clips
   from earlier episodes. Despite the step name "Transcription Correction", this stage does not correct text.
5. **Production**: draft translations per language (windows of 20 lines with surrounding context and per-line syllable
   budgets and short/long variants, then a completeness review pass), a designed voice per speaker and the voice
   audition (the take judge ranks each speaker's references), then a flat pool of (segment, language) jobs runs
   through `SegmentSynchronizer` on 4 orchestrator threads; every synthesis makes `TAKE_VERSIONS` takes and the judge
   keeps the best. LLM and TTS calls go through `LLMService`/`TTSService`.
   Progress is checkpointed to `stage5_checkpoint_<task_id>.json` in the task's workspace.
6. **Mix**: per-language mastering (pan, echo, atempo), then sidechain ducking against the original track.
7. **Mux**: the new tracks (AAC by default, `AUDIO_CODEC`) are added in front of the original audio, which is copied;
   the source file is swapped in place (the original stays as `<file>.bak`). After a successful task the series memory
   (glossary, voice bank) is updated.

Stages run through `DubbingPipeline._run_step`, which caches each result in `job_steps` and reuses it on retry.

## Setting up and running checks

Requires [uv](https://docs.astral.sh/uv/) and `ffmpeg` on PATH. The lightweight dev environment does **not** install
torch, llama-cpp, pyannote, faster-whisper or OmniVoice; those imports are optional and the code falls back to mocks.

```bash
uv sync --group dev --python 3.10

# Unit tests: MOCK_MODE=1 is required (otherwise config.py exits without HF_TOKEN).
# Point DATA_DIR somewhere writable; the default is /app/data.
MOCK_MODE=1 DATA_DIR=/tmp/dubarr-data PYTHONPATH=.:src uv run pytest tests/

# Lint: CI runs ruff check and flake8 (both must pass) and pylint (informational).
# Keep ruff format clean too; CI does not check it.
uv run ruff check .
uv run ruff format --check .
uv run flake8 .
```

CI (`.github/workflows/pipeline.yml`) runs the same lint + tests on Python 3.10, then builds the Docker image and runs
`scripts/smoke_test.py` against a container started with `MOCK_MODE=1`. Images are pushed to `ghcr.io` only from `main`
and tags.

`.github/workflows/e2e-cpu.yml` runs `scripts/e2e_test.py`: it cuts a 60 s public-domain clip (NASA), dubs it with the
real pipeline on a CPU runner using small models (Whisper `small`, Qwen2.5 3B GGUF, OmniVoice on CPU), and checks the
muxed tracks (count, tags, duration, not silent). It also builds `demo.mp4` (one line in English, then dubbed;
`scripts/demo_video.py`), and on `main` publishes it to the `demo` release that the README links to. It runs nightly, on push to `main`, on manual dispatch, and on PRs that
touch `src/`, dependencies, the workflow or the script. Without the `HF_TOKEN` secret, diarization is replaced by a single speaker. It tests
plumbing, not translation quality.

Running the real pipeline needs an NVIDIA GPU (roughly 10 GB for the in-process LLM, about 4 GB for Whisper, up to
6 GB for OmniVoice on long lines), `HF_TOKEN` with access to the pyannote models, and the Docker image. The image has a
single Python 3.12 venv (`/app/.venv`) with everything, OmniVoice included; the TTS server is started as a subprocess
on that same interpreter. Use `docker compose up --build`. OmniVoice is the only TTS: XTTS was removed, don't bring
back per-tool venvs.

## Configuration (environment variables)

All settings are environment variables, read in `src/config.py` (duration control in `src/duration_policy.py`).
The full reference is the table in `specification/specs/005_deployment_and_configuration.md`; the feature specs list
the settings that belong to them. When you add or change a variable, update that table (and `README.md` if users need
it) in the same PR. The ones you need most often while developing:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MOCK_MODE` | `0` | Skip all model loading; engines return canned output. Required for tests. |
| `DATA_DIR` | `/app/data` | DB (`queue.db`), logs, models, temp files, series memory. Point it somewhere writable in tests. |
| `HF_TOKEN` | none | Required unless `MOCK_MODE=1` (pyannote). |
| `DEBUG` / `VERBOSE` | `0` / `0` | Keep intermediate artifacts in `videos/debug_<name>/` (incl. `report_<lang>.tsv`); DEBUG log level. |
| `LLM_API_URL` / `LLM_API_MODEL` | none | Use an OpenAI-compatible server instead of the in-process GGUF. |
| `TAKE_CHECK` | `judge` | `judge` / `whisper` (judge plus Whisper similarity, for calibration) / `off`. |
| `SERIES_MEMORY` | `on` | Cross-episode glossary and voice bank in `DATA_DIR/series/<show>/` (Spec 006). |

## Development utilities

- `python src/main.py` (with `PYTHONPATH=.:src`) processes everything in `videos/` without the API or the DB worker —
  handy for debugging one file.
- `src/verify_output.py` re-transcribes generated segments with faster-whisper, a quick check of the audio without listening.
- `scripts/e2e_test.py` dubs a short clip end to end and verifies the muxed tracks (used by `.github/workflows/e2e-cpu.yml`).

## Conventions

- **Spec first for behaviour changes** (see the workflow above).
- **Layering** (see `specification/ARCHITECTURE.md`): HTTP and auth only in `src/api`; pipeline logic in `src/core`;
  subprocesses, DB and HTTP clients in `src/infrastructure`. In practice `core/` still calls `subprocess`/ffmpeg directly
  in several places; prefer adding to `FFmpegWrapper` over adding more of that.
- **Imports are flat from `src/`** (`from config import ...`, `from core.pipeline import ...`). Code runs with
  `PYTHONPATH=.:src` and the Docker workdir is `/app/src`. Don't switch to `src.`-prefixed imports piecemeal.
- **Every heavy engine must honour `MOCK_MODE`** and must import its ML library inside `try/except ImportError`, so the
  dev environment and CI stay light.
- **Long loops must check `abort_event`.** Every subprocess call should have a `timeout`.
- **GPU work goes through the services.** In Stage 5, LLM and TTS calls must be submitted via `LLMService`/`TTSService`.
  `LLMService` must stay at 1 worker: llama-cpp-python segfaults with concurrent CUDA inference.
- **GPU selection**: never use `nvidia-smi` indices as torch device indices; use `GPUManager.get_best_gpu`.
- Style: ruff, line length 127.
- Tests live in `tests/test_<module>.py`, use `unittest.mock` heavily, and must not need a GPU, network or model files.
  Add or update tests together with logic changes.
- **Commits:** small, logically cohesive changes with `feat:` / `fix:` / `refactor:` / `docs:` / `test:` / `ux:`
  prefixes. Do not mix unrelated features, fixes and docs in one commit. Push to the working branch and confirm CI
  (lint, unit tests, smoke test, nightly E2E) went green.

## Gotchas

- `config.py` runs at import time: it prints debug paths, reads env vars, and calls `sys.exit(1)` if `HF_TOKEN` is
  missing and `MOCK_MODE` is off. Set `MOCK_MODE=1` before importing anything from `src/` in tests or scripts.
- Each task works in `TEMP_DIR/task_<id>` (`standalone` without a task), set by `DubbingPipeline._enter_workspace`
  for the pipeline, the synchronizer and the shared `TTSManager`. It is deleted after a finished task (kept with
  `DEBUG=1` or after a failure, so a retry reuses cached steps). Write temp files to `self.temp_dir` / `work_dir`,
  never to `TEMP_DIR` directly. Per-segment files carry the language (`raw_<lang>_<idx>.wav`,
  `attempt_<lang>_<idx>_<n>.wav`); keep it that way, because target languages run concurrently and share segment indices.
- Cached step results are JSON in SQLite. `_run_step` only re-runs a step when a cached path no longer exists; the
  per-task workspace keeps those paths from pointing at another task's files.
- Prompts in `prompts.py` are system prompts sent as chat messages with `response_format=json_object`; the model's chat
  template (applied by llama-cpp-python or the server) adds the special tokens. Don't put ChatML tokens in them.
- The mux encodes only the dubbed tracks (AC3) and copies every original stream (audio, subtitles, MKV attachments).
  Dubbed tracks are tagged with ISO 639-2 codes (`config.ISO_639_2`, `pl` -> `pol`): MP4 drops two-letter codes.
- `SERIES_CONTEXT` is read in `config.py` but nothing uses it; the series comes from the media metadata.
- The OmniVoice server is restarted automatically on HTTP 500s and connection errors, then the request is retried once.
- `omnivoice_server.py` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before importing torch. It usually
  runs next to another GPU process (a shared `llama-server`), where the default allocator dies on 20 MB
  allocations even with memory free.
- Long lines are synthesized in ~`TTS_MAX_CHUNK_SECONDS` chunks split at sentence ends (then commas/semicolons, words
  only as a last resort). OmniVoice's VRAM peak grows with the length of the generated clip, and one long block also
  drifts out of sync with the original pauses.

## External compute node (glowonog)

Heavy work can be offloaded to the `glowonog` machine (10.252.1.122) over an OpenAI-compatible API:

| Service | Endpoint | Model | Notes |
| --- | --- | --- | --- |
| LLM (llama.cpp) | `http://10.252.1.122:8080/v1` | `gemma` | Gemma 4 26B A4B MoE, system prompts, JSON mode. No auth needed. |

Set `LLM_API_URL=http://10.252.1.122:8080/v1` and `LLM_API_MODEL=gemma` to use it instead of the in-process model.
There is no separate STT service there: transcription runs in-process with faster-whisper.
The node runs two GPUs (RTX 3060 12GB + RTX 3070 8GB) and usually hosts a long-context llama-server for the whole
house, so leave ~6GB free on the card that runs speech synthesis (see the hardware notes in `README.md`).
