# dubarr 🎬

> [!NOTE]
> This is a hobby project ("toy"), created entirely in **vibe-coding** mode (AI-driven development).

An automated system for creating professional AI-powered dubbing (voice-over) using **fully local** AI models. Designed to integrate seamlessly with the `*arr` stack (Radarr/Sonarr) for feature-length movies and TV series.

**100% Private. 100% Local. No Cloud APIs.**

---

## 🌟 Key Features
*   **In-Place Multi-Track Processing**: Adds new "AI - [Language]" audio tracks to your existing files (MKV, MP4, MOV) without overwriting original audio.
*   **Feature-Length Optimization**: Designed for 2h+ movies, maintaining character consistency and context throughout.
*   **Intelligent Audio Sync**: Uses duration-aware translation and "Ripple Shifting" to prevent audio overlaps and maintain perfect lip-sync.
*   **Character Consistency**: Uses advanced diarization to identify speakers and assign consistent AI voices.
*   **Cinematic Quality**: Implements EBU R128 normalization, aggressive ducking (12:1), and "Ambient Ghosting" (preserving room acoustics) to blend AI voices perfectly with original background.
*   **Subtitle-Assisted Transcription**: Automatically detects and uses external or embedded subtitles to ensure perfect spelling of names and terms.
*   **Multi-Format Support**: Works with MKV, MP4, and MOV containers, preserving the original container format.
*   **Multi-GPU Support**: Optimized to split LLM, Audio Analysis, and TTS tasks across multiple cards for maximum speed.
*   **Automatic Strategy Selection**: Scales from Multi-GPU setups to Single-GPU or CPU-only modes based on detected VRAM.

---

## 🎬 Demo

Want to see it in action? Check out our comparison demo where we take an original English clip and show the AI Dubbing results for Polish, French, and German sequentially.

[![dubarr demo](https://img.youtube.com/vi/MkJtJ11j_BU/0.jpg)](https://www.youtube.com/watch?v=MkJtJ11j_BU)

> [!TIP]
> Notice how the AI maintains the same voice characteristics and emotional tone across all target languages while blending with the original background atmosphere.

---

## 💻 Hardware Requirements

The system automatically detects your hardware and chooses the best processing strategy.

| Setup | VRAM | Strategy | Description |
| :--- | :--- | :--- | :--- |
| **Multi-GPU** | > 18GB (Combined) | **PARALLEL** | LLM and TTS run simultaneously on separate cards. Peak performance. |
| **Single GPU** | > 18GB (e.g. 3090/4090) | **PARALLEL** | Everything runs on one card simultaneously. |
| **Single GPU** | < 18GB (e.g. 3060 12GB) | **SEQUENTIAL** | Uses an **Inference Lock** to swap between models, preventing Out-Of-Memory errors. |
| **CPU Only** | N/A (32GB RAM rec.) | **SEQUENTIAL** | Models run on CPU using 8-bit quantization. Significantly slower (10-20x). |

---

## ⚡ Performance & Speed

Processing time depends on your hardware and the number of target languages.

*   **Rule of Thumb**: Expect processing to take approximately **2x to 3x the video duration** per language on a modern GPU (e.g., RTX 3090/4090).
*   **Example**: A 10-minute video dubbed into 3 languages (`pl`, `de`, `fr`) takes about 20-30 minutes to complete.
*   **Parallelism**: If you have multiple GPUs, the system will automatically utilize them to speed up LLM and TTS tasks simultaneously.

---

## 🚀 Getting Started

### 1. Prerequisites
*   **Hugging Face Token**: Required for Pyannote 3.1 (Diarization). [Accept terms here](https://huggingface.co/pyannote/speaker-diarization-3.1).

### 2. Deployment (Docker Compose)
The easiest way to run `dubarr` is using Docker Compose. The image is automatically built and published to `ghcr.io`.

```yaml
services:
  dubarr:
    image: ghcr.io/neutrinus/dubarr:main
    container_name: dubarr
    ports:
      - 8000:8000
    environment:
      - PUID=1000
      - PGID=1000
      - HF_TOKEN=your_token_here
      - TARGET_LANGS=pl
      - DATA_DIR=/app/data # Mount point for all app data (DB, logs, models, caches)
      - API_USER=dubarr # Optional: Custom API username
      - API_PASS=dubarr # Optional: Custom API password
    volumes:
      - ./data:/app/data          # Persistent storage (DB, logs, models, caches)
      - /path/to/media:/app/videos # Your movie/tv library
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
```

### 3. Usage
`dubarr` works as a service. You send tasks via a Webhook.

**Manual Trigger (cURL):**
```bash
curl -u dubarr:dubarr -X POST http://localhost:8000/webhook \
  -H "Content-Type: application/json" \
  -d '{"path": "/movies/Diuna/Dune.mkv"}'
```

---

## 🏗️ The Pipeline (Step-by-Step)

The system processes each video through 7 major stages:

1.  **Stage 1: Audio Separation (Demucs)**: Uses `htdemucs_ft` to isolate vocals from background music and effects.
2.  **Stage 2: Audio Analysis (Whisper/Pyannote)**:
    *   **Diarization**: Recognizes who speaks and when (using `pyannote/speaker-diarization-community-1`).
    *   **Transcription**: Converts speech to text using Whisper Large-v3.
3.  **Stage 3: Global Analysis (LLM Stage 1 & 2)**: Gemma 3 12B analyzes the full script to create plot summaries and phonetic glossaries for characters.
4.  **Stage 4: Transcription Correction (Editor)**: Fixes ASR errors while maintaining standard orthography. Uses **reference subtitles** (if found) as ground truth for proper nouns.
5.  **Stage 5: Production**:
    *   **Duration-Aware Translation**: Adapts text for dubbing while strictly respecting the available time window (syllable-count constraints).
    *   **Synthesis (XTTS v2)**: Generates the new audio track with cloned voices and real-time artifact guards (ZCR checking).
6.  **Stage 6: Final Mix**: Aggressive sidechain ducking (12:1 ratio) and Ambient Ghosting (low-pass original audio) for a rich, cinematic audio field.
7.  **Stage 7: Muxing**: Assembles the final video with all new tracks named "AI - [Language]".

---

## 📊 Monitoring & Debugging
*   **Web Dashboard**: Access the management interface at `http://localhost:8000/` to view the task queue, monitor video metadata (size, duration, languages), and track live processing progress with real-time timers.
*   **Live Logs**: Integrated WebSocket-based log viewer in the dashboard.
*   **Artifacts**: With `DEBUG=1`, individual audio segments and LLM translations are saved alongside the video for quality auditing.

---

## 🔗 Similar Projects
*   [Open Dubbing (Softcatala)](https://github.com/Softcatala/open-dubbing) - Focusing on open datasets and Catalan.
*   [PyVideoTrans](https://github.com/jianchang512/pyvideotrans) - Comprehensive GUI-based tool.
*   [VideoTranslator](https://github.com/davy1ex/videoTranslator) - Various local/cloud engine support.

---

## ⚖️ License
[LICENSE](LICENSE)
