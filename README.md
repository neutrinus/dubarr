# dubarr 🎬

> [!NOTE]
> To jest projekt hobbystyczny ("zabawka"), stworzony w całości w trybie **vibe-coding** (AI-driven development).

An automated system for creating professional AI-powered dubbing (voice-over) using **fully local** AI models. Designed to integrate seamlessly with the `*arr` stack (Radarr/Sonarr) for feature-length movies and TV series.

**100% Private. 100% Local. No Cloud APIs.**

---

## 🌟 Key Features
*   **Feature-Length Optimization**: Designed for 2h+ movies, maintaining character consistency and context throughout.
*   **Character Consistency**: Uses advanced diarization to identify speakers and assign consistent AI voices.
*   **Cinematic Quality**: Isolates original background music/effects and mixes them with the new AI voice track.
*   **Subtitle-Assisted Transcription**: Automatically detects and uses external or embedded subtitles to ensure perfect spelling of names and terms.
*   **In-Place Processing**: Adds a new "AI - [Language]" audio track to your existing files without overwriting originals.
*   **Multi-GPU Support**: Optimized to split LLM, Audio Analysis, and TTS tasks across multiple cards for maximum speed.
*   **Automatic Strategy Selection**: Scales from Multi-GPU setups to Single-GPU or CPU-only modes based on detected VRAM.

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
      - 8080:8080
    environment:
      - PUID=1000
      - PGID=1000
      - HF_TOKEN=your_token_here
      - TARGET_LANGS=pl
    volumes:
      - ./config:/config        # Queue database and settings
      - ./logs:/logs            # Processing logs and debug artifacts
      - ./models:/app/models    # LLM and AI models
      - /path/to/movies:/movies
      - /path/to/tv:/tvseries
      - dubarr_cache:/data/cache # Persist model downloads
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped

volumes:
  dubarr_cache:
```

### 3. Usage
`dubarr` works as a service. You send tasks via a Webhook.

**Manual Trigger (cURL):**
```bash
curl -X POST http://localhost:8080/webhook \
  -H "Content-Type: application/json" \
  -d '{"path": "/movies/Diuna/Dune.mkv"}'
```

**Radarr/Sonarr Integration:**
1.  Go to **Settings -> Connect**.
2.  Add a **Webhook** or **Custom Script**.
3.  Set the URL to `http://dubarr-ip:8080/webhook`.
4.  Trigger on **On Download** or **On Upgrade**.

---

## 🏗️ The Pipeline (Step-by-Step)

The system processes each video through 7 major stages:

1.  **Audio Separation (Demucs)**: Isolates vocals from background music and effects.
2.  **Audio Analysis (Whisper/Pyannote)**:
    *   **Diarization**: Recognizes who speaks and when.
    *   **Transcription**: Converts speech to text using Whisper Large-v3.
3.  **Global Analysis (LLM Stage 1 & 2)**: Gemma 3 12B analyzes the full script to create plot summaries and phonetic glossaries for characters.
4.  **Transcription Correction (Editor)**: Fixes ASR errors while maintaining standard orthography. Uses **reference subtitles** (if found) as ground truth for proper nouns.
5.  **Dynamic Voice Sampling**:
    *   **Current Segment**: Extracts original audio for each specific line to match tone/emotion.
    *   **Rolling Cache**: Falls back to the last successfully extracted clean sample for that speaker if the current segment is noisy.
    *   **Golden Sample**: Uses a pre-vetted high-quality sample as a global fallback.
6.  **Production**:
    *   **Translator**: Adapts text for dubbing (conciseness, grammar).
    *   **Synthesis (XTTS/F5-TTS)**: Generates the new audio track with cloned voices and real-time artifact guards (ZCR checking).
7.  **Post-Production**:
    *   **Final Mix**: Dynamic ducking and stereo-separation for a rich audio field.
    *   **Muxing**: Assembles the final MKV with the new track named "AI - [Language]".

---

## 📊 Monitoring & Debugging
*   **Web Dashboard**: Access the management interface at `http://localhost:8080/` (or your server's IP) to view the task queue, retry failed tasks, or monitor worker status.
*   **Logs**: Check `logs/processing.log` for a real-time resource monitor (CPU, RAM, GPU, Queue depths).
*   **Artifacts**: With `DEBUG=1`, individual audio segments and LLM translations are saved in `logs/` for quality auditing.

---

## 🔗 Similar Projects
*   [Open Dubbing (Softcatala)](https://github.com/Softcatala/open-dubbing) - Focusing on open datasets and Catalan.
*   [PyVideoTrans](https://github.com/jianchang512/pyvideotrans) - Comprehensive GUI-based tool.
*   [VideoTranslator](https://github.com/davy1ex/videoTranslator) - Various local/cloud engine support.

---

## ⚖️ License
[LICENSE](LICENSE)
