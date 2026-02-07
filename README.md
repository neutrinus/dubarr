# dubarr 🎬

An automated system for creating professional AI-powered dubbing (voice-over) using **fully local** AI models. Optimized for home hardware (Multi-GPU), ensuring your data never leaves your machine. Supports arbitrary language pairs (e.g., English -> Polish, German -> Polish, etc.).

## 🎯 Project Goal
The primary objective of this system is to provide high-quality, fully automated dubbing for **feature-length movies and TV series**. Unlike short-form content tools, this architecture is designed to:
*   Handle complex narratives and maintain context over long durations (2h+).
*   Preserve character consistency across thousands of lines.
*   Retain the cinematic atmosphere (background audio, emotional tone) using purely local AI power.

## 🚀 Quick Start

### 1. Preparation
*   Place video files (.mkv/.mp4) in the `videos/` folder.
*   The LLM model must be located at `models/gemma-3-12b-it-Q4_K_M.gguf`.
*   Obtain a Hugging Face token (`HF_TOKEN`) with access to the Pyannote 3.1 model.

### 2. Build
Build the Docker image locally:
```bash
docker build -t dubarr .
```

### 3. Execution
```bash
docker run --rm --gpus all \
  -v $(pwd)/videos:/videos \
  -v $(pwd)/output:/output \
  -v $(pwd)/models:/models \
  -v $(pwd)/hf_cache:/root/.cache/huggingface \
  -v $(pwd)/tts_cache:/root/.local/share/tts \
  -e HF_TOKEN=YOUR_TOKEN \
  -e TARGET_LANGS=pl \
  -e DEBUG=1 \
  dubarr
```

---

## 🏗️ The Pipeline (Step-by-Step)

The system processes each video through 7 major local stages:

1.  **Audio Separation (Demucs):** Isolates vocals from background music and effects. Runs on **GPU 1**.
2.  **Audio Analysis (Whisper/Pyannote):**
    *   **Diarization:** Recognizes who speaks and when (**GPU 1**).
    *   **Transcription:** Converts speech to text using Whisper Large-v3 (**GPU 0**). Captures confidence levels (`avg_logprob`) for quality control.
3.  **Global Analysis (LLM Stage 1 & 2):** Gemma 3 12B (**GPU 0**) analyzes the full script:
    *   **Context Extraction:** Creates plot summaries and phonetic glossaries for names.
    *   **Speaker Profiling:** Identifies character names, gender, and voice traits.
4.  **Transcription Correction (Editor):** Fixes ASR errors while maintaining standard orthography.
5.  **Voice Reference & Dynamic Sampling (v2):**
    *   **Architecture Shift:** Migrated from "Golden Sample only" to a **Dynamic Multi-Level Reference** system.
    *   **The Sampling Hierarchy:**
        1.  **Dynamic (Current Segment):** The system extracts and cleans the original audio for the *specific line* being dubbed. If it meets quality criteria (ZCR < 0.25, duration > 0.8s), it's used as the primary reference. This ensures perfect emotional and tonal matching for every sentence.
        2.  **Rolling Cache (Last Good):** If the current segment is poor (e.g., loud music/noise), the system uses the *last successfully extracted clean sample* for that speaker.
        3.  **Global Reference (Golden Sample):** High-scoring (Score > 65) segments extracted during the initial analysis phase serve as a global fallback.
        4.  **Generic Fallback:** Built-in high-quality voices ("Daisy Morgan", "Damien Sayre") are used as a final safety net.

6.  **Production (Parallel Batch Pipeline):**
    *   **Batch Processing:** Groups lines in chunks of 10 to drastically reduce LLM prompt overhead and speed up processing.
    *   **Translator (Draft):** Translates text into target language JSON format.
    *   **Critic (QA Refiner):** Verifies translation, grammar, and phonetic glossary compliance.
    *   **Synthesis (XTTS):** Generates speech (**GPU 1**) with dynamic cloning and hallucination safeguards:
        *   *Dynamic Cloning:* Each line can use a unique reference sample for better variety and accuracy.
        *   *Tuned Parameters:* Increased temperature (0.75) and repetition penalty (1.2) to boost creativity and kill infinite loops.
        *   *Real-time Artifact Guard:* Measures **ZCR** of generated audio. If screeching is detected (> 0.25), automatically retries generation using a stable generic voice.
        *   *Hard Cut (Artifact Killer):* Automatically calculates the maximum plausible duration for a line (based on syllables) and ruthlessly cuts off any "hallucinated" audio tails. Tightened buffer (+1.5s) to eliminate foreign language bleed-in.
        *   *Short-Line Stability:* Automatically adds ellipsis padding (...) to lines shorter than 3 words to prevent XTTS from entering infinite hallucination loops.
        *   *Reverse Trimming:* Uses advanced FFmpeg filters (`areverse`) to robustly remove trailing silence/noise.
        *   *Room Tone (Reverb):* Adds a subtle `aecho` effect to movie characters to blend them into the scene. Narrators remain clean.
        *   *Dynamic Time Stretching:* Fine-tunes audio duration in FFmpeg (limit 1.25x).
7.  **Post-Production:**
    *   **Final Mix:** Dynamic ducking and stereo-separation (7% panning) for a rich audio field.
    *   **Muxing:** Assemble final video with a track named "AI - [Language]".

---

## 🔒 100% Local & Private
This project is designed for users who value privacy and control.
*   **No Cloud APIs:** No OpenAI, no ElevenLabs, no Google Cloud.
*   **Data Sovereignty:** Your videos and transcripts are never uploaded to any server.
*   **Hardware Efficiency:** Maximizes dual-GPU setups (e.g., RTX 3060 + 3070). Background threads are **daemonized** for clean process termination.

## 🔗 Similar Projects
If this project doesn't fit your needs, check out these excellent alternatives:
*   [Open Dubbing (Softcatala)](https://github.com/Softcatala/open-dubbing) - AI-based dubbing tool focusing on Catalan and open datasets.
*   [PyVideoTrans](https://github.com/jianchang512/pyvideotrans) - A comprehensive GUI-based video translation and dubbing tool.
*   [VideoTranslator](https://github.com/davy1ex/videoTranslator) - A tool for translating videos with various local/cloud engines.

## 📊 Debugging & Profiling
With `DEBUG=1`, the system generates:
*   `translations_draft_[lang].json` & `translations_final_[lang].json`: Full transparency of the translation process.
*   `segments/`: Individual audio files for every line, allowing you to audit XTTS quality.
*   `processing.log`: Now includes a **Real-time Resource Monitor** logging CPU, RAM, GPU Load/VRAM, and Queue depths (Text vs Audio) every 5 seconds.
*   `Profiling Report`: Detailed execution times for every pipeline stage.

## ⚡ Performance Architecture
To maximize hardware utilization, the system now runs on a **Fully Asynchronous Pipeline**:
1.  **Producer (GPU 0):** LLM translates and pushes text to `Q_Text`.
2.  **TTS Worker (GPU 1):** Pulls text, generates raw audio, checks ZCR (Quality), and pushes to `Q_Audio`.
3.  **Post-Processor (CPU):** Pulls raw audio, trims silence, masters dynamics (FFmpeg), and saves the file.
This ensures GPU 1 doesn't wait for CPU tasks.