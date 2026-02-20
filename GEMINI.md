# Gemini CLI Project Instructions

## 1. Project Context & Documentation
This project follows a **Spec-Driven Development** methodology. The source of truth for all features and architecture is located in the `specification/` directory.

-   **`specification/ARCHITECTURE.md`**: The core architectural rules (Layered Monolith). **Read this first.**
-   **`specification/specs/*.md`**: Detailed feature specifications.

## 2. Mandatory Workflow
Before implementing any code changes (features, refactors, or bug fixes that alter behavior), you **MUST** follow this process:

1.  **Read the Spec:** Identify the relevant specification file in `specification/specs/`.
2.  **Verify Alignment:** Check if your proposed changes align with the existing spec.
3.  **Update the Spec (If needed):**
    -   If the request changes logic or adds a feature, **you must update the markdown specification first**.
    -   Present the spec change to the user for confirmation.
    -   Only proceed to code implementation after the spec is updated and committed (or approved).
4.  **Implement:** Write code that strictly adheres to the updated specification.

**Exception:** Small bug fixes that do not change the intended behavior (e.g., typos, crash fixes) do not require a spec update, but should be cross-referenced with the spec to ensure no regression.

## 3. Architecture Rules
(See `specification/ARCHITECTURE.md` for full details)

-   **No Business Logic in API:** `src/api` should only handle HTTP/Auth. Logic goes to `src/core`.
-   **No External Calls in Core:** `src/core` should use interfaces/managers. `src/infrastructure` handles I/O.
-   **Docker First:** The application runs in Docker. Configuration is via Environment Variables.
-   **Test-Driven:** Always add/update tests in `tests/` for new logic.

## 4. Technology Stack
-   **Language:** Python 3.10+
-   **Framework:** FastAPI (API), SQLite (DB), FFmpeg (Media)
-   **AI/ML:** Whisper (ASR), Pyannote (Diarization), XTTS (TTS), Llama.cpp/OpenAI (LLM).
-   **Deployment:** Docker Compose.

## 5. Local Development Caveats
**Critical:** This project uses a **Dual-Environment Architecture** to support Coqui-TTS (which requires Python 3.10) alongside the main application (Python 3.12).
-   **Docker:** Handles this automatically.
-   **Local Run:** You CANNOT simply run `pip install .` and start the server. You must:
    1.  Create a Python 3.12 venv for the main app.
    2.  Create a separate Python 3.10 venv for TTS.
    3.  Manually run `src/tts_server.py` in the TTS venv (port 5050).
    4.  Run `src/api/server.py` in the main venv.

## 6. Development Utilities
-   **`src/verify_output.py`**: Manual script to verify audio integrity and transcription quality of generated segments using `faster-whisper`.
-   **`src/main.py` (Standalone Mode)**: Can be run directly (`python src/main.py`) to process videos in the `videos/` folder without the API or Database worker. Useful for quick debugging.

## 7. Common Commands
-   **Run Tests:** `pytest`
-   **Lint:** `ruff check .`
-   **Format:** `ruff format .`
-   **Run Server (Dev):** `uvicorn src.api.server:app --reload`

## 8. Completion & Source Control
Once a functionality or bug fix is implemented and successfully verified through tests and manual validation:
1.  **Prepare Commits:** Group related changes into logical, concise commits. Follow the existing commit message style (e.g., `feat:`, `fix:`, `docs:`, `ux:`).
2.  **Push Changes:** Push the committed changes to the remote repository (`main` or the current feature branch) to ensure the codebase remains up to date.
3.  **Final Status:** Confirm that the push was successful and that the CI/CD pipeline (if applicable) is triggered.
