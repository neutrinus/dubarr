# Spec 004: API Interface

## 1. Overview
The Dubarr API provides a REST interface for submitting video dubbing tasks, monitoring progress, and retrieving results. It also includes a WebSocket endpoint for real-time logs.

## 2. Base URL
`http://<host>:8000` (`PORT` changes the port).

## 3. Authentication
-   **Mechanism:** Basic Auth (`Authorization: Basic <base64_credentials>`)
-   **Credentials:** `API_USER` and `API_PASS` environment variables (default `dubarr` / `dubarr`).
-   **Protected:** `/upload`, `/webhook`, `/download`, `/debug`. The dashboard, `/health`, `/ws/logs` and the
    `/retry`, `/purge`, `/delete` form actions have no auth; dubarr is meant for a trusted home network.

## 4. Endpoints

### 4.1 Task Management

#### `POST /upload`
Uploads a video file and queues it for processing.
-   **Method:** POST (Multipart Form-Data)
-   **Parameter:** `file` (Video file: `.mp4`, `.mkv`, etc.)
-   **Response:** `303 See Other` (Redirects to Dashboard).
-   **Task Metadata:**
    -   Extracts metadata (duration, audio languages) via FFmpeg.
    -   Stores the file in `/app/videos/` (the app's `videos` folder); an existing file of the same name is refused.
    -   Creates a `QUEUED` task.

#### `POST /webhook`
Registers an existing file for processing. The path must exist inside the container; a relative path is resolved
against `/app/videos`. Two payloads are accepted:
-   `{"path": ...}` (scripts, cURL);
-   the Sonarr/Radarr built-in **Webhook** connection: the file is `movieFile.path` / `episodeFile.path`, else the
    `relativePath` joined with `movie.folderPath` / `series.path`. The `Test` event (sent when the connection is
    saved) returns `{"status": "test"}`; other events without a file (Grab, Rename, ...) return `{"status": "ignored"}`.
    Neither queues anything. Use the *On File Import* and *On File Upgrade* triggers.
-   **Method:** POST (JSON)
-   **Payload:** `{"path": "/absolute/path/to/video.mp4"}`, or a Sonarr/Radarr webhook body.
-   **Response:** `{"status": "queued" | "re-queued" | "ignored", "path": "<absolute path>"}`. A known path that is
    finished or failed is queued again; one that is queued or processing is ignored.

#### `GET /download/{task_id}`
Downloads the task's file (after a successful task, the dubbed file that replaced the source in place).
-   **Method:** GET
-   **Auth Required:** Yes
-   **Response:**
    -   **200:** File stream (`application/octet-stream`).
    -   **404:** Task or file not found.

#### `POST /retry/{task_id}`
Retries a failed task. Preserves cached steps.
-   **Method:** POST
-   **Response:** `303 See Other`.

#### `POST /purge/{task_id}`
Clears cached steps for a task (forces full re-processing).
-   **Method:** POST
-   **Response:** `303 See Other`.

#### `POST /delete/{task_id}`
Deletes a task from the database. Does *not* delete the file on disk.
-   **Method:** POST
-   **Response:** `303 See Other`.

### 4.2 Debug & Artifacts

#### `GET /debug/{task_id}/{artifact_name}`
Retrieves intermediate processing results (JSON artifacts).
-   **Method:** GET
-   **Artifact Names:**
    -   `transcription`: Raw transcription with timestamps.
    -   `context`: Global story context and glossary.
    -   `speakers`: Identified speaker profiles.
    -   `media`: Media metadata (Stage 2b).
    -   `production`: Final translated segments per language.
-   **Response:** `200 OK` (JSON).

### 4.3 Monitoring

#### `GET /health`
Returns system health status and queue statistics.
-   **Method:** GET
-   **Response:**
    ```json
    {
        "status": "online",
        "worker_alive": true,
        "queue_stats": {
            "QUEUED": 2,
            "PROCESSING": 1,
            "DONE": 10,
            "ERROR": 1
        },
        "engine_statuses": {
            "llm": "READY",
            "tts": "READY",
            "whisper": "READY",
            "diarization": "READY"
        }
    }
    ```

#### `WS /ws/logs`
WebSocket endpoint for streaming live logs from `DATA_DIR/logs/processing.log`.
-   **Protocol:** WebSocket
-   **Behavior:**
    -   Sends last 50 lines on connect.
    -   Streams new lines as they are written.

## 5. Web Interface
-   **Dashboard (`/`)**:
    -   Lists all tasks sorted by creation date.
    -   Shows progress bars (completed stages).
    -   Provides action buttons (Retry, Delete, Download).
    -   Displays system status (Worker, Engines).
