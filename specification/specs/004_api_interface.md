# Spec 004: API Interface

## 1. Overview
The Dubarr API provides a REST interface for submitting video dubbing tasks, monitoring progress, and retrieving results. It also includes a WebSocket endpoint for real-time logs.

## 2. Base URL
-   **Local:** `http://localhost:8000` (Default)
-   **Production:** `http://dubarr.example.com` (Configurable)

## 3. Authentication
-   **Mechanism:** Basic Auth (`Authorization: Basic <base64_credentials>`)
-   **Credentials:** `API_USER` and `API_PASS` environment variables.

## 4. Endpoints

### 4.1 Task Management

#### `POST /upload`
Uploads a video file and queues it for processing.
-   **Method:** POST (Multipart Form-Data)
-   **Parameter:** `file` (Video file: `.mp4`, `.mkv`, etc.)
-   **Response:** `303 See Other` (Redirects to Dashboard).
-   **Task Metadata:**
    -   Extracts metadata (duration, audio languages) via FFmpeg.
    -   Stores file in `videos/`.
    -   Creates a `QUEUED` task.

#### `POST /webhook`
Registers an external file path for processing (e.g., from a shared volume).
-   **Method:** POST (JSON)
-   **Payload:**
    ```json
    {
        "path": "/absolute/path/to/video.mp4",
        "eventType": "optional_event_type"
    }
    ```
-   **Response:**
    ```json
    {
        "status": "queued",
        "path": "/absolute/path/to/video.mp4"
    }
    ```

#### `GET /download/{task_id}`
Downloads the processed video file.
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

### 4.2 Monitoring

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
WebSocket endpoint for streaming live logs from `logs/processing.log`.
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
