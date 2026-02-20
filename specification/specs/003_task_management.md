# Spec 003: Task Management & Database

## 1. Overview
Dubarr uses a **Task Queue** system backed by SQLite to manage long-running video processing jobs. This ensures reliability, persistence, and the ability to resume interrupted tasks.

## 2. Database Schema
The database is located at `videos/dubarr.db` (default path).

### 2.1 Tasks Table (`tasks`)
Stores the primary metadata for each video processing request.

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | INTEGER PK | Unique Task ID. |
| `path` | TEXT UNIQUE | Absolute path to the source video file. |
| `status` | TEXT | Current state: `QUEUED`, `PROCESSING`, `DONE`, `ERROR`. |
| `target_langs` | TEXT | Comma-separated list of target languages (e.g., "pl,de"). |
| `source_lang` | TEXT | Detected source language (e.g., "en"). |
| `file_size` | INTEGER | Size in bytes. |
| `video_duration` | REAL | Duration in seconds. |
| `has_subtitles` | BOOLEAN | Whether the source file has embedded subtitles. |
| `started_at` | TIMESTAMP | Time when processing began. |
| `created_at` | TIMESTAMP | Time when task was added. |
| `updated_at` | TIMESTAMP | Last status update. |

### 2.2 Job Steps Table (`job_steps`)
Stores the result of each pipeline stage to enable caching and resumption.

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | INTEGER PK | Unique Step ID. |
| `task_id` | INTEGER FK | References `tasks.id`. |
| `step_name` | TEXT | Name of the stage (e.g., "Stage 1: Audio Separation"). |
| `status` | TEXT | `PENDING`, `DONE`, `ERROR`. |
| `result_data` | TEXT (JSON) | Serialized output of the stage (e.g., file paths, transcripts). |
| `error_msg` | TEXT | Error message if failed. |
| `updated_at` | TIMESTAMP | Last update. |

## 3. Workflow

### 3.1 Task Lifecycle
1.  **Creation:** API receives upload/webhook -> Inserts `QUEUED` task.
2.  **Pickup:** Worker thread polls for `QUEUED` tasks (FIFO).
3.  **Processing:**
    -   Worker updates status to `PROCESSING`.
    -   Pipeline executes stages.
    -   Each stage result is saved to `job_steps`.
4.  **Completion:**
    -   Success: Status -> `DONE`.
    -   Failure: Status -> `ERROR`.

### 3.2 Resumption Logic (Caching)
-   Before executing a stage, the pipeline checks `job_steps` for a `DONE` entry with the same `task_id` and `step_name`.
-   **Hit:** The cached result (`result_data`) is loaded, and the computation is skipped.
-   **Miss:** The stage executes normally, and the result is saved.
-   **Force Retry:** The API exposes an endpoint (`/purge/{task_id}`) to clear `job_steps` entries, forcing a full re-run.

### 3.3 Worker Reset
On startup, the application checks for tasks stuck in `PROCESSING` (e.g., due to a crash). These are automatically reset to `QUEUED` to be picked up again.
