import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import os

# Set environment variables for tests before importing the app
os.environ["DATA_DIR"] = "/tmp/dubarr_test_data"
os.environ["API_USER"] = "testuser"
os.environ["API_PASS"] = "testpass"
os.environ["HF_TOKEN"] = "testtoken"
os.environ["MOCK_MODE"] = "1"

with patch("infrastructure.database.Database"), patch("main.AIDubber"), patch("core.worker.JobWorker"):
    from api.server import app

client = TestClient(app)


@pytest.fixture
def mock_db():
    with patch("api.server.db") as mock:
        yield mock


def test_health_check(mock_db):
    mock_db.get_queue_stats.return_value = {"QUEUED": 1, "DONE": 5}
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["queue_stats"]["QUEUED"] == 1


def test_auth_failure():
    response = client.get("/download/1")
    assert response.status_code == 401


def test_dashboard(mock_db):
    mock_db.get_all_tasks.return_value = [
        {
            "id": 1,
            "path": "test.mp4",
            "status": "DONE",
            "progress": 7,
            "started_at": None,
            "created_at": "2026-02-20 12:00:00",
            "file_size": 100,
            "video_duration": 60,
            "source_lang": "en",
            "target_langs": "pl",
            "has_subtitles": False,
        }
    ]
    mock_db.get_task_progress.return_value = 7
    # Auth is needed for some parts but root dashboard usually isn't protected in this app?
    # Let's check api/server.py
    response = client.get("/")
    assert response.status_code == 200
    assert "test.mp4" in response.text


def test_webhook_success(mock_db):
    with patch("infrastructure.ffmpeg.FFmpegWrapper.get_metadata", return_value={}):
        with patch("os.path.getsize", return_value=1000):
            mock_db.add_task.return_value = "queued"
            response = client.post("/webhook", json={"path": "/videos/movie.mp4"}, auth=("testuser", "testpass"))
            assert response.status_code == 200
            assert response.json()["status"] == "queued"


def test_delete_task(mock_db):
    response = client.post("/delete/1", follow_redirects=False)
    assert response.status_code == 303  # Redirects to /
    mock_db.delete_task.assert_called_once_with(1)
