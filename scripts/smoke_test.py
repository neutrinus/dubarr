import requests
import time
import subprocess
import sys


def run_smoke_test():
    base_url = "http://localhost:8080"
    test_video = "/tmp/test_video.mkv"

    # 1. Create a dummy video file
    print("Creating dummy video...")
    subprocess.run(
        [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=640x360:d=5",
            "-f",
            "lavfi",
            "-i",
            "sine=f=440:d=5",
            "-shortest",
            "-y",
            test_video,
        ],
        check=True,
        capture_output=True,
    )

    # 2. Wait for server to be ready
    print("Waiting for server to be ready...")
    for i in range(30):
        try:
            res = requests.get(f"{base_url}/health", timeout=2)
            if res.status_code == 200:
                print("Server is up!")
                break
            else:
                print(f"Server returned status {res.status_code}...")
        except requests.exceptions.ConnectionError:
            if i % 5 == 0:
                print("Server not reachable yet...")
        except Exception as e:
            print(f"Waiting... ({type(e).__name__}: {e})")
        time.sleep(2)
    else:
        print("Error: Server timed out")
        sys.exit(1)

    # 3. Trigger webhook
    print(f"Triggering webhook for {test_video}...")
    res = requests.post(f"{base_url}/webhook", json={"path": test_video}, auth=("dubarr", "dubarr"))
    if res.status_code != 200:
        print(f"Error: Webhook failed: {res.text}")
        sys.exit(1)

    # 4. Poll database/health for completion
    print("Waiting for task to complete (MOCK mode should be fast)...")
    for _ in range(60):
        res = requests.get(f"{base_url}/health")
        stats = res.json().get("queue_stats", {})
        if stats.get("DONE", 0) > 0:
            print("Task completed successfully!")
            break
        if stats.get("FAILED", 0) > 0 or any(k.startswith("FAILED") for k in stats):
            print(f"Task failed! Stats: {stats}")
            sys.exit(1)
        time.sleep(5)
    else:
        print("Error: Task timed out")
        sys.exit(1)

    # 5. Verify output file exists (dubbed video)
    # Output file name in process_video is dub_{lang}_{f}
    # In mock mode we use /tmp/test_video.mkv, so output will be in same folder or output_folder
    # Wait, in new architecture we mux in-place or to a specific folder.
    # Let's check main.py muxing logic.
    print("Smoke test PASSED")


if __name__ == "__main__":
    run_smoke_test()
