import os
import logging

from config import (
    DEBUG_MODE,
    setup_logging,
    VIDEO_FOLDER,
    TEMP_DIR,
    MODEL_PATH,
    DEVICE_LLM,
    DEVICE_AUDIO,
    USE_LOCK,
    TARGET_LANGS,
)
from core.llm_engine import LLMManager
from core.tts_manager import TTSManager
from core.pipeline import DubbingPipeline

setup_logging()
logger = logging.getLogger(__name__)


class AIDubber:
    def __init__(self):
        logger.info("AIDubber: Initializing components...")
        self.video_folder = VIDEO_FOLDER
        self.target_langs = TARGET_LANGS
        self.debug_mode = DEBUG_MODE

        # Shared resources
        import threading

        self.inference_lock = threading.Lock() if USE_LOCK else None

        # Components (Dependency Injection ready)
        self.llm_manager = LLMManager(
            model_path=MODEL_PATH,
            device=DEVICE_LLM,
            inference_lock=self.inference_lock,
            debug_mode=self.debug_mode,
            target_langs=self.target_langs,
        )

        # Golden speaker refs shared across languages
        self.speaker_refs = {}
        self.tts_manager = TTSManager(
            device=DEVICE_AUDIO,
            inference_lock=self.inference_lock,
            temp_dir=TEMP_DIR,
            speaker_refs=self.speaker_refs,
            abort_event=threading.Event(),  # Placeholder, pipeline has its own
        )

    def process_video(self, f, task_id=None):
        """Orchestrates the pipeline for a single video."""
        from infrastructure.database import Database
        from config import DB_PATH

        pipeline = DubbingPipeline(
            llm_manager=self.llm_manager,
            tts_manager=self.tts_manager,
            target_langs=self.target_langs,
            db=Database(DB_PATH),
            debug_mode=self.debug_mode,
        )
        pipeline.process_video(f, task_id=task_id)

    def run(self):
        """Main loop for scanning the video folder."""
        try:
            for f in os.listdir(self.video_folder):
                if f.endswith((".mkv", ".mp4", ".mov")):
                    self.process_video(f)
        except Exception:
            logger.exception("FATAL ERROR in main loop")


if __name__ == "__main__":
    try:
        AIDubber().run()
    except Exception:
        logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
        logger.exception("FATAL ERROR")
        raise
