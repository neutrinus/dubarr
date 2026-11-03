"""What dubarr remembers about a series between its episodes (Spec 006): settled name spellings and the voice bank.

Everything is automatic; the user never curates it. Files live in ``DATA_DIR/series/<key>/`` and are written
atomically at the end of a successful task.
"""

import json
import logging
import os
import re
import tempfile
from typing import Dict, Optional

from config import DATA_DIR, SERIES_MEMORY

logger = logging.getLogger(__name__)

SERIES_DIR = os.path.join(DATA_DIR, "series")


def series_key(media_info: Optional[Dict]) -> Optional[str]:
    """Folder name of the series of an episode ("BattleBots" -> "battlebots"); None for movies and unknown titles."""
    if not media_info or media_info.get("kind") != "episode":
        return None
    key = re.sub(r"[^a-z0-9]+", "-", (media_info.get("title") or "").lower()).strip("-")
    return key or None


def _read_json(path: str) -> Dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        logger.warning(f"Series memory: cannot read {path} ({e}); starting from scratch")
        return {}


def _write_json(path: str, data: Dict):
    """Temp file + rename, so a crash never leaves half a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


class SeriesMemory:
    def __init__(self, key: str, root: str = SERIES_DIR):
        self.key = key
        self.dir = os.path.join(root, key)
        self.glossary_path = os.path.join(self.dir, "glossary.json")

    @classmethod
    def for_media(cls, media_info: Optional[Dict], root: str = SERIES_DIR) -> Optional["SeriesMemory"]:
        if not SERIES_MEMORY:
            return None
        key = series_key(media_info)
        return cls(key, root) if key else None

    # --- Glossary (Spec 006, 3) ---

    @staticmethod
    def lang_key(target_langs) -> str:
        return ",".join(target_langs)

    def glossary(self, lang_key: str) -> Dict[str, str]:
        """Settled spellings {term: spelling} for these target languages."""
        entries = _read_json(self.glossary_path).get(lang_key) or {}
        return {t: e["spelling"] for t, e in entries.items() if isinstance(e, dict) and e.get("checked") and e.get("spelling")}

    def save_glossary(self, lang_key: str, glossary: Dict[str, str], used_terms=None):
        """Stores the checked spellings of an episode; a stored spelling is never replaced, only counted."""
        data = _read_json(self.glossary_path)
        entries = data.setdefault(lang_key, {})
        used = set(used_terms) if used_terms is not None else set(glossary)
        added = 0
        for term, spelling in glossary.items():
            if not spelling:
                continue
            entry = entries.get(term)
            if entry is None:
                entries[term] = {"spelling": spelling, "checked": True, "episodes": 1}
                added += 1
            elif term in used:
                entry["episodes"] = int(entry.get("episodes", 0)) + 1
        _write_json(self.glossary_path, data)
        logger.info(f"Series memory '{self.key}': glossary has {len(entries)} names ({added} new)")
