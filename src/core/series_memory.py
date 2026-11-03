"""What dubarr remembers about a series between its episodes (Spec 006): settled name spellings and the voice bank.

Everything is automatic; the user never curates it. Files live in ``DATA_DIR/series/<key>/`` and are written
atomically at the end of a successful task.
"""

import json
import logging
import math
import os
import re
import shutil
import tempfile
import uuid
from typing import Dict, List, Optional

from config import DATA_DIR, SERIES_MEMORY, VOICE_BANK_CLIPS, VOICE_MATCH_MIN
from infrastructure.tts_client import ref_text_path

logger = logging.getLogger(__name__)

SERIES_DIR = os.path.join(DATA_DIR, "series")
# A match must beat the second best person by this much: two similar commentators must not merge
VOICE_MATCH_MARGIN = 0.05


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

    # --- Voice bank (Spec 006, 4) ---

    @property
    def voices_path(self) -> str:
        return os.path.join(self.dir, "voices.json")

    @property
    def clips_dir(self) -> str:
        return os.path.join(self.dir, "clips")

    def persons(self) -> Dict[str, Dict]:
        return _read_json(self.voices_path).get("persons") or {}

    def match(self, prints: Dict[str, List[float]]) -> Dict[str, str]:
        """Diarization speaker -> person id, for speakers whose voiceprint clearly matches a known person."""
        persons = {pid: p for pid, p in self.persons().items() if p.get("print")}
        matches = {}
        for spk, emb in (prints or {}).items():
            ranked = sorted(((_cosine(emb, p["print"]), pid) for pid, p in persons.items()), reverse=True)
            if not ranked:
                continue
            best, pid = ranked[0]
            runner_up = ranked[1][0] if len(ranked) > 1 else -1.0
            if best >= VOICE_MATCH_MIN and best - runner_up >= VOICE_MATCH_MARGIN:
                matches[spk] = pid
                logger.info(f"Series memory '{self.key}': {spk} is {pid} ({persons[pid].get('name') or '?'}, {best:.2f})")
        return matches

    def hints(self, matches: Dict[str, str]) -> str:
        """Known persons of the series, for the analysis prompt."""
        persons = self.persons()
        lines = []
        for spk, pid in sorted(matches.items()):
            p = persons.get(pid) or {}
            facts = ", ".join(x for x in (p.get("name"), p.get("gender"), p.get("desc")) if x)
            lines.append(f"- {spk} is probably {facts or 'a known person'} (known from {p.get('episodes', 1)} episodes)")
        return "Known speakers of this series (from their voices):\n" + "\n".join(lines) if lines else ""

    def settled_genders(self, matches: Dict[str, str]) -> Dict[str, str]:
        """Genders known from at least 2 episodes are not guessed again."""
        persons = self.persons()
        out = {}
        for spk, pid in matches.items():
            p = persons.get(pid) or {}
            if p.get("gender") in ("male", "female") and int(p.get("episodes", 0)) >= 2:
                out[spk] = p["gender"]
        return out

    def clips(self, pid: str) -> List[str]:
        """The person's stored reference clips, best first."""
        p = self.persons().get(pid) or {}
        paths = [os.path.join(self.clips_dir, c["file"]) for c in p.get("clips") or [] if c.get("file")]
        return [x for x in paths if os.path.exists(x)]

    def save_voices(self, speakers: Dict[str, Dict]):
        """Updates the bank after a successful episode.

        ``speakers``: spk -> {"person": id or None, "print": voiceprint or None, "profile": {name, gender, desc},
        "clips": [(path, audition score)]}. Speakers without a print and without a match are skipped.
        """
        data = _read_json(self.voices_path)
        persons = data.setdefault("persons", {})
        os.makedirs(self.clips_dir, exist_ok=True)
        for spk, info in sorted(speakers.items()):
            pid = info.get("person")
            emb = info.get("print")
            if pid not in persons:
                if not emb:
                    continue
                pid = f"V{len(persons) + 1:03d}"
                while pid in persons:
                    pid = f"V{int(pid[1:]) + 1:03d}"
                persons[pid] = {"print": None, "episodes": 0, "clips": []}
            person = persons[pid]
            n = int(person.get("episodes", 0))
            if emb:
                old = person.get("print")
                person["print"] = [round((o * n + e) / (n + 1), 6) for o, e in zip(old, emb)] if old else list(emb)
            person["episodes"] = n + 1
            for key in ("name", "gender", "desc"):
                value = (info.get("profile") or {}).get(key)
                if value and value != "unknown" and not person.get(key):
                    person[key] = value
            self._merge_clips(pid, person, info.get("clips") or [])
        _write_json(self.voices_path, data)
        logger.info(f"Series memory '{self.key}': voice bank has {len(persons)} persons")

    def _merge_clips(self, pid: str, person: Dict, clips):
        """The best VOICE_BANK_CLIPS clips by audition score, from the bank and this episode."""
        entries = {c["file"]: c for c in person.get("clips") or [] if c.get("file")}
        for path, score in clips:
            if score is None or not path or not os.path.exists(path):
                continue
            if os.path.dirname(os.path.abspath(path)) == os.path.abspath(self.clips_dir):
                name = os.path.basename(path)  # a bank clip auditioned again: its score is updated
            else:
                name = f"{pid}_{uuid.uuid4().hex[:8]}.wav"
                shutil.copy(path, os.path.join(self.clips_dir, name))
                if os.path.exists(ref_text_path(path)):
                    shutil.copy(ref_text_path(path), ref_text_path(os.path.join(self.clips_dir, name)))
            entries[name] = {"file": name, "score": round(float(score), 2)}
        ranked = sorted(entries.values(), key=lambda c: -c.get("score", 0.0))
        for dropped in ranked[VOICE_BANK_CLIPS:]:
            path = os.path.join(self.clips_dir, dropped["file"])
            for p in (path, ref_text_path(path)):
                if os.path.exists(p):
                    os.remove(p)
        person["clips"] = ranked[:VOICE_BANK_CLIPS]


def _cosine(a, b) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else -1.0


def merge_matched_speakers(diar: List[Dict], prints: Dict, matches: Dict[str, str]):
    """Speakers matching the same person become one (the first label): a person split by the diarization speaks
    with one voice. Returns (diar, prints, matches) relabelled."""
    first: Dict[str, str] = {}
    rename = {}
    for spk in sorted(matches):
        pid = matches[spk]
        if pid in first:
            rename[spk] = first[pid]
            logger.info(f"Series memory: {spk} and {first[pid]} are the same person ({pid}), merged")
        else:
            first[pid] = spk
    if not rename:
        return diar, prints, matches
    diar = [dict(d, speaker=rename.get(d["speaker"], d["speaker"])) for d in diar]
    merged: Dict[str, List[List[float]]] = {}
    for spk, emb in (prints or {}).items():
        if emb:
            merged.setdefault(rename.get(spk, spk), []).append(emb)
    prints = {spk: [sum(v) / len(v) for v in zip(*embs)] for spk, embs in merged.items()}
    matches = {spk: pid for spk, pid in matches.items() if spk not in rename}
    return diar, prints, matches
