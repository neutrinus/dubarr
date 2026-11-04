"""Identifies what a video is (show/episode or movie) and gathers context for the translator.

Sources, in order: Kodi-style .nfo files next to the media (written by Sonarr/Radarr), then TVmaze for episodes
(free, no account) or TMDB for movies (only with TMDB_API_KEY). The .nfo wins for plot and character names,
the online source fills the gaps (e.g. actor gender, which .nfo files do not carry).
"""

import logging
import os
import re
from typing import Dict, List, Optional

from infrastructure import metadata_client

logger = logging.getLogger(__name__)

EPISODE_RE = re.compile(r"(?<![a-z0-9])s(\d{1,2})[ ._-]?e(\d{1,3})(?!\d)|(?<!\d)(\d{1,2})x(\d{2,3})(?!\d)", re.IGNORECASE)
YEAR_RE = re.compile(r"[\s.(\[_-]((?:19|20)\d{2})(?=[\s.)\]_-]|$)")
ID_RE = re.compile(r"[\[{]?(tvdb|tmdb|imdb)(?:id)?[-=]((?:tt)?\d+)[\]}]?", re.IGNORECASE)
SEASON_DIR_RE = re.compile(r"^(season\s*\d+|specials|s\d{1,2})$", re.IGNORECASE)
# Release tokens that end a movie title when there is no year
JUNK_RE = re.compile(
    r"[\s.\[(_-](?:2160p|1080p|720p|480p|4k|uhd|web[- .]?dl|webrip|bluray|blu-ray|brrip|bdrip|dvdrip|hdtv|remux|"
    r"x264|x265|h\.?264|h\.?265|hevc|proper|repack|extended|unrated|imax)\b.*$",
    re.IGNORECASE,
)

MAX_CAST = 20
MAX_GUESTS = 15
MAX_PLOT = 1200


def _clean_title(text: str) -> str:
    text = ID_RE.sub(" ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    if " " not in text.strip():
        text = text.replace(".", " ").replace("_", " ")
    text = re.sub(r"\(\s*\)", " ", text)
    return re.sub(r"\s+", " ", text).strip(" -.")


def _split_year(text: str):
    m = YEAR_RE.search(" " + text)
    if not m:
        return _clean_title(text), None
    return _clean_title((" " + text)[: m.start()]), int(m.group(1))


def parse_media_path(path: str) -> Dict:
    """Guesses title, year, season/episode and database ids from a Sonarr/Radarr/scene style path."""
    stem = os.path.splitext(os.path.basename(path))[0]
    parent = os.path.basename(os.path.dirname(path))
    grandparent = os.path.basename(os.path.dirname(os.path.dirname(path)))
    show_dir = grandparent if SEASON_DIR_RE.match(parent) else parent

    ids = {}
    for kind, value in ID_RE.findall(f"{show_dir} {parent} {stem}"):
        ids.setdefault(f"{kind.lower()}_id", value)

    info: Dict = {"kind": "movie", "title": "", "year": None, "season": None, "episode": None, **ids}
    m = EPISODE_RE.search(stem)
    if m:
        info["kind"] = "episode"
        info["season"] = int(m.group(1) or m.group(3))
        info["episode"] = int(m.group(2) or m.group(4))
        title, year = _split_year(stem[: m.start()])
        dir_title, dir_year = _split_year(show_dir) if show_dir else ("", None)
        info["title"] = title or dir_title
        info["year"] = year or dir_year
        episode_title = _clean_title(JUNK_RE.sub("", " " + stem[m.end() :]))
        if episode_title:
            info["episode_title"] = episode_title
        return info

    title, year = _split_year(stem)
    if not year:
        title = _clean_title(JUNK_RE.sub("", stem))
    if not title and parent:
        title, year = _split_year(parent)
    info["title"], info["year"] = title, year
    return info


def _find_nfos(path: str, kind: str) -> Dict[str, Optional[Dict]]:
    folder = os.path.dirname(path)
    stem = os.path.splitext(path)[0]
    found = {"item": None, "show": None}
    for candidate in (stem + ".nfo", os.path.join(folder, "movie.nfo")):
        if os.path.isfile(candidate):
            found["item"] = metadata_client.read_nfo(candidate)
            if found["item"]:
                break
    if kind == "episode":
        for candidate in (os.path.join(folder, "tvshow.nfo"), os.path.join(os.path.dirname(folder), "tvshow.nfo")):
            if os.path.isfile(candidate):
                found["show"] = metadata_client.read_nfo(candidate)
                if found["show"]:
                    break
    return found


def _merge_cast(primary: List[Dict], secondary: List[Dict]) -> List[Dict]:
    """Keeps the primary list (from .nfo) and fills in gender from the secondary one by actor or character."""
    if not primary:
        return secondary or []
    by_actor = {c["actor"].lower(): c for c in secondary or [] if c.get("actor")}
    by_char = {c["character"].lower(): c for c in secondary or [] if c.get("character")}
    merged = []
    for c in primary:
        other = by_actor.get(c.get("actor", "").lower()) or by_char.get(c.get("character", "").lower()) or {}
        merged.append({**c, "gender": c.get("gender") or other.get("gender", "")})
    return merged


# Separators between the names in an episode title ("Backlash vs. Disposable Hero, Tentoumushi vs. Sallad")
NAME_SPLIT_RE = re.compile(r"\s+vs\.?\s+|\s*,\s*|\s+&\s+|\s+and\s+|\s+-\s+", re.IGNORECASE)
# A part in parentheses is a remark, not a name ("(Lightweight Finals)")
PAREN_RE = re.compile(r"\([^)]*\)")


def episode_names(info: Dict) -> List[str]:
    """Names in the episode title and the cast: they occur in the episode, with the right spelling (Spec 006, 3)."""
    names: List[str] = []
    title = PAREN_RE.sub(" ", (info or {}).get("episode_title") or "")
    parts = NAME_SPLIT_RE.split(title) if NAME_SPLIT_RE.search(title) else []
    for part in parts:
        part = part.strip()
        if part and len(part.split()) <= 4:
            names.append(part)
    for c in ((info or {}).get("cast") or []) + ((info or {}).get("guest_cast") or []):
        for key in ("character", "actor"):
            if c.get(key):
                names.append(c[key])
    seen = set()
    return [n for n in names if not (n.lower() in seen or seen.add(n.lower()))]


def cast_gender(info: Dict, name: str) -> Optional[str]:
    """The cast list's gender of the character (or actor) a speaker is named after; None when unknown or ambiguous."""
    key = re.sub(r"[^a-z ]", "", (name or "").lower()).split()
    if not key:
        return None
    found = set()
    for c in ((info or {}).get("cast") or []) + ((info or {}).get("guest_cast") or []):
        if c.get("gender") not in ("male", "female"):
            continue
        for field in ("character", "actor"):
            words = re.sub(r"[^a-z ]", "", (c.get(field) or "").lower()).split()
            # The whole name, or its last word (the surname: "Kirk" for "Captain James Tiberius Kirk")
            if words and (words == key or words[-1] == key[-1]):
                found.add(c["gender"])
    return found.pop() if len(found) == 1 else None


def lookup_media_info(path: str, online: bool = True, tmdb_api_key: str = "", timeout: int = 10, local: bool = True) -> Dict:
    """Returns what is known about the video (empty dict if nothing was found). Never raises."""
    try:
        guess = parse_media_path(path)
        nfo = _find_nfos(path, guess["kind"]) if local else {"item": None, "show": None}
        item, show = nfo["item"] or {}, nfo["show"] or {}
        if item.get("kind") == "episodedetails":
            guess["kind"] = "episode"
            guess["season"] = int(item["season"]) if item.get("season", "").isdigit() else guess["season"]
            guess["episode"] = int(item["episode"]) if item.get("episode", "").isdigit() else guess["episode"]
        title = show.get("title") or item.get("showtitle") or (item.get("title") if guess["kind"] == "movie" else "")
        title = title or guess["title"]
        year = int(show["year"]) if show.get("year", "").isdigit() else guess["year"]

        remote = None
        if online and title:
            if guess["kind"] == "episode":
                remote = metadata_client.tvmaze_episode_info(
                    title,
                    guess["season"],
                    guess["episode"],
                    year,
                    guess.get("tvdb_id", ""),
                    guess.get("imdb_id", ""),
                    timeout,
                )
            else:
                remote = metadata_client.tmdb_movie_info(
                    tmdb_api_key, title, year, guess.get("tmdb_id", ""), guess.get("imdb_id", ""), timeout
                )
        remote = remote or {}

        if guess["kind"] == "episode":
            info = {
                "kind": "episode",
                "title": show.get("title") or remote.get("title") or title,
                "year": show.get("year") or remote.get("year") or (str(year) if year else ""),
                "genres": show.get("genres") or remote.get("genres") or [],
                "plot": show.get("plot") or remote.get("plot", ""),
                "season": guess["season"],
                "episode": guess["episode"],
                "episode_title": item.get("title") or remote.get("episode_title", "") or guess.get("episode_title", ""),
                "episode_plot": item.get("plot") or remote.get("episode_plot", ""),
                "cast": _merge_cast(show.get("cast"), remote.get("cast")),
                # Episode .nfo actors are usually the guest stars plus the main cast; keep only those not in the show cast
                "guest_cast": _merge_cast(item.get("cast"), remote.get("guest_cast")),
            }
            main_actors = {c.get("actor", "").lower() for c in info["cast"]}
            info["guest_cast"] = [c for c in info["guest_cast"] if c.get("actor", "").lower() not in main_actors]
        else:
            info = {
                "kind": "movie",
                "title": item.get("title") or remote.get("title") or title,
                "year": item.get("year") or remote.get("year") or (str(year) if year else ""),
                "genres": item.get("genres") or remote.get("genres") or [],
                "plot": item.get("plot") or remote.get("plot", ""),
                "cast": _merge_cast(item.get("cast"), remote.get("cast")),
            }

        sources = [s for s, ok in (("nfo", item or show), (remote.get("source", ""), remote)) if ok]
        if not sources:
            logger.info(f"Media metadata: nothing found for '{title}' (guessed from the file name)")
            # The file name alone still tells the translator what this is
            if not title:
                return {}
            return {
                "kind": info["kind"],
                "title": title,
                "year": info["year"],
                "season": guess["season"],
                "episode": guess["episode"],
                "episode_title": guess.get("episode_title", ""),
                "sources": [],
            }
        info["sources"] = sources
        logger.info(f"Media metadata: '{info['title']}' from {', '.join(sources)} ({len(info['cast'])} cast members)")
        return info
    except Exception as e:
        logger.warning(f"Media metadata lookup failed: {e}")
        return {}


def _cast_lines(cast: List[Dict], limit: int) -> List[str]:
    lines = []
    for c in cast[:limit]:
        if not c.get("character") and not c.get("actor"):
            continue
        line = f"- {c.get('character') or '?'}"
        if c.get("actor"):
            line += f" (played by {c['actor']})"
        if c.get("gender"):
            line += f", {c['gender']}"
        lines.append(line)
    return lines


def _short(text: str, limit: int = MAX_PLOT) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."


def format_media_context(info: Dict) -> str:
    """Renders the metadata as a plain-text block for LLM prompts. Empty string if nothing is known."""
    if not info or not info.get("title"):
        return ""
    head = info["title"] + (f" ({info['year']})" if info.get("year") else "")
    kind = "TV series" if info.get("kind") == "episode" else "Movie"
    lines = [f"{kind}: {head}"]
    if info.get("genres"):
        lines.append(f"Genres: {', '.join(info['genres'])}")
    if info.get("season") is not None and info.get("episode") is not None:
        ep = f"Episode: S{info['season']:02d}E{info['episode']:02d}"
        if info.get("episode_title"):
            ep += f' "{info["episode_title"]}"'
        lines.append(ep)
    if info.get("plot"):
        lines.append(f"Synopsis: {_short(info['plot'])}")
    if info.get("episode_plot"):
        lines.append(f"Episode synopsis: {_short(info['episode_plot'])}")
    cast = _cast_lines(info.get("cast") or [], MAX_CAST)
    if cast:
        lines.append("Main characters (character, actor, gender):")
        lines.extend(cast)
    guests = _cast_lines(info.get("guest_cast") or [], MAX_GUESTS)
    if guests:
        lines.append("Characters in this episode:")
        lines.extend(guests)
    return "\n".join(lines)


def format_short_context(info: Dict) -> str:
    """One or two lines (title, episode, genres) for prompts that run per line."""
    if not info or not info.get("title"):
        return ""
    text = info["title"] + (f" ({info['year']})" if info.get("year") else "")
    if info.get("season") is not None and info.get("episode") is not None:
        text += f", S{info['season']:02d}E{info['episode']:02d}"
        if info.get("episode_title"):
            text += f' "{info["episode_title"]}"'
    if info.get("genres"):
        text += f" [{', '.join(info['genres'])}]"
    return text
