"""Clients for media metadata sources: Kodi-style .nfo files, TVmaze (no account) and TMDB (optional API key).

Every function returns plain dicts and None (or an empty result) on any failure, so a lookup never breaks a job.
"""

import html
import logging
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

TVMAZE_URL = "https://api.tvmaze.com"
TMDB_URL = "https://api.themoviedb.org/3"
USER_AGENT = "dubarr (https://github.com/neutrinus/dubarr)"

# TMDB encodes gender as 0 unknown, 1 female, 2 male, 3 non-binary
TMDB_GENDER = {1: "female", 2: "male", 3: "non-binary"}


def strip_html(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text))).strip()


def _gender(value) -> str:
    if isinstance(value, str) and value.lower() in ("male", "female", "non-binary"):
        return value.lower()
    if isinstance(value, int):
        return TMDB_GENDER.get(value, "")
    return ""


def _get_json(url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None, timeout: int = 10):
    try:
        resp = requests.get(url, params=params, headers={"User-Agent": USER_AGENT, **(headers or {})}, timeout=timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"Metadata request failed ({url}): {e}")
        return None


# --- .nfo (written next to the media by Sonarr/Radarr/Kodi/Jellyfin) ---


def read_nfo(path: str) -> Optional[Dict]:
    """Parses a Kodi-style .nfo (movie, tvshow or episodedetails). Returns None if it is not XML or unreadable."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
        # Some tools append a URL line after the XML; parse only the XML document
        end = raw.rfind(b">")
        root = ET.fromstring(raw[: end + 1])
    except Exception as e:
        logger.debug(f"Could not parse NFO {path}: {e}")
        return None

    def text(tag):
        el = root.find(tag)
        return el.text.strip() if el is not None and el.text else ""

    cast = []
    for actor in root.findall("actor"):
        name = (actor.findtext("name") or "").strip()
        role = (actor.findtext("role") or "").strip()
        if name or role:
            cast.append({"actor": name, "character": role, "gender": ""})

    return {
        "kind": root.tag,  # movie | tvshow | episodedetails
        "title": text("title"),
        "showtitle": text("showtitle"),
        "year": text("year") or text("premiered")[:4],
        "plot": strip_html(text("plot") or text("outline")),
        "genres": [g.text.strip() for g in root.findall("genre") if g.text],
        "season": text("season"),
        "episode": text("episode"),
        "cast": cast,
    }


# --- TVmaze (free, no account) ---


def _tvmaze_cast(entries: List[Dict]) -> List[Dict]:
    cast = []
    for c in entries or []:
        person = c.get("person") or {}
        character = c.get("character") or {}
        cast.append(
            {
                "actor": person.get("name", ""),
                "character": character.get("name", ""),
                "gender": _gender(person.get("gender")),
            }
        )
    return cast


def tvmaze_find_show(
    name: str = "", year: Optional[int] = None, tvdb_id: str = "", imdb_id: str = "", timeout: int = 10
) -> Optional[Dict]:
    """Finds a show by TVDB/IMDb id or by name (preferring a match on the premiere year). Returns the raw show."""
    if tvdb_id or imdb_id:
        params = {"thetvdb": tvdb_id} if tvdb_id else {"imdb": imdb_id}
        show = _get_json(f"{TVMAZE_URL}/lookup/shows", params, timeout=timeout)
        if show:
            return show
    if not name:
        return None
    results = _get_json(f"{TVMAZE_URL}/search/shows", {"q": name}, timeout=timeout) or []
    simple = re.sub(r"\s+-\s+", " ", name)  # Sonarr writes "Star Trek: Voyager" as "Star Trek - Voyager"
    if not results and simple != name:
        results = _get_json(f"{TVMAZE_URL}/search/shows", {"q": simple}, timeout=timeout) or []
    shows = [r["show"] for r in results if isinstance(r, dict) and r.get("show")]
    if not shows:
        return None
    if year:
        for show in shows:
            if (show.get("premiered") or "").startswith(str(year)):
                return show
    return shows[0]


def tvmaze_episode_info(
    name: str = "",
    season: Optional[int] = None,
    episode: Optional[int] = None,
    year: Optional[int] = None,
    tvdb_id: str = "",
    imdb_id: str = "",
    timeout: int = 10,
) -> Optional[Dict]:
    show = tvmaze_find_show(name, year, tvdb_id, imdb_id, timeout)
    if not show:
        return None
    show_id = show["id"]
    info = {
        "source": "tvmaze",
        "kind": "episode",
        "title": show.get("name", ""),
        "year": (show.get("premiered") or "")[:4],
        "genres": show.get("genres") or [],
        "language": show.get("language") or "",
        "plot": strip_html(show.get("summary")),
        "cast": _tvmaze_cast(_get_json(f"{TVMAZE_URL}/shows/{show_id}/cast", timeout=timeout)),
    }
    if season is not None and episode is not None:
        ep = _get_json(f"{TVMAZE_URL}/shows/{show_id}/episodebynumber", {"season": season, "number": episode}, timeout=timeout)
        if ep:
            info.update(
                {
                    "season": season,
                    "episode": episode,
                    "episode_title": ep.get("name", ""),
                    "episode_plot": strip_html(ep.get("summary")),
                    "guest_cast": _tvmaze_cast(_get_json(f"{TVMAZE_URL}/episodes/{ep['id']}/guestcast", timeout=timeout)),
                }
            )
    return info


# --- TMDB (optional, needs a free API key) ---


def _tmdb_auth(api_key: str):
    # v4 read access tokens are JWTs and go in the header; v3 keys go in the query string
    if api_key.startswith("ey"):
        return {}, {"Authorization": f"Bearer {api_key}"}
    return {"api_key": api_key}, {}


def tmdb_movie_info(
    api_key: str, name: str = "", year: Optional[int] = None, tmdb_id: str = "", imdb_id: str = "", timeout: int = 10
) -> Optional[Dict]:
    if not api_key:
        return None
    params, headers = _tmdb_auth(api_key)
    movie_id = tmdb_id
    if not movie_id and imdb_id:
        found = _get_json(f"{TMDB_URL}/find/{imdb_id}", {**params, "external_source": "imdb_id"}, headers, timeout) or {}
        if found.get("movie_results"):
            movie_id = found["movie_results"][0]["id"]
    if not movie_id and name:
        query = {**params, "query": name}
        if year:
            query["year"] = year
        results = (_get_json(f"{TMDB_URL}/search/movie", query, headers, timeout) or {}).get("results") or []
        if results:
            movie_id = results[0]["id"]
    if not movie_id:
        return None

    movie = _get_json(f"{TMDB_URL}/movie/{movie_id}", {**params, "append_to_response": "credits"}, headers, timeout)
    if not movie:
        return None
    cast = [
        {"actor": c.get("name", ""), "character": c.get("character", ""), "gender": _gender(c.get("gender"))}
        for c in (movie.get("credits") or {}).get("cast", [])
    ]
    return {
        "source": "tmdb",
        "kind": "movie",
        "title": movie.get("title", ""),
        "original_title": movie.get("original_title", ""),
        "year": (movie.get("release_date") or "")[:4],
        "genres": [g.get("name", "") for g in movie.get("genres", [])],
        "language": movie.get("original_language", ""),
        "plot": movie.get("overview", ""),
        "tagline": movie.get("tagline", ""),
        "cast": cast,
    }
