"""Per-line production report (DEBUG): what was said, what the dub says, and how the take was made."""

import csv
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

COLUMNS = [
    "index",
    "start",
    "end",
    "speaker",
    "slot",
    "limit",
    "duration",
    "fill",
    "status",
    "voice",
    "score",
    "voice_match",
    "text_match",
    "similarity",
    "attempts",
    "source",
    "dub",
    "heard",
    "ctc_heard",
    "asr",
    "speaker_moved",
    "speaker_doubt",
    "energy",
    "ref_energy",
]


def _ts(seconds: float) -> str:
    minutes, rest = divmod(max(seconds, 0.0), 60)
    return f"{int(minutes):02d}:{rest:05.2f}"


def _num(value) -> str:
    return "" if value is None else f"{value:.2f}"


def report_rows(script: List[Dict], results: List[Dict], ref_energy: Optional[Dict[str, str]] = None) -> List[Dict]:
    """One row per dubbed line, in script order. ``ref_energy``: reference clip -> energy class (Spec 001, 2.4)."""
    ref_energy = ref_energy or {}
    rows = []
    for res in sorted(results, key=lambda r: r["index"]):
        line = script[res["index"]]
        slot = line["end"] - line["start"]
        limit = line.get("max_end", line["end"]) - min(line.get("min_start", line["start"]), line["start"])
        rows.append(
            {
                "index": res["index"],
                "start": _ts(line["start"]),
                "end": _ts(line["end"]),
                "speaker": line.get("speaker", ""),
                "slot": f"{slot:.2f}",
                "limit": f"{limit:.2f}",
                "duration": f"{res.get('duration', 0.0):.2f}",
                "fill": f"{res.get('duration', 0.0) / slot:.0%}" if slot > 0 else "",
                "status": res.get("status", ""),
                "voice": res.get("voice_type") or "",
                "score": _num(res.get("score")),
                "voice_match": _num(res.get("voice_match")),
                "text_match": _num(res.get("text_match")),
                "similarity": _num(res.get("similarity")),
                "attempts": res.get("attempts", ""),
                "source": line.get("text_en", line.get("text", "")),
                "dub": res.get("final_text", ""),
                "heard": res.get("heard") or "",
                "ctc_heard": res.get("ctc_heard") or "",
                "asr": line.get("asr") or "",
                "speaker_moved": line.get("speaker_moved") or "",
                "speaker_doubt": line.get("speaker_doubt") or "",
                "energy": line.get("energy") or "",
                "ref_energy": ref_energy.get(res.get("ref") or "", "DESIGNED" if res.get("voice_type") == "DESIGNED" else ""),
            }
        )
    return rows


def summary(rows: List[Dict]) -> str:
    """One log line: how many lines, how full the slots are, and how many were not accepted."""
    if not rows:
        return "no dubbed lines"
    slots = sum(float(r["slot"]) for r in rows)
    spoken = sum(float(r["duration"]) for r in rows)
    statuses: Dict[str, int] = {}
    for r in rows:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    sims = [float(r["similarity"]) for r in rows if r["similarity"]]
    scores = [float(r["score"]) for r in rows if r["score"]]
    parts = [f"{len(rows)} lines", f"fill {spoken / slots:.0%}" if slots else "fill n/a"]
    parts += [f"{k} {v}" for k, v in sorted(statuses.items())]
    if scores:
        parts.append(f"judge min {min(scores):.2f} mean {sum(scores) / len(scores):.2f}")
    if sims:
        parts.append(f"similarity min {min(sims):.2f} mean {sum(sims) / len(sims):.2f}")
    classes = ("calm", "lively", "shout")
    matched = [r for r in rows if r.get("energy") in classes and r.get("ref_energy") in classes]
    if matched:
        same = sum(r["energy"] == r["ref_energy"] for r in matched)
        parts.append(f"energy of the clip = the line's in {same}/{len(matched)}")
    asr = [r.get("asr") for r in rows if r.get("asr") in ("fixed", "guessed", "unsure")]
    if asr:
        parts.append("ASR " + ", ".join(f"{k} {asr.count(k)}" for k in ("fixed", "guessed", "unsure") if asr.count(k)))
    return ", ".join(parts)


def write_report(path: str, script: List[Dict], results: List[Dict], ref_energy: Optional[Dict[str, str]] = None) -> str:
    """Writes the report as a tab-separated file (opens in any spreadsheet) and returns its summary."""
    rows = report_rows(script, results, ref_energy)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return summary(rows)
