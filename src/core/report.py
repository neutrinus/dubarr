"""Per-line production report (DEBUG): what was said, what the dub says, and how the take was made."""

import csv
import logging
from typing import Dict, List

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
    "similarity",
    "attempts",
    "source",
    "dub",
    "heard",
]


def _ts(seconds: float) -> str:
    minutes, rest = divmod(max(seconds, 0.0), 60)
    return f"{int(minutes):02d}:{rest:05.2f}"


def report_rows(script: List[Dict], results: List[Dict]) -> List[Dict]:
    """One row per dubbed line, in script order."""
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
                "similarity": "" if res.get("similarity") is None else f"{res['similarity']:.2f}",
                "attempts": res.get("attempts", ""),
                "source": line.get("text_en", line.get("text", "")),
                "dub": res.get("final_text", ""),
                "heard": res.get("heard") or "",
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
    parts = [f"{len(rows)} lines", f"fill {spoken / slots:.0%}" if slots else "fill n/a"]
    parts += [f"{k} {v}" for k, v in sorted(statuses.items())]
    if sims:
        parts.append(f"similarity min {min(sims):.2f} mean {sum(sims) / len(sims):.2f}")
    return ", ".join(parts)


def write_report(path: str, script: List[Dict], results: List[Dict]) -> str:
    """Writes the report as a tab-separated file (opens in any spreadsheet) and returns its summary."""
    rows = report_rows(script, results)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return summary(rows)
