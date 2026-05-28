"""
Log ingestion and parsing for Sherlock.

Parses raw log text into structured entries. Handles common formats:
  - "2026-01-01T12:00:00 ERROR service: message"
  - "[ERROR] message"
  - plain lines (defaulted to INFO)
"""

import re
from sherlock.models import LogEntry


LEVELS = ("CRITICAL", "ERROR", "WARN", "WARNING", "INFO", "DEBUG")

_TS = r"(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)"
_LEVEL = r"(?P<level>CRITICAL|ERROR|WARN|WARNING|INFO|DEBUG)"
_SVC = r"(?P<svc>[\w\-.]+)"

# Pattern: timestamp LEVEL service: message
_PATTERN_FULL = re.compile(rf"^{_TS}\s+{_LEVEL}\s+{_SVC}[:\]]?\s*(?P<msg>.*)$", re.IGNORECASE)
# Pattern: [LEVEL] message  or  LEVEL: message
_PATTERN_LEVEL = re.compile(rf"^\[?{_LEVEL}\]?[:\s]\s*(?P<msg>.*)$", re.IGNORECASE)


def parse_logs(raw: str) -> list[LogEntry]:
    """Parse raw multi-line log text into structured LogEntry objects."""
    entries: list[LogEntry] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue

        m = _PATTERN_FULL.match(line)
        if m:
            entries.append(LogEntry(
                timestamp=m.group("ts"),
                level=_normalize_level(m.group("level")),
                service=m.group("svc"),
                message=m.group("msg").strip(),
            ))
            continue

        m = _PATTERN_LEVEL.match(line)
        if m:
            entries.append(LogEntry(
                level=_normalize_level(m.group("level")),
                message=m.group("msg").strip(),
            ))
            continue

        entries.append(LogEntry(level="INFO", message=line))
    return entries


def _normalize_level(level: str) -> str:
    level = level.upper()
    return "WARN" if level == "WARNING" else level


def log_stats(entries: list[LogEntry]) -> dict:
    """Compute summary statistics over parsed log entries."""
    counts: dict[str, int] = {}
    services: dict[str, int] = {}
    for e in entries:
        counts[e.level] = counts.get(e.level, 0) + 1
        if e.service:
            services[e.service] = services.get(e.service, 0) + 1
    return {
        "total": len(entries),
        "by_level": counts,
        "by_service": services,
        "error_count": counts.get("ERROR", 0) + counts.get("CRITICAL", 0),
    }
