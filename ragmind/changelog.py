"""Read CHANGELOG.md so the app can show what's new in the running version."""

import os
import re

_RELEASE = re.compile(r"^##\s+\[([^\]]+)\]\s*(?:-\s*(.+))?$")
_SECTION = re.compile(r"^###\s+(.+)$")


def find_changelog(start: str) -> str | None:
    """CHANGELOG.md next to `start` (Docker image) or up to two folders above (checkout)."""
    folder = os.path.abspath(start)
    for _ in range(3):
        path = os.path.join(folder, "CHANGELOG.md")
        if os.path.isfile(path):
            return path
        folder = os.path.dirname(folder)
    return None


def _plain(text: str) -> str:
    return re.sub(r"\*\*|`", "", text).strip()


def parse(markdown: str) -> list[dict]:
    """Releases, newest first: [{"version", "date", "summary", "sections": [{"title", "items"}]}]."""
    releases, current, section = [], None, None
    for raw in markdown.splitlines():
        line = raw.rstrip()
        m = _RELEASE.match(line)
        if m:
            current = {"version": m.group(1).strip(), "date": (m.group(2) or "").strip(),
                       "summary": "", "sections": []}
            releases.append(current)
            section = None
            continue
        if current is None:
            continue
        m = _SECTION.match(line)
        if m:
            section = {"title": m.group(1).strip(), "items": []}
            current["sections"].append(section)
        elif line.startswith("- ") and section is not None:
            section["items"].append(_plain(line[2:]))
        elif line.startswith("  ") and section is not None and section["items"]:
            section["items"][-1] += " " + _plain(line)          # wrapped bullet
        elif line.strip() and section is None:
            current["summary"] = (current["summary"] + " " + _plain(line)).strip()
    return releases


def latest(path: str | None) -> dict | None:
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            releases = parse(f.read())
    except OSError:
        return None
    return releases[0] if releases else None
