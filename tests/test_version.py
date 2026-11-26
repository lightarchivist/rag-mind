"""Version number and release notes stay consistent."""

import re
from pathlib import Path

import pytest

from ragmind import __version__, changelog

ROOT = Path(__file__).resolve().parents[1]


def test_changelog_readme_and_code_agree():
    latest = changelog.latest(str(ROOT / "CHANGELOG.md"))
    assert latest["version"] == __version__, "add a CHANGELOG.md entry for the new __version__"
    assert re.match(r"\d{4}-\d{2}-\d{2}$", latest["date"])
    assert latest["sections"] and all(sec["items"] for sec in latest["sections"])
    assert f"version-{__version__}-" in (ROOT / "README.md").read_text(), "update the README badge"


def test_parse_handles_summary_wrapped_bullets_and_markdown():
    md = """# Changelog
intro text
## [1.1.0] - 2026-01-02
Short summary line.
### Fixed
- First fix with `code` and **bold**
  that wraps onto a second line.
- Second fix.
## [1.0.0] - 2026-01-01
### Added
- Old thing.
"""
    rel = changelog.parse(md)
    assert [r["version"] for r in rel] == ["1.1.0", "1.0.0"]
    assert rel[0]["summary"] == "Short summary line."
    assert rel[0]["sections"][0] == {"title": "Fixed", "items": [
        "First fix with code and bold that wraps onto a second line.", "Second fix."]}


def test_missing_changelog_is_harmless(tmp_path):
    assert changelog.latest(None) is None
    assert changelog.latest(str(tmp_path / "nope.md")) is None


def test_web_reports_version_and_release_notes(web):
    d = web.version()
    assert d["version"] == __version__
    assert d["release"]["version"] == __version__
    assert web.health()["version"] == __version__


def test_cli_version(cli, capsys):
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert capsys.readouterr().out.strip() == f"RagMind {__version__}"
