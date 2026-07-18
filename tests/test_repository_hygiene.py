"""Repository-wide deterministic text-hygiene checks."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".cff",
    ".cfg",
    ".ini",
    ".md",
    ".py",
    ".rst",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_FILENAMES = {
    ".coveragerc",
    ".gitignore",
    "LICENSE",
}


def _repository_text_files() -> list[Path]:
    """Return tracked and pending nonignored text files."""
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    paths = [ROOT / item for item in result.stdout.decode().split("\0") if item]
    return sorted(
        path
        for path in paths
        if path.exists()
        and (path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_FILENAMES)
    )


def test_repository_text_has_no_trailing_whitespace() -> None:
    failures: list[str] = []
    for path in _repository_text_files():
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(keepends=True),
            start=1,
        ):
            content = line.rstrip("\r\n")
            if content.endswith((" ", "\t")):
                failures.append(f"{path.relative_to(ROOT)}:{line_number}")
    assert not failures, "trailing whitespace found at:\n" + "\n".join(failures)
