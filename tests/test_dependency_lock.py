"""Contracts for the complete cross-platform dependency lock."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from jsonschema import Draft202012Validator

from scripts.check_dependency_lock import validate_dependency_lock


ROOT = Path(__file__).resolve().parents[1]
SETUP_UV = "astral-sh/setup-uv@11f9893b081a58869d3b5fccaea48c9e9e46f990"


def test_dependency_lock_is_complete_and_hashed() -> None:
    report = validate_dependency_lock()
    assert report["direct_ci_packages"] == 12
    assert report["direct_integrity_packages"] == 4
    assert report["direct_cff_packages"] == 1
    assert report["locked_registry_packages"] >= 60
    assert report["locked_artifacts"] == report["sha256_hashes"]
    assert report["sha256_hashes"] >= report["locked_registry_packages"]
    assert report["source_only_packages"] == 1


def test_authority_metadata_validator_is_draft_2020_12() -> None:
    assert Draft202012Validator.META_SCHEMA["$schema"].endswith("/draft/2020-12/schema")


def test_workflows_install_and_execute_the_checked_lock() -> None:
    gate1 = (ROOT / ".github/workflows/test.yml").read_text(encoding="utf-8")
    gate2 = (ROOT / ".github/workflows/link-check.yml").read_text(encoding="utf-8")
    for workflow in (gate1, gate2):
        assert f"uses: {SETUP_UV}" in workflow
        assert 'version: "0.11.28"' in workflow
        assert "uv lock --check --refresh" in workflow
        assert ".venv/bin/python scripts/check_dependency_lock.py" in workflow
        assert 'echo "$PWD/.venv/bin" >> "$GITHUB_PATH"' in workflow
        assert "pip install" not in workflow
    assert "uv sync --locked --only-group ci --python python" in gate1
    assert "UV_PROJECT_ENVIRONMENT=.venv-cff uv sync --locked" in gate1
    assert "--only-group cff-validation --python python" in gate1
    assert ".venv-cff/bin/cffconvert --validate" in gate1
    assert "uv sync --locked --only-group integrity --python python" in gate2
    assert "dependency-lock-installability:" in gate1
    assert "runs-on: macos-14" in gate1
    assert 'UV_NO_CACHE: "1"' in gate1
    assert 'platform.machine() == "arm64"' in gate1
    assert gate1.count("uv sync --locked --only-group ci --python python") == 2


def test_lock_checker_fails_closed_under_python_optimization() -> None:
    result = subprocess.run(
        [sys.executable, "-O", "scripts/check_dependency_lock.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "cannot run with Python optimization" in result.stderr


def test_dependabot_retains_reviewed_python_lock_updates() -> None:
    dependabot = (ROOT / ".github/dependabot.yml").read_text(encoding="utf-8")
    assert "package-ecosystem: pip" in dependabot
    assert "directory: /" in dependabot
