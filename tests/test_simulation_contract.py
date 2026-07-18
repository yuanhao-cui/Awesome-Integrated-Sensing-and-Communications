"""Machine-check the seven declared simulation reproducibility contracts."""

from __future__ import annotations

import ast
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "code" / "baselines"
EXPECTED_BASELINES = {
    "csi_ratio_doppler_estimation",
    "isac_capacity_distortion",
    "isac_energy_efficient_beamforming",
    "isac_resource_allocation",
    "ofdm_ambiguity_function",
    "ris_isac_beamforming",
    "xl_mimo_beam_training",
}
EVIDENCE_LEVELS = {
    "exact-reproduction",
    "equation-level",
    "research-reference",
    "educational-surrogate",
}
PIN_PATTERN = re.compile(r"^([A-Za-z0-9_.-]+)==([^=<>!~]+)$")


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant: {value}")


def _load_manifests() -> dict[str, dict[str, object]]:
    manifests: dict[str, dict[str, object]] = {}
    for path in sorted(BASELINES.glob("*/reproducibility.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict), f"{path} must contain one mapping"
        manifests[path.parent.name] = data
    return manifests


MANIFESTS = _load_manifests()


def _direct_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = PIN_PATTERN.fullmatch(line)
        assert match is not None, f"{path}: direct dependency is not exactly pinned: {line}"
        normalized = match.group(1).lower().replace("_", "-")
        assert normalized not in pins, f"{path}: duplicate dependency {normalized}"
        pins[normalized] = match.group(2)
    return pins


def test_exactly_seven_baseline_manifests_are_declared() -> None:
    assert set(MANIFESTS) == EXPECTED_BASELINES


def test_documented_evidence_vocabulary_matches_the_contract() -> None:
    for relative_path in (
        "README.md",
        "CONTRIBUTING.md",
        "code/README.md",
        ".github/PAPER_REVIEWER.md",
        ".github/ISSUE_TEMPLATE/new_baseline.yml",
        ".github/PULL_REQUEST_TEMPLATE.md",
    ):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert all(level in text for level in EVIDENCE_LEVELS), relative_path
        assert "algorithm-level" not in text, relative_path
        assert "figure-level" not in text, relative_path


@pytest.mark.parametrize("baseline", sorted(EXPECTED_BASELINES))
def test_manifest_schema_is_complete_and_typed(baseline: str) -> None:
    manifest = MANIFESTS[baseline]
    assert manifest.get("schema_version") == 1
    assert manifest.get("baseline") == baseline
    assert isinstance(manifest.get("reference"), dict) and manifest["reference"]
    assert manifest.get("evidence_level") in EVIDENCE_LEVELS
    assert manifest.get("paper_figure_parity") is False
    assert isinstance(manifest.get("model"), str) and manifest["model"].strip()

    for field in ("assumptions", "limitations"):
        values = manifest.get(field)
        assert isinstance(values, list) and values
        assert all(isinstance(value, str) and value.strip() for value in values)

    parameters = manifest.get("parameters")
    assert isinstance(parameters, dict) and parameters
    oracle = manifest.get("oracle")
    assert isinstance(oracle, dict) and oracle
    assert isinstance(oracle.get("type"), str) and oracle["type"].strip()
    assert isinstance(oracle.get("expected"), dict) and oracle["expected"]

    tolerances = manifest.get("tolerances")
    assert isinstance(tolerances, dict) and tolerances
    for name, value in tolerances.items():
        assert type(value) in (int, float), f"{baseline}: tolerance {name} is not numeric"
        assert value >= 0, f"{baseline}: tolerance {name} is negative"

    evidence = manifest.get("evidence")
    assert isinstance(evidence, dict) and evidence
    assert isinstance(evidence.get("command"), str) and evidence["command"].strip()
    certificate_checks = evidence.get("certificate_checks")
    assert isinstance(certificate_checks, list) and certificate_checks
    assert len(certificate_checks) == len(set(certificate_checks))
    assert all(
        isinstance(check, str) and re.fullmatch(r"[a-z][a-z0-9_]*", check)
        for check in certificate_checks
    )
    checks = evidence.get("checks")
    assert isinstance(checks, list) and checks
    assert all(isinstance(check, str) and check.strip() for check in checks)


@pytest.mark.parametrize("baseline", sorted(EXPECTED_BASELINES))
def test_manifest_command_emits_a_passing_matching_certificate(baseline: str) -> None:
    manifest = MANIFESTS[baseline]
    command = shlex.split(manifest["evidence"]["command"])
    assert command and command[0] == "python"
    command[0] = sys.executable
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "PYTHONWARNINGS": "error",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    if baseline in {"isac_capacity_distortion", "xl_mimo_beam_training"}:
        # Gate 1 executes this exact CLI in a clean process before pytest.  Calling
        # another compiled numerical child after the parent process has initialized
        # libomp can exhaust restricted macOS shared memory.  Exercise these same
        # verifiers in-process here so the full suite remains order-independent.
        if baseline == "isac_capacity_distortion":
            from code.baselines.isac_capacity_distortion.examples import (
                verify_surrogate,
            )

            certificate = verify_surrogate.build_certificate()
        else:
            from scripts.verify_simulation_baseline import verify_xl_mimo

            certificate = verify_xl_mimo()
        json.dumps(certificate, sort_keys=True, allow_nan=False)
    else:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=180,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        try:
            certificate = json.loads(
                result.stdout,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as error:
            pytest.fail(f"{baseline}: invalid JSON certificate: {error}\n{result.stdout}")
        json.dumps(certificate, sort_keys=True, allow_nan=False)
    assert certificate.get("schema_version") == 1
    assert certificate.get("baseline") == baseline
    assert certificate.get("evidence_level") == manifest["evidence_level"]
    assert certificate.get("paper_figure_parity") is manifest["paper_figure_parity"]
    assert certificate.get("status") == "pass"
    checks = certificate.get("checks")
    assert isinstance(checks, dict) and checks
    assert all(type(value) is bool and value for value in checks.values())
    assert set(checks) == set(manifest["evidence"]["certificate_checks"])


def test_root_direct_dependencies_are_exact_and_complete() -> None:
    pins = _direct_pins(ROOT / "requirements-ci.txt")
    assert pins == {
        "pytest": "8.3.5",
        "pytest-cov": "6.0.0",
        "numpy": "1.26.4",
        "scipy": "1.13.1",
        "matplotlib": "3.10.9",
        "pyyaml": "6.0.2",
        "cvxpy": "1.6.5",
        "scikit-learn": "1.5.2",
        "torch": "2.5.1",
        "jsonschema": "4.25.1",
        "yamllint": "1.35.1",
        "ruff": "0.9.10",
    }
    integrity_pins = _direct_pins(ROOT / "requirements-integrity.txt")
    assert integrity_pins == {
        "pytest": pins["pytest"],
        "pyyaml": pins["pyyaml"],
        "yamllint": pins["yamllint"],
        "jsonschema": pins["jsonschema"],
    }


@pytest.mark.parametrize("baseline", sorted(EXPECTED_BASELINES))
def test_local_requirements_use_the_root_constraint(baseline: str) -> None:
    root_pins = _direct_pins(ROOT / "requirements-ci.txt")
    path = BASELINES / baseline / "requirements.txt"
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert lines[0] == "-c ../../../requirements-ci.txt"
    dependencies = [line.lower().replace("_", "-") for line in lines[1:]]
    assert dependencies and len(dependencies) == len(set(dependencies))
    assert all(PIN_PATTERN.fullmatch(line) is None for line in lines[1:])
    assert set(dependencies) <= set(root_pins)


def test_xl_mimo_package_metadata_matches_root_pins() -> None:
    setup_path = BASELINES / "xl_mimo_beam_training" / "setup.py"
    tree = ast.parse(setup_path.read_text(encoding="utf-8"))
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
    )
    wanted = {"python_requires", "install_requires", "extras_require"}
    keywords = {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in call.keywords
        if keyword.arg in wanted
    }
    assert keywords["python_requires"] == ">=3.10,<3.13"
    root_pins = _direct_pins(ROOT / "requirements-ci.txt")
    for requirement in keywords["install_requires"] + keywords["extras_require"]["dev"]:
        match = PIN_PATTERN.fullmatch(requirement)
        assert match is not None
        name = match.group(1).lower().replace("_", "-")
        assert root_pins[name] == match.group(2)
