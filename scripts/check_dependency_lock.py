"""Validate the complete, cross-platform dependency lock without network access."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 job
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
PIN = re.compile(r"([A-Za-z0-9][A-Za-z0-9_.-]*)==([^;\s]+)")
SUPPORTED_PYTHON = ">=3.10, <3.13"
NVIDIA_LINUX_PACKAGES = {
    "nvidia-cublas-cu12",
    "nvidia-cuda-cupti-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cufft-cu12",
    "nvidia-curand-cu12",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu12",
    "nvidia-nccl-cu12",
    "nvidia-nvjitlink-cu12",
    "nvidia-nvtx-cu12",
}
LINUX_X86_64_MARKER = "platform_machine == 'x86_64' and sys_platform == 'linux'"


def _canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _requirement_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = PIN.fullmatch(line)
        if match is None:
            raise AssertionError(f"{path.name}: non-exact requirement: {raw_line!r}")
        name = _canonical_name(match.group(1))
        if name in pins:
            raise AssertionError(f"{path.name}: duplicate requirement: {name}")
        pins[name] = match.group(2)
    return pins


def _group_pins(project: dict[str, Any], group: str) -> dict[str, str]:
    groups = project.get("dependency-groups", {})
    active: set[str] = set()

    def expand(name: str) -> dict[str, str]:
        if name in active:
            raise AssertionError(f"cyclic dependency-group include: {name}")
        if name not in groups:
            raise AssertionError(f"missing dependency group: {name}")
        active.add(name)
        pins: dict[str, str] = {}
        for item in groups[name]:
            if isinstance(item, dict):
                included = item.get("include-group")
                if set(item) != {"include-group"} or not isinstance(included, str):
                    raise AssertionError(f"{name}: unsupported group entry: {item!r}")
                included_pins = expand(included)
                overlap = pins.keys() & included_pins.keys()
                if overlap:
                    raise AssertionError(f"{name}: duplicate included pins: {sorted(overlap)}")
                pins.update(included_pins)
                continue
            if not isinstance(item, str):
                raise AssertionError(f"{name}: unsupported requirement: {item!r}")
            match = PIN.fullmatch(item)
            if match is None:
                raise AssertionError(f"{name}: non-exact direct requirement: {item!r}")
            package_name = _canonical_name(match.group(1))
            if package_name in pins:
                raise AssertionError(f"{name}: duplicate direct requirement: {package_name}")
            pins[package_name] = match.group(2)
        active.remove(name)
        return pins

    return expand(group)


def validate_dependency_lock(root: Path = ROOT) -> dict[str, int]:
    """Return lock statistics after asserting the repository's lock contract."""

    if not __debug__:
        raise RuntimeError(
            "dependency-lock validation cannot run with Python optimization"
        )

    project = _load_toml(root / "pyproject.toml")
    lock = _load_toml(root / "uv.lock")

    assert project["project"]["requires-python"] == ">=3.10,<3.13"
    assert project["project"]["dependencies"] == []
    assert project["tool"]["uv"] == {
        "package": False,
        "default-groups": [],
        "fork-strategy": "fewest",
        "build-constraint-dependencies": ["setuptools==83.0.0"],
        "required-environments": [
            "platform_machine == 'x86_64' and sys_platform == 'linux'",
            "platform_machine == 'arm64' and sys_platform == 'darwin'",
        ],
        "conflicts": [
            [{"group": "cff-validation"}, {"group": "integrity"}],
            [{"group": "cff-validation"}, {"group": "ci"}],
        ],
    }
    assert lock["version"] == 1
    assert lock["requires-python"] == SUPPORTED_PYTHON
    assert lock["options"]["fork-strategy"] == "fewest"
    assert lock["manifest"]["build-constraints"] == [
        {"name": "setuptools", "specifier": "==83.0.0"}
    ]

    ci_pins = _group_pins(project, "ci")
    integrity_pins = _group_pins(project, "integrity")
    cff_pins = _group_pins(project, "cff-validation")
    assert ci_pins == _requirement_pins(root / "requirements-ci.txt")
    assert integrity_pins == _requirement_pins(root / "requirements-integrity.txt")
    assert cff_pins == _requirement_pins(root / "requirements-cff.txt")
    assert set(integrity_pins.items()) <= set(ci_pins.items())
    assert cff_pins == {"cffconvert": "2.0.0"}

    packages = lock.get("package")
    assert isinstance(packages, list) and packages
    registry_packages = [package for package in packages if "registry" in package.get("source", {})]
    virtual_packages = [package for package in packages if "virtual" in package.get("source", {})]
    assert len(virtual_packages) == 1
    assert virtual_packages[0]["name"] == project["project"]["name"]

    locked_versions: dict[str, set[str]] = {}
    artifact_count = 0
    hash_count = 0
    source_only_packages: set[str] = set()
    for package in registry_packages:
        name = _canonical_name(package["name"])
        version = package.get("version")
        assert isinstance(version, str) and version
        locked_versions.setdefault(name, set()).add(version)
        artifacts: list[dict[str, Any]] = []
        if "sdist" in package:
            artifacts.append(package["sdist"])
        artifacts.extend(package.get("wheels", []))
        assert artifacts, f"{name}: registry package has no locked distribution"
        if not package.get("wheels"):
            source_only_packages.add(name)
        for artifact in artifacts:
            assert set(artifact) >= {"url", "hash"}, f"{name}: incomplete artifact"
            assert artifact["url"].startswith("https://files.pythonhosted.org/")
            assert SHA256.fullmatch(artifact["hash"]), f"{name}: invalid artifact hash"
            artifact_count += 1
            hash_count += 1

    for name, version in ci_pins.items():
        versions = locked_versions.get(name, set())
        assert version in versions, f"{name}: direct pin is not locked"
        if name != "jsonschema":
            assert versions == {version}, f"{name}: direct pin is not locked exactly"
    assert locked_versions.get("jsonschema") == {"3.2.0", "4.25.1"}
    for name, version in cff_pins.items():
        assert locked_versions.get(name) == {version}, f"{name}: CFF pin is not locked exactly"

    all_names = {_canonical_name(package["name"]) for package in packages}
    for package in packages:
        for dependency in package.get("dependencies", []):
            dependency_name = _canonical_name(dependency["name"])
            assert dependency_name in all_names, (
                f"{package['name']}: unresolved locked dependency {dependency_name}"
            )

    torch = next(package for package in packages if package["name"] == "torch")
    torch_dependencies = {
        _canonical_name(dependency["name"]): dependency.get("marker", "")
        for dependency in torch["dependencies"]
    }
    assert NVIDIA_LINUX_PACKAGES <= set(torch_dependencies)
    for name in NVIDIA_LINUX_PACKAGES:
        assert torch_dependencies[name] == LINUX_X86_64_MARKER
    assert "triton" in torch_dependencies
    assert torch_dependencies["triton"] == LINUX_X86_64_MARKER
    assert source_only_packages == {"docopt"}

    return {
        "direct_ci_packages": len(ci_pins),
        "direct_integrity_packages": len(integrity_pins),
        "direct_cff_packages": len(cff_pins),
        "locked_registry_packages": len(registry_packages),
        "locked_artifacts": artifact_count,
        "sha256_hashes": hash_count,
        "source_only_packages": len(source_only_packages),
    }


def main() -> int:
    report = validate_dependency_lock()
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
