"""Tests for endpoint, curve, demo, and certificate integration."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from ..examples.verify_surrogate import build_certificate
from ..src.bounds import evaluate_reference_endpoints, evaluate_surrogate_curve


def test_reference_endpoints_are_finite_and_rate_ordered() -> None:
    channel = np.array([[1.0, 0.2], [0.1, 0.5]], dtype=np.complex128)
    endpoint = evaluate_reference_endpoints(channel, 5, 0.4, 0.3, 1.0)
    assert float(endpoint["isotropic_crb"]) > 0
    assert float(endpoint["water_filling_crb"]) > 0
    assert float(endpoint["water_filling_rate"]) >= float(endpoint["isotropic_rate"])
    np.testing.assert_allclose(
        np.trace(endpoint["water_filling_covariance"]),  # type: ignore[arg-type]
        2,
        atol=1e-14,
    )


def test_surrogate_curve_shape_endpoints_and_repeatability() -> None:
    channel = np.array([[1.0, 0.2], [0.1, 0.5]], dtype=np.complex128)
    arguments = dict(
        alphas=[0.0, 0.2, 0.7, 1.0],
        Hc=channel,
        T=5,
        sigma_c2=0.4,
        sigma_s2=0.3,
        power_per_tx=1.0,
    )
    first = evaluate_surrogate_curve(**arguments)
    second = evaluate_surrogate_curve(**arguments)
    for first_array, second_array in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_array, second_array)
    crb, rate, covariance = first
    assert crb.shape == (4,)
    assert rate.shape == (4,)
    assert covariance.shape == (4, 2, 2)
    assert rate[-1] >= rate[0]


@pytest.mark.parametrize("alphas", [[], [[0.2]], [-0.1], [np.nan]])
def test_surrogate_curve_rejects_invalid_alpha_sequences(alphas: object) -> None:
    with pytest.raises(ValueError, match="alpha"):
        evaluate_surrogate_curve(alphas, np.eye(2), 2, 1, 1, 1)  # type: ignore[arg-type]


def test_certificate_passes_all_numeric_oracles() -> None:
    certificate = build_certificate()
    assert certificate["status"] == "pass"
    assert certificate["claim"] == {
        "level": "educational-surrogate",
        "paper_figure_parity": False,
    }
    assert all(certificate["checks"].values())  # type: ignore[union-attr]


def test_certificate_module_is_executable_json() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "code.baselines.isac_capacity_distortion.examples.verify_surrogate",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    certificate = json.loads(completed.stdout)
    assert certificate["status"] == "pass"
    assert completed.stderr == ""


def test_demo_module_runs_without_writing_results() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "code.baselines.isac_capacity_distortion.examples.demo",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "synthetic channel" in completed.stdout
    assert "water-filling rate" in completed.stdout
    assert completed.stderr == ""
