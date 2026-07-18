#!/usr/bin/env python3
"""Small deterministic demonstration of the local surrogate API."""

from __future__ import annotations

import numpy as np

from ..src import evaluate_reference_endpoints, evaluate_surrogate_curve


def main() -> None:
    """Evaluate a fixed synthetic channel and print computed values."""

    channel = np.array(
        [[1.0 + 0.2j, 0.4 - 0.1j], [0.1 + 0.3j, 0.8 + 0.0j]],
        dtype=np.complex128,
    )
    endpoint = evaluate_reference_endpoints(
        channel,
        T=8,
        sigma_c2=0.5,
        sigma_s2=0.25,
        power_per_tx=1.0,
        Jp=0.2 * np.eye(2),
    )
    crb, rate, _ = evaluate_surrogate_curve(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        channel,
        T=8,
        sigma_c2=0.5,
        sigma_s2=0.25,
        power_per_tx=1.0,
        Jp=0.2 * np.eye(2),
    )

    print("Educational Gaussian-ISAC surrogate (synthetic channel)")
    print(f"isotropic rate: {endpoint['isotropic_rate']:.9f} nats/use")
    print(
        "water-filling rate: "
        f"{endpoint['water_filling_rate']:.9f} nats/use"
    )
    print("alpha, CRB, rate")
    for alpha, crb_value, rate_value in zip(
        np.linspace(0, 1, 5),
        crb,
        rate,
        strict=True,
    ):
        print(f"{alpha:.2f}, {crb_value:.9f}, {rate_value:.9f}")


if __name__ == "__main__":
    main()
