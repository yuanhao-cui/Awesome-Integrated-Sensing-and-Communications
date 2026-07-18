"""Execute the checked simulation artifact and validate its numeric certificate."""

import json
import subprocess
import sys


MODULE = (
    "code.baselines.isac_energy_efficient_beamforming.examples."
    "verify_reference_slice"
)


def test_reference_slice_cli_emits_passing_numeric_certificate() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", MODULE, "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    certificate = json.loads(completed.stdout)
    assert certificate["status"] == "pass"
    assert certificate["claim"] == "equation-level-single-user-fixed-direction"
    assert certificate["dinkelbach"]["power_error_watt"] <= certificate[
        "dinkelbach"
    ]["allowed_power_error_watt"]
    assert certificate["crb"]["relative_error"] <= certificate["crb"][
        "relative_tolerance"
    ]
