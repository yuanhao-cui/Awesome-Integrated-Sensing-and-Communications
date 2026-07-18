#!/usr/bin/env python3
"""
Generate synthetic diagnostic figures for the checked CSI-ratio estimator.

Produces four publication-quality figures using synthetic CSI data:
  - B1: CSI-ratio circle in the complex plane
  - B2: circle-phase estimate over sliding windows
  - B3: circle-phase estimation error vs SNR
  - B4: CSI-ratio trajectory with circle fitting visualization
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as CirclePatch

BASELINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASELINE_DIR)

from src.circle_fit import least_squares_circle_fit  # noqa: E402
from src.csi_ratio import compute_csi_ratio  # noqa: E402
from src.mobius_estimator import mobius_doppler_estimate  # noqa: E402
from src.signal_model import csi_with_doppler  # noqa: E402

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Helper: generate CSI data ─────────────────────────────────────────────────
def generate_csi(f_D=50.0, fs=2000.0, duration=0.5, snr_db=np.inf, seed=42):
    """Return (t, H1, H2, R) for a given Doppler frequency and SNR."""
    rng = np.random.default_rng(seed)
    N = int(fs * duration)
    t = np.arange(N) / fs
    H1, H2 = csi_with_doppler(
        t,
        f_D=f_D,
        snr_db=snr_db,
        amplitude_ratio=1.2,
        phase_offset=np.pi / 6,
        cfo_hz=50.0,
        tmo_hz=10.0,
        rng=rng,
    )
    R = compute_csi_ratio(H1, H2)
    return t, H1, H2, R


# ══════════════════════════════════════════════════════════════════════════════
# Figure B1: CSI-Ratio Circle in Complex Plane
# ══════════════════════════════════════════════════════════════════════════════
def figure_b1():
    """Show the CSI-ratio samples forming a circle in the complex plane."""
    print("  Generating Figure B1 (CSI-ratio circle)...")
    t, H1, H2, R = generate_csi(f_D=50.0, snr_db=40.0)

    A, B, r = least_squares_circle_fit(R)
    fig, ax = plt.subplots(figsize=(7, 7))

    # Colour-coded by time
    sc = ax.scatter(
        np.real(R), np.imag(R), c=t * 1000, cmap="plasma", s=12, alpha=0.85, zorder=5
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.78, pad=0.02)
    cb.set_label("Time (ms)")

    # Fitted circle
    circle = CirclePatch(
        (A, B), r, fill=False, edgecolor="#E53935", lw=2, ls="--", label="Fitted circle"
    )
    ax.add_patch(circle)
    ax.plot(
        A,
        B,
        "rx",
        ms=12,
        mew=3,
        label=f"Center {A:.3f} {B:+.3f}j",
    )

    # Unit circle for reference
    theta_circ = np.linspace(0, 2 * np.pi, 200)
    ax.plot(
        np.cos(theta_circ),
        np.sin(theta_circ),
        color="grey",
        lw=0.8,
        ls=":",
        alpha=0.5,
        label="Unit circle",
    )

    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_title(
        "B1 — Synthetic Pure-Rotation CSI Ratio  "
        "($f_D=50$ Hz, SNR=40 dB)"
    )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    path = os.path.join(RESULTS_DIR, "B1_csi_ratio_circle.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"    → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure B2: Sliding-window rotation estimate
# ══════════════════════════════════════════════════════════════════════════════
def figure_b2():
    """Plot the circle-phase estimate for deterministic sliding windows."""
    print("  Generating Figure B2 (windowed circle-phase estimate)...")
    f_D_true = 50.0
    fs = 2000.0
    duration = 1.0
    snr_db = 30.0
    window_ms = 100  # sliding window in ms
    step_ms = 10
    window_N = int(fs * window_ms / 1000)
    step_N = int(fs * step_ms / 1000)

    t, H1, H2, R = generate_csi(f_D=f_D_true, fs=fs, duration=duration, snr_db=snr_db)
    T_s = 1.0 / fs

    t_centers, mobius_ests = [], []
    for start in range(0, len(R) - window_N + 1, step_N):
        R_w = R[start : start + window_N]
        t_w = t[start : start + window_N]
        t_centers.append(np.mean(t_w) * 1000)

        m = mobius_doppler_estimate(R_w, T_s)
        mobius_ests.append(m["f_D"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(
        f_D_true, color="k", ls="--", lw=1.5, label=f"True $f_D$ = {f_D_true} Hz"
    )
    ax.plot(
        t_centers,
        mobius_ests,
        "o-",
        ms=4,
        color="#1976D2",
        label="Checked circle-phase estimator",
    )

    ax.set_xlabel("Window center time (ms)")
    ax.set_ylabel("Estimated $f_D$ (Hz)")
    ax.set_title(
        "B2 — Synthetic Windowed Circle-Phase Diagnostic  "
        f"($f_D$={f_D_true} Hz, SNR={snr_db} dB, {window_ms} ms window)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    path = os.path.join(RESULTS_DIR, "B2_estimation_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"    → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure B3: Estimation Error vs SNR
# ══════════════════════════════════════════════════════════════════════════════
def figure_b3():
    """Sweep SNR and report the circle-phase adapter's absolute error."""
    print("  Generating Figure B3 (error vs SNR)...")
    f_D_true = 50.0
    fs = 2000.0
    duration = 0.5
    snr_range = np.arange(0, 45, 3)  # 0–42 dB
    n_trials = 20

    median_error = np.zeros(len(snr_range))
    lower_quartile = np.zeros(len(snr_range))
    upper_quartile = np.zeros(len(snr_range))
    valid_fraction = np.zeros(len(snr_range))

    for i_snr, snr_db in enumerate(snr_range):
        m_errs = []
        for trial in range(n_trials):
            t, H1, H2, R = generate_csi(
                f_D=f_D_true, fs=fs, duration=duration, snr_db=snr_db, seed=trial * 100 + 7
            )
            T_s = 1.0 / fs
            try:
                estimate = mobius_doppler_estimate(R, T_s)
            except ValueError:
                continue
            m_errs.append(abs(abs(estimate["f_D"]) - f_D_true))
        valid_fraction[i_snr] = len(m_errs) / n_trials
        if m_errs:
            (
                lower_quartile[i_snr],
                median_error[i_snr],
                upper_quartile[i_snr],
            ) = np.quantile(m_errs, [0.25, 0.5, 0.75])
        else:
            lower_quartile[i_snr] = np.nan
            median_error[i_snr] = np.nan
            upper_quartile[i_snr] = np.nan

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.semilogy(
        snr_range,
        median_error,
        "o-",
        ms=5,
        color="#1976D2",
        label="Median absolute error",
    )
    ax.fill_between(
        snr_range,
        lower_quartile,
        upper_quartile,
        color="#1976D2",
        alpha=0.2,
        label="Interquartile range",
    )

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(r"Median |$\hat{f}_D - f_D$| (Hz)")
    validity_axis = ax.twinx()
    validity_axis.plot(
        snr_range,
        valid_fraction,
        "s--",
        ms=4,
        color="#616161",
        label="Accepted-trial fraction",
    )
    validity_axis.set_ylabel("Accepted-trial fraction")
    validity_axis.set_ylim(-0.03, 1.03)

    ax.set_title(
        "B3 — Synthetic Pure-Rotation Error Diagnostic  "
        "($f_D=50$ Hz, 20 fixed seeds)"
    )
    ax.grid(True, which="both", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    handles_valid, labels_valid = validity_axis.get_legend_handles_labels()
    ax.legend(handles + handles_valid, labels + labels_valid, loc="upper right")

    path = os.path.join(RESULTS_DIR, "B3_error_vs_snr.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"    → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure B4: CSI-Ratio Trajectory with Circle Fitting
# ══════════════════════════════════════════════════════════════════════════════
def figure_b4():
    """Visualize the CSI-ratio trajectory, circle fit, and Mobius shift."""
    print("  Generating Figure B4 (trajectory + circle fitting)...")
    t, H1, H2, R = generate_csi(f_D=50.0, snr_db=35.0, duration=0.2)

    A, B, r = least_squares_circle_fit(R)
    C_0 = A + 1j * B
    R_shifted = R - C_0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # ── Left: Original trajectory ──
    sc1 = ax1.scatter(
        np.real(R), np.imag(R), c=t * 1000, cmap="plasma", s=10, alpha=0.85, zorder=5
    )
    cb1 = fig.colorbar(sc1, ax=ax1, shrink=0.82)
    cb1.set_label("Time (ms)")
    circle1 = CirclePatch(
        (A, B), r, fill=False, edgecolor="#E53935", lw=2, ls="--"
    )
    ax1.add_patch(circle1)
    ax1.plot(A, B, "rx", ms=10, mew=2.5)
    ax1.annotate(
        f"Center=({A:.3f}, {B:.3f})",
        (A, B),
        textcoords="offset points",
        xytext=(12, -15),
        fontsize=9,
        color="#E53935",
    )

    # Mark start & end
    ax1.plot(np.real(R[0]), np.imag(R[0]), "go", ms=8, zorder=10, label="Start")
    ax1.plot(np.real(R[-1]), np.imag(R[-1]), "mD", ms=7, zorder=10, label="End")

    ax1.set_xlabel("Real")
    ax1.set_ylabel("Imaginary")
    ax1.set_title("CSI-Ratio Trajectory (original)")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=9)

    # ── Right: Shifted to origin ──
    sc2 = ax2.scatter(
        np.real(R_shifted),
        np.imag(R_shifted),
        c=t * 1000,
        cmap="plasma",
        s=10,
        alpha=0.85,
        zorder=5,
    )
    cb2 = fig.colorbar(sc2, ax=ax2, shrink=0.82)
    cb2.set_label("Time (ms)")

    theta_c = np.linspace(0, 2 * np.pi, 200)
    ax2.plot(
        r * np.cos(theta_c),
        r * np.sin(theta_c),
        color="#E53935",
        lw=2,
        ls="--",
        label=f"Circle (r={r:.3f})",
    )
    ax2.plot(0, 0, "k+", ms=10, mew=2, label="Origin")

    # Arrow showing rotation direction
    mid_idx = len(R_shifted) // 4
    ax2.annotate(
        "",
        xy=(np.real(R_shifted[mid_idx + 5]), np.imag(R_shifted[mid_idx + 5])),
        xytext=(np.real(R_shifted[mid_idx]), np.imag(R_shifted[mid_idx])),
        arrowprops=dict(arrowstyle="->", color="green", lw=2),
    )

    ax2.set_xlabel("Real")
    ax2.set_ylabel("Imaginary")
    ax2.set_title("Shifted to Origin  ($R_s = R - C_0$)")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=9)

    fig.suptitle(
        "B4 — Synthetic CSI-Ratio Trajectory & Circle Fit  ($f_D=50$ Hz)",
        fontsize=15,
        y=1.01,
    )
    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, "B4_trajectory_circle_fit.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"    → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating local synthetic diagnostic figures...")
    figure_b1()
    figure_b2()
    figure_b3()
    figure_b4()
    print("Done — all figures saved to results/")
