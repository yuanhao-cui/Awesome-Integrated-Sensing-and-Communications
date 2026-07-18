#!/usr/bin/env python3
"""
Generate figures for OFDM Ambiguity Function Analysis

Produces 4 figures:
1. OFDM ambiguity function 3D surface
2. OFDM ambiguity function contour
3. LFM ambiguity function contour (comparison)
4. Range/Doppler resolution comparison
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ofdm_ambiguity import (
    generate_ofdm_signal,
    compute_ambiguity_function,
    generate_lfm_signal,
    compute_range_resolution,
    compute_doppler_resolution,
    compute_papr,
)


def generate_all_figures(output_dir: str = "figures"):
    """
    Generate all 4 required figures.

    Parameters
    ----------
    output_dir : str
        Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)

    # Parameters
    n_subcarriers = 64
    cp_len = 16
    lfm_bandwidth = 20e6
    fs = 40e6
    pulse_width = 10e-6

    print("Generating figures for OFDM Ambiguity Function Analysis...")
    print("=" * 60)

    # Generate signals
    print("1. Generating OFDM signal...")
    ofdm_signal = generate_ofdm_signal(
        n_subcarriers, cp_len, rng=np.random.default_rng(42)
    )

    print("2. Generating LFM signal...")
    lfm_signal = generate_lfm_signal(lfm_bandwidth, pulse_width, fs)

    # Define ambiguity function ranges
    # Use samples for delay, normalized frequency for Doppler
    signal_len = len(ofdm_signal)
    tau_range = np.arange(-signal_len // 2, signal_len // 2 + 1)
    nu_range = (np.arange(81) - 40) / 81.0

    print("3. Computing OFDM ambiguity function...")
    af_ofdm = compute_ambiguity_function(ofdm_signal, tau_range, nu_range)

    print("4. Computing LFM ambiguity function...")
    # Use smaller range for LFM (different signal length)
    lfm_tau = np.arange(-100, 101, 2)
    lfm_nu = (np.arange(81) - 40) / 81.0
    af_lfm = compute_ambiguity_function(lfm_signal, lfm_tau, lfm_nu)

    # Figure 1: OFDM Ambiguity Function 3D
    print("\n--- Figure 1: OFDM Ambiguity Function 3D ---")
    fig1 = plt.figure(figsize=(12, 9))
    ax1 = fig1.add_subplot(111, projection='3d')

    TAU, NU = np.meshgrid(tau_range, nu_range)
    af_db = 10 * np.log10(af_ofdm + 1e-15)
    af_db = np.clip(af_db, -40, 0)

    surf1 = ax1.plot_surface(TAU, NU, af_db, cmap='jet', alpha=0.85)
    ax1.set_xlabel('Delay τ (samples)', fontsize=12)
    ax1.set_ylabel('Doppler ν (cycles/sample)', fontsize=12)
    ax1.set_zlabel('|χ(τ, ν)|² (dB)', fontsize=12)
    ax1.set_title(f'OFDM Ambiguity Function\n(N={n_subcarriers} subcarriers, CP={cp_len})', fontsize=14)
    fig1.colorbar(surf1, shrink=0.5, aspect=5)
    ax1.view_init(elev=25, azim=45)

    fig1_path = os.path.join(output_dir, 'ofdm_ambiguity_3d.png')
    fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"   Saved: {fig1_path}")

    # Figure 2: OFDM Ambiguity Function Contour
    print("\n--- Figure 2: OFDM Ambiguity Function Contour ---")
    fig2, ax2 = plt.subplots(figsize=(10, 8))

    levels = [-40, -30, -20, -10, -6, -3]
    contour2 = ax2.contour(TAU, NU, af_db, levels=levels, colors='black', linewidths=1)
    contourf2 = ax2.contourf(TAU, NU, af_db, levels=50, cmap='jet')
    ax2.clabel(contour2, inline=True, fontsize=9, fmt='%.0f dB')
    ax2.set_xlabel('Delay τ (samples)', fontsize=12)
    ax2.set_ylabel('Doppler ν (cycles/sample)', fontsize=12)
    ax2.set_title(f'OFDM Ambiguity Function Contour\n(N={n_subcarriers} subcarriers, CP={cp_len})', fontsize=14)
    fig2.colorbar(contourf2, label='|χ(τ, ν)|² (dB)')
    ax2.grid(True, alpha=0.3)

    fig2_path = os.path.join(output_dir, 'ofdm_ambiguity_contour.png')
    fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"   Saved: {fig2_path}")

    # Figure 3: LFM Ambiguity Function Contour (Comparison)
    print("\n--- Figure 3: LFM Ambiguity Function Contour (Comparison) ---")
    fig3, ax3 = plt.subplots(figsize=(10, 8))

    LFM_TAU, LFM_NU = np.meshgrid(lfm_tau, lfm_nu)
    af_lfm_db = 10 * np.log10(af_lfm + 1e-15)
    af_lfm_db = np.clip(af_lfm_db, -40, 0)

    contour3 = ax3.contour(LFM_TAU, LFM_NU, af_lfm_db, levels=levels, colors='black', linewidths=1)
    contourf3 = ax3.contourf(LFM_TAU, LFM_NU, af_lfm_db, levels=50, cmap='jet')
    ax3.clabel(contour3, inline=True, fontsize=9, fmt='%.0f dB')
    ax3.set_xlabel('Delay τ (samples)', fontsize=12)
    ax3.set_ylabel('Doppler ν (cycles/sample)', fontsize=12)
    ax3.set_title(f'LFM (Chirp) Ambiguity Function Contour\n(B={lfm_bandwidth/1e6:.0f} MHz, T={pulse_width*1e6:.0f} μs)', fontsize=14)
    fig3.colorbar(contourf3, label='|χ(τ, ν)|² (dB)')
    ax3.grid(True, alpha=0.3)

    fig3_path = os.path.join(output_dir, 'lfm_ambiguity_contour.png')
    fig3.savefig(fig3_path, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"   Saved: {fig3_path}")

    # Figure 4: Range/Doppler Resolution Comparison
    print("\n--- Figure 4: Range/Doppler Resolution Comparison ---")
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6))

    # Range resolution vs bandwidth
    bandwidths = np.array([5, 10, 20, 50, 100, 200, 500, 1000])  # MHz
    range_res = compute_range_resolution(bandwidths * 1e6)

    ax4a.semilogx(bandwidths, range_res, 'b-o', linewidth=2, markersize=8, label='Theoretical')
    ax4a.set_xlabel('Bandwidth (MHz)', fontsize=12)
    ax4a.set_ylabel('Range Resolution (m)', fontsize=12)
    ax4a.set_title('Range Resolution vs Bandwidth', fontsize=14)
    ax4a.grid(True, alpha=0.3, which='both')
    ax4a.legend(fontsize=11)

    # Doppler resolution vs coherent time
    coherent_times = np.array([0.1, 0.5, 1, 5, 10, 50, 100, 500])  # ms
    doppler_res = compute_doppler_resolution(coherent_times * 1e-3)

    ax4b.loglog(coherent_times, doppler_res, 'r-o', linewidth=2, markersize=8, label='Theoretical')
    ax4b.set_xlabel('Coherent Processing Interval (ms)', fontsize=12)
    ax4b.set_ylabel('Doppler Resolution (Hz)', fontsize=12)
    ax4b.set_title('Doppler Resolution vs Coherent Time', fontsize=14)
    ax4b.grid(True, alpha=0.3, which='both')
    ax4b.legend(fontsize=11)

    # Add annotations for ISAC parameters
    ax4a.axvline(x=20, color='gray', linestyle='--', alpha=0.5)
    ax4a.annotate('Selected 20 MHz point', xy=(20, compute_range_resolution(20e6)),
                   xytext=(30, 20), fontsize=10, color='gray',
                   arrowprops=dict(arrowstyle='->', color='gray'))

    ax4b.axvline(x=10, color='gray', linestyle='--', alpha=0.5)
    ax4b.annotate('10 ms CPI', xy=(10, compute_doppler_resolution(10e-3)),
                   xytext=(30, 5), fontsize=10, color='gray',
                   arrowprops=dict(arrowstyle='->', color='gray'))

    fig4.suptitle('Resolution Limits for ISAC Waveforms', fontsize=16, y=1.02)
    fig4.tight_layout()

    fig4_path = os.path.join(output_dir, 'resolution_comparison.png')
    fig4.savefig(fig4_path, dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print(f"   Saved: {fig4_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary of ISAC Resolution Analysis:")
    print("-" * 40)
    print("OFDM Signal Parameters:")
    print(f"  - Subcarriers: {n_subcarriers}")
    print(f"  - Cyclic Prefix: {cp_len}")
    print("  - Occupied bandwidth/sample rate: normalized (not parameterized)")
    print(f"  - Signal Length: {len(ofdm_signal)} samples")
    print(f"  - PAPR: {compute_papr(ofdm_signal):.2f} ({10*np.log10(compute_papr(ofdm_signal)):.1f} dB)")

    print("\nLFM Signal Parameters:")
    print(f"  - Bandwidth: {lfm_bandwidth/1e6:.0f} MHz")
    print(f"  - Pulse Width: {pulse_width*1e6:.1f} μs")
    print(f"  - Signal Length: {len(lfm_signal)} samples")
    print(f"  - PAPR: {compute_papr(lfm_signal):.2f}")

    print("\nTheoretical Resolution:")
    print(
        "  - Illustrative c/(2B) cell at the LFM bandwidth: "
        f"{compute_range_resolution(lfm_bandwidth):.2f} m"
    )
    print(f"  - Doppler Resolution: {compute_doppler_resolution(pulse_width):.2f} Hz")

    print("\nThese diagnostics describe only the generated realization and chosen grids.")

    print("\n✓ All figures generated successfully!")

    return {
        'ofdm_3d': fig1_path,
        'ofdm_contour': fig2_path,
        'lfm_contour': fig3_path,
        'resolution': fig4_path
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate OFDM Ambiguity Function figures")
    parser.add_argument("--output", "-o", default="figures", help="Output directory")
    args = parser.parse_args()

    figures = generate_all_figures(args.output)

    print(f"\nFigures saved to: {args.output}/")
    for name, path in figures.items():
        print(f"  - {name}: {path}")
