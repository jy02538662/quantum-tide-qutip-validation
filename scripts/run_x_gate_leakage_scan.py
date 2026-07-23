from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.hamiltonians import oscillator_projectors, x_gate_full_space_hamiltonian, x_gate_initial_state
from src.metrics import FIG_DIR, ensure_outputs, save_csv
from src.params import PARAMS
from src.schedules import gate_time_x


def run_point(
    qt,
    v_low: float,
    leakage_coupling: float,
    oscillator_ratio: float,
    leakage_alpha: float,
    samples: int = 120,
):
    h = x_gate_full_space_hamiltonian(
        v_low,
        leakage_coupling=leakage_coupling,
        oscillator_ratio=oscillator_ratio,
        leakage_alpha=leakage_alpha,
        qubit_detuning=0.0,
    )
    psi0 = x_gate_initial_state()
    _, high_projector = oscillator_projectors()
    target = qt.tensor(qt.basis(2, 1), qt.basis(PARAMS.oscillator_levels, 0))
    t_gate = float(gate_time_x(v_low, omega_rabi0=PARAMS.omega_rabi0, v0=PARAMS.v0))
    times = np.linspace(0.0, t_gate, samples)
    result = qt.mesolve(
        h,
        psi0,
        times,
        c_ops=[],
        e_ops=[high_projector],
        options={"store_states": True},
    )
    high = np.asarray(result.expect[0])
    final_state = result.states[-1]
    f_x = abs(target.overlap(final_state)) ** 2
    return float(np.max(high)), float(high[-1]), float(f_x), t_gate


def main():
    ensure_outputs()
    try:
        import qutip as qt
    except Exception as exc:
        print(f"[x_gate_leakage_scan] skipped: QuTiP unavailable ({exc})")
        return

    v_values = np.array([0.08, 0.10, 0.14, 0.18, 0.24, 0.32])
    coupling_values = np.array([0.004, 0.006, 0.008, 0.010, 0.012, 0.016])
    ratio_values = np.array([2.0, 2.4, 3.0, 3.6, 4.5])
    alpha_values = np.array([0.5, 3.0])
    leakage_threshold = 1e-3
    fidelity_threshold = 0.999

    rows = {
        "alpha": [],
        "v_low": [],
        "leakage_coupling": [],
        "oscillator_ratio": [],
        "max_high_population": [],
        "final_high_population": [],
        "F_X": [],
        "t_gate": [],
        "passes_leakage": [],
        "passes_fidelity": [],
        "passes_both": [],
    }

    total = len(alpha_values) * len(v_values) * len(coupling_values) * len(ratio_values)
    count = 0
    for alpha in alpha_values:
        for v_low in v_values:
            for coupling in coupling_values:
                for ratio in ratio_values:
                    count += 1
                    max_high, final_high, f_x, t_gate = run_point(qt, v_low, coupling, ratio, alpha)
                    passes_leakage = max_high < leakage_threshold
                    passes_fidelity = f_x > fidelity_threshold
                    rows["alpha"].append(alpha)
                    rows["v_low"].append(v_low)
                    rows["leakage_coupling"].append(coupling)
                    rows["oscillator_ratio"].append(ratio)
                    rows["max_high_population"].append(max_high)
                    rows["final_high_population"].append(final_high)
                    rows["F_X"].append(f_x)
                    rows["t_gate"].append(t_gate)
                    rows["passes_leakage"].append(int(passes_leakage))
                    rows["passes_fidelity"].append(int(passes_fidelity))
                    rows["passes_both"].append(int(passes_leakage and passes_fidelity))
                    print(
                        f"[{count:03d}/{total}] alpha={alpha:.1f} v={v_low:.3f} "
                        f"g={coupling:.3f} ratio={ratio:.1f} "
                        f"max_high={max_high:.3e} F_X={f_x:.6f} "
                        f"pass={passes_leakage and passes_fidelity}"
                    )

    save_csv("x_gate_leakage_scan_alpha.csv", rows)

    alpha_arr = np.asarray(rows["alpha"])
    max_high = np.asarray(rows["max_high_population"])
    f_x = np.asarray(rows["F_X"])
    passes_both = np.asarray(rows["passes_both"], dtype=bool)

    print("\n[x_gate_leakage_scan] summary")
    print(f"  total points: {total}")
    for alpha in alpha_values:
        mask = alpha_arr == alpha
        alpha_passes = int(np.sum(passes_both & mask))
        best_idx_local = np.where(mask)[0][np.argmax(f_x[mask] - 10 * np.maximum(max_high[mask] - leakage_threshold, 0))]
        print(f"  alpha={alpha:.1f} pass count: {alpha_passes}/{int(np.sum(mask))}")
        print(
            "    best combined score: "
            f"v={rows['v_low'][best_idx_local]:.3f}, "
            f"g={rows['leakage_coupling'][best_idx_local]:.3f}, "
            f"ratio={rows['oscillator_ratio'][best_idx_local]:.1f}, "
            f"max_high={rows['max_high_population'][best_idx_local]:.3e}, "
            f"F_X={rows['F_X'][best_idx_local]:.6f}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, alpha in zip(axes, alpha_values):
        mask = alpha_arr == alpha
        sc = ax.scatter(
            np.asarray(rows["v_low"])[mask],
            np.asarray(rows["leakage_coupling"])[mask],
            c=np.log10(max_high[mask]),
            s=70 + 30 * np.asarray(rows["oscillator_ratio"])[mask],
            cmap="viridis_r",
        )
        ax.set_xscale("log")
        ax.set_xlabel("v_low / v0")
        ax.set_title(f"alpha = {alpha:.1f}")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("leakage coupling g0")
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist())
    cbar.set_label("log10(max high-level population)")
    fig.suptitle("X-gate leakage scan; marker size = oscillator ratio")
    fig.savefig(FIG_DIR / "x_gate_leakage_scan_alpha.png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
