from pathlib import Path
import sys
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.hamiltonians import oscillator_projectors, x_gate_drag_hamiltonian
from src.metrics import FIG_DIR, ensure_outputs, save_csv
from src.params import PARAMS
from src.schedules import gate_time_x


def run_point(qt, point: dict, oscillator_levels: int):
    params = replace(PARAMS, oscillator_levels=oscillator_levels)
    v_low = point["v_low"]
    g0 = point["g0_over_omega_rabi0"] * params.omega_rabi0
    total_time = float(gate_time_x(v_low, omega_rabi0=params.omega_rabi0, v0=params.v0))
    h = x_gate_drag_hamiltonian(
        v_low,
        total_time,
        params=params,
        leakage_coupling=g0,
        oscillator_ratio=point["oscillator_ratio"],
        leakage_alpha=3.0,
        drag_lambda=0.0,
        qubit_detuning=0.0,
    )
    psi0 = qt.tensor(qt.basis(2, 0), qt.basis(oscillator_levels, 0))
    target = qt.tensor(qt.basis(2, 1), qt.basis(oscillator_levels, 0))
    _, high_projector = oscillator_projectors(params=params)
    times = np.linspace(0.0, total_time, 220)
    result = qt.mesolve(
        h,
        psi0,
        times,
        c_ops=[],
        e_ops=[high_projector],
        options={"store_states": True},
    )
    final_state = result.states[-1]
    f_g = abs(target.overlap(final_state)) ** 2
    high = np.asarray(result.expect[0])
    return float(f_g), float(np.max(high)), float(high[-1]), total_time


def main():
    ensure_outputs()
    try:
        import qutip as qt
    except Exception as exc:
        print(f"[truncation_convergence] skipped: QuTiP unavailable ({exc})")
        return

    points = [
        {"label": "r002_v024_ratio3", "v_low": 0.240, "g0_over_omega_rabi0": 0.020, "oscillator_ratio": 3.0},
        {"label": "r003_v024_ratio45", "v_low": 0.240, "g0_over_omega_rabi0": 0.030, "oscillator_ratio": 4.5},
        {"label": "r004_v024_ratio55", "v_low": 0.240, "g0_over_omega_rabi0": 0.040, "oscillator_ratio": 5.5},
        {"label": "near_fail_v018", "v_low": 0.180, "g0_over_omega_rabi0": 0.020, "oscillator_ratio": 7.0},
    ]
    n_values = [5, 8, 10]
    leakage_threshold = 1e-3
    fidelity_threshold = 0.999

    rows = {
        "point_label": [],
        "oscillator_levels": [],
        "v_low": [],
        "g0_over_omega_rabi0": [],
        "oscillator_ratio": [],
        "t_gate": [],
        "F_g": [],
        "max_high_population": [],
        "final_high_population": [],
        "passes_leakage": [],
        "passes_state_transfer": [],
        "passes_both": [],
    }

    for point in points:
        for n in n_values:
            f_g, max_high, final_high, t_gate = run_point(qt, point, n)
            pass_l = max_high < leakage_threshold
            pass_f = f_g > fidelity_threshold
            rows["point_label"].append(point["label"])
            rows["oscillator_levels"].append(n)
            rows["v_low"].append(point["v_low"])
            rows["g0_over_omega_rabi0"].append(point["g0_over_omega_rabi0"])
            rows["oscillator_ratio"].append(point["oscillator_ratio"])
            rows["t_gate"].append(t_gate)
            rows["F_g"].append(f_g)
            rows["max_high_population"].append(max_high)
            rows["final_high_population"].append(final_high)
            rows["passes_leakage"].append(int(pass_l))
            rows["passes_state_transfer"].append(int(pass_f))
            rows["passes_both"].append(int(pass_l and pass_f))
            print(
                f"{point['label']} N={n} F_g={f_g:.6f} "
                f"P_high={max_high:.3e} pass={pass_l and pass_f}"
            )

    save_csv("truncation_convergence_smooth_xgate.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for point in points:
        mask = [label == point["label"] for label in rows["point_label"]]
        n = np.asarray(rows["oscillator_levels"])[mask]
        high = np.asarray(rows["max_high_population"])[mask]
        fidelity = np.asarray(rows["F_g"])[mask]
        axes[0].plot(n, high, marker="o", label=point["label"])
        axes[1].plot(n, fidelity, marker="o", label=point["label"])
    axes[0].axhline(leakage_threshold, color="tab:red", linestyle="--")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("oscillator truncation N")
    axes[0].set_ylabel("P_high,max")
    axes[0].set_title("High-level population convergence")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].axhline(fidelity_threshold, color="tab:red", linestyle="--")
    axes[1].set_xlabel("oscillator truncation N")
    axes[1].set_ylabel("F_g")
    axes[1].set_title("State-transfer fidelity convergence")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "truncation_convergence_smooth_xgate.png", dpi=180)


if __name__ == "__main__":
    main()
