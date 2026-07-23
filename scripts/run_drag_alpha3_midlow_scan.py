from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.hamiltonians import oscillator_projectors, x_gate_drag_hamiltonian, x_gate_initial_state
from src.metrics import FIG_DIR, ensure_outputs, save_csv
from src.params import PARAMS
from src.schedules import gate_time_x


def run_point(qt, v_low: float, coupling_ratio: float, oscillator_ratio: float, drag_lambda: float):
    g0 = coupling_ratio * PARAMS.omega_rabi0
    total_time = float(gate_time_x(v_low, omega_rabi0=PARAMS.omega_rabi0, v0=PARAMS.v0))
    h = x_gate_drag_hamiltonian(
        v_low,
        total_time,
        leakage_coupling=g0,
        oscillator_ratio=oscillator_ratio,
        leakage_alpha=3.0,
        drag_lambda=drag_lambda,
        qubit_detuning=0.0,
    )
    psi0 = x_gate_initial_state()
    _, high_projector = oscillator_projectors()
    target = qt.tensor(qt.basis(2, 1), qt.basis(PARAMS.oscillator_levels, 0))
    times = np.linspace(0.0, total_time, 180)
    result = qt.mesolve(
        h,
        psi0,
        times,
        c_ops=[],
        e_ops=[high_projector],
        options={"store_states": True},
    )
    high = np.asarray(result.expect[0])
    f_x = abs(target.overlap(result.states[-1])) ** 2
    return g0, total_time, float(np.max(high)), float(high[-1]), float(f_x)


def main():
    ensure_outputs()
    try:
        import qutip as qt
    except Exception as exc:
        print(f"[drag_scan] skipped: QuTiP unavailable ({exc})")
        return

    v_values = np.array([0.18, 0.24, 0.32])
    coupling_ratios = np.array([0.020, 0.030, 0.040, 0.050])
    oscillator_ratios = np.array([3.0, 3.6, 4.5, 5.5, 7.0])
    drag_values = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    leakage_threshold = 1e-3
    fidelity_threshold = 0.999

    rows = {
        "v_low": [],
        "g0_over_omega_rabi0": [],
        "g0": [],
        "oscillator_ratio": [],
        "drag_lambda": [],
        "t_gate": [],
        "max_high_population": [],
        "final_high_population": [],
        "F_X": [],
        "passes_leakage": [],
        "passes_fidelity": [],
        "passes_both": [],
    }

    total = len(v_values) * len(coupling_ratios) * len(oscillator_ratios) * len(drag_values)
    count = 0
    for v_low in v_values:
        for coupling_ratio in coupling_ratios:
            for oscillator_ratio in oscillator_ratios:
                for drag_lambda in drag_values:
                    count += 1
                    g0, total_time, max_high, final_high, f_x = run_point(
                        qt,
                        v_low,
                        coupling_ratio,
                        oscillator_ratio,
                        drag_lambda,
                    )
                    passes_leakage = max_high < leakage_threshold
                    passes_fidelity = f_x > fidelity_threshold
                    passes_both = passes_leakage and passes_fidelity
                    rows["v_low"].append(v_low)
                    rows["g0_over_omega_rabi0"].append(coupling_ratio)
                    rows["g0"].append(g0)
                    rows["oscillator_ratio"].append(oscillator_ratio)
                    rows["drag_lambda"].append(drag_lambda)
                    rows["t_gate"].append(total_time)
                    rows["max_high_population"].append(max_high)
                    rows["final_high_population"].append(final_high)
                    rows["F_X"].append(f_x)
                    rows["passes_leakage"].append(int(passes_leakage))
                    rows["passes_fidelity"].append(int(passes_fidelity))
                    rows["passes_both"].append(int(passes_both))
                    print(
                        f"[{count:03d}/{total}] v={v_low:.3f} r={coupling_ratio:.3f} "
                        f"ratio={oscillator_ratio:.1f} drag={drag_lambda:+.1f} "
                        f"P_high={max_high:.3e} F_X={f_x:.6f} pass={passes_both}"
                    )

    save_csv("drag_alpha3_midlow_scan.csv", rows)

    passes = np.asarray(rows["passes_both"], dtype=bool)
    f_x_arr = np.asarray(rows["F_X"])
    high_arr = np.asarray(rows["max_high_population"])
    print("\n[drag_scan] summary")
    print(f"  total points: {total}")
    print(f"  pass count: {int(np.sum(passes))}/{total}")

    for v_low in v_values:
        mask_v = np.asarray(rows["v_low"]) == v_low
        print(f"  v={v_low:.3f}: pass {int(np.sum(passes & mask_v))}/{int(np.sum(mask_v))}")

    best_idx = int(np.argmax(f_x_arr - 10 * np.maximum(high_arr - leakage_threshold, 0)))
    print(
        "  best combined point: "
        f"v={rows['v_low'][best_idx]:.3f}, "
        f"r={rows['g0_over_omega_rabi0'][best_idx]:.3f}, "
        f"ratio={rows['oscillator_ratio'][best_idx]:.1f}, "
        f"drag={rows['drag_lambda'][best_idx]:+.1f}, "
        f"P_high={rows['max_high_population'][best_idx]:.3e}, "
        f"F_X={rows['F_X'][best_idx]:.6f}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sc0 = axes[0].scatter(
        rows["drag_lambda"],
        rows["F_X"],
        c=rows["v_low"],
        s=30 + 400 * np.asarray(rows["g0_over_omega_rabi0"]),
        cmap="viridis",
    )
    axes[0].axhline(fidelity_threshold, color="tab:red", linestyle="--")
    axes[0].set_xlabel("DRAG lambda")
    axes[0].set_ylabel("F_X")
    axes[0].set_title("DRAG fidelity scan")
    axes[0].grid(True, alpha=0.3)
    fig.colorbar(sc0, ax=axes[0], label="v_low")

    sc1 = axes[1].scatter(
        rows["drag_lambda"],
        np.log10(high_arr),
        c=rows["oscillator_ratio"],
        s=30 + 400 * np.asarray(rows["g0_over_omega_rabi0"]),
        cmap="plasma",
    )
    axes[1].axhline(np.log10(leakage_threshold), color="tab:red", linestyle="--")
    axes[1].set_xlabel("DRAG lambda")
    axes[1].set_ylabel("log10(P_high,max)")
    axes[1].set_title("DRAG leakage scan")
    axes[1].grid(True, alpha=0.3)
    fig.colorbar(sc1, ax=axes[1], label="oscillator ratio")
    fig.suptitle("alpha=3 smooth X + DRAG scan; marker size = g0/Omega_R0")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "drag_alpha3_midlow_scan.png", dpi=180)


if __name__ == "__main__":
    main()
