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
    coupling_ratio: float,
    oscillator_ratio: float,
    samples: int = 140,
):
    g0 = coupling_ratio * PARAMS.omega_rabi0
    h = x_gate_full_space_hamiltonian(
        v_low,
        leakage_coupling=g0,
        oscillator_ratio=oscillator_ratio,
        leakage_alpha=3.0,
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
    f_x = abs(target.overlap(result.states[-1])) ** 2
    return g0, float(np.max(high)), float(high[-1]), float(f_x), t_gate


def main():
    ensure_outputs()
    try:
        import qutip as qt
    except Exception as exc:
        print(f"[g0_ratio_scan] skipped: QuTiP unavailable ({exc})")
        return

    v_values = np.array([0.10, 0.14, 0.18, 0.24, 0.32, 0.42, 0.56])
    ratio_values = np.array([2.4, 3.0, 3.6, 4.5, 5.5, 7.0])
    coupling_ratios = np.array([0.010, 0.015, 0.020, 0.030, 0.040, 0.050])
    leakage_threshold = 1e-3
    fidelity_threshold = 0.999

    rows = {
        "alpha": [],
        "g0_over_omega_rabi0": [],
        "g0": [],
        "v_low": [],
        "oscillator_ratio": [],
        "max_high_population": [],
        "final_high_population": [],
        "F_X": [],
        "t_gate": [],
        "passes_leakage": [],
        "passes_fidelity": [],
        "passes_both": [],
    }

    total = len(v_values) * len(ratio_values) * len(coupling_ratios)
    count = 0
    for coupling_ratio in coupling_ratios:
        for v_low in v_values:
            for oscillator_ratio in ratio_values:
                count += 1
                g0, max_high, final_high, f_x, t_gate = run_point(
                    qt,
                    v_low,
                    coupling_ratio,
                    oscillator_ratio,
                )
                passes_leakage = max_high < leakage_threshold
                passes_fidelity = f_x > fidelity_threshold
                passes_both = passes_leakage and passes_fidelity
                rows["alpha"].append(3.0)
                rows["g0_over_omega_rabi0"].append(coupling_ratio)
                rows["g0"].append(g0)
                rows["v_low"].append(v_low)
                rows["oscillator_ratio"].append(oscillator_ratio)
                rows["max_high_population"].append(max_high)
                rows["final_high_population"].append(final_high)
                rows["F_X"].append(f_x)
                rows["t_gate"].append(t_gate)
                rows["passes_leakage"].append(int(passes_leakage))
                rows["passes_fidelity"].append(int(passes_fidelity))
                rows["passes_both"].append(int(passes_both))
                print(
                    f"[{count:03d}/{total}] r={coupling_ratio:.3f} v={v_low:.3f} "
                    f"ratio={oscillator_ratio:.1f} g0={g0:.4e} "
                    f"max_high={max_high:.3e} F_X={f_x:.6f} pass={passes_both}"
                )

    save_csv("g0_over_omega_rabi0_alpha3_scan.csv", rows)

    r_arr = np.asarray(rows["g0_over_omega_rabi0"])
    v_arr = np.asarray(rows["v_low"])
    osc_arr = np.asarray(rows["oscillator_ratio"])
    max_high = np.asarray(rows["max_high_population"])
    f_x = np.asarray(rows["F_X"])
    passes = np.asarray(rows["passes_both"], dtype=bool)

    print("\n[g0_ratio_scan] summary")
    print(f"  total points: {total}")
    print(f"  pass count: {int(np.sum(passes))}/{total}")
    for coupling_ratio in coupling_ratios:
        mask = r_arr == coupling_ratio
        print(f"  r={coupling_ratio:.3f}: pass {int(np.sum(passes & mask))}/{int(np.sum(mask))}")

    if np.any(passes):
        pass_indices = np.where(passes)[0]
        best_idx = pass_indices[np.argmax(f_x[passes] - max_high[passes])]
        print(
            "  best hard-pass point: "
            f"r={r_arr[best_idx]:.3f}, "
            f"v={v_arr[best_idx]:.3f}, "
            f"ratio={osc_arr[best_idx]:.1f}, "
            f"max_high={max_high[best_idx]:.3e}, "
            f"F_X={f_x[best_idx]:.6f}, "
            f"t_gate={rows['t_gate'][best_idx]:.3f}"
        )
    else:
        best_idx = int(np.argmax(f_x - 10 * np.maximum(max_high - leakage_threshold, 0)))
        print(
            "  no hard-pass; best near-pass: "
            f"r={r_arr[best_idx]:.3f}, "
            f"v={v_arr[best_idx]:.3f}, "
            f"ratio={osc_arr[best_idx]:.1f}, "
            f"max_high={max_high[best_idx]:.3e}, "
            f"F_X={f_x[best_idx]:.6f}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sc0 = axes[0].scatter(v_arr, r_arr, c=np.log10(max_high), s=25 * osc_arr, cmap="viridis_r")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("v_low / v0")
    axes[0].set_ylabel("g0 / Omega_R0")
    axes[0].set_title("Leakage under alpha=3")
    axes[0].grid(True, which="both", alpha=0.3)
    fig.colorbar(sc0, ax=axes[0], label="log10(P_high,max)")

    sc1 = axes[1].scatter(v_arr, r_arr, c=f_x, s=25 * osc_arr, cmap="plasma")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("v_low / v0")
    axes[1].set_ylabel("g0 / Omega_R0")
    axes[1].set_title("X-gate fidelity under alpha=3")
    axes[1].grid(True, which="both", alpha=0.3)
    fig.colorbar(sc1, ax=axes[1], label="F_X")
    fig.suptitle("Physical coupling-ratio scan; marker size = oscillator ratio")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "g0_over_omega_rabi0_alpha3_scan.png", dpi=180)


if __name__ == "__main__":
    main()
