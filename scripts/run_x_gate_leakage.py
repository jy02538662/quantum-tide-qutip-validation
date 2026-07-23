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


def main():
    ensure_outputs()
    try:
        import qutip as qt
    except Exception as exc:
        print(f"[x_gate_leakage] skipped: QuTiP unavailable ({exc})")
        return

    v_low = PARAMS.v_low_default
    h = x_gate_full_space_hamiltonian(v_low)
    psi0 = x_gate_initial_state()
    projectors, high_projector = oscillator_projectors()
    t_gate = float(gate_time_x(v_low, omega_rabi0=PARAMS.omega_rabi0, v0=PARAMS.v0))
    times = np.linspace(0.0, t_gate, 220)
    e_ops = projectors + [high_projector]
    result = qt.mesolve(h, psi0, times, c_ops=[], e_ops=e_ops)

    rows = {"time": times}
    for i in range(PARAMS.oscillator_levels):
        rows[f"P_osc_{i}"] = np.asarray(result.expect[i])
    rows["P_high_ge2"] = np.asarray(result.expect[-1])
    save_csv("x_gate_leakage.csv", rows)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(times, rows["P_osc_0"], label="osc |0>")
    ax.plot(times, rows["P_osc_1"], label="osc |1>")
    ax.plot(times, rows["P_high_ge2"], label="high levels >=2")
    ax.axhline(1e-3, color="tab:red", linestyle="--", label="1e-3 target")
    ax.set_xlabel("time")
    ax.set_ylabel("population")
    ax.set_title("Full-space X-gate leakage stress test")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "x_gate_leakage.png", dpi=180)
    print(f"[x_gate_leakage] max high population={np.max(rows['P_high_ge2']):.3e}")


if __name__ == "__main__":
    main()
