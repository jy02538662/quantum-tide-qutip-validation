from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.hamiltonians import oscillator_projectors, x_gate_drag_hamiltonian
from src.metrics import FIG_DIR, ensure_outputs, save_csv
from src.noise import gamma_decay
from src.params import PARAMS
from src.schedules import gate_time_x


def gamma_phi_from_ratio(v_low: float, ratio: float, kind: str):
    if kind == "none":
        return 0.0
    phi0 = ratio * PARAMS.omega_rabi0
    if kind == "white":
        return phi0
    if kind == "1/f":
        return phi0 / max(v_low, 1e-12)
    raise ValueError(f"unknown noise kind: {kind}")


def extend_hamiltonian(qt, h_sys, ref_dim: int = 2):
    iref = qt.qeye(ref_dim)
    extended = []
    for item in h_sys:
        if isinstance(item, list):
            extended.append([qt.tensor(iref, item[0]), item[1]])
        else:
            extended.append(qt.tensor(iref, item))
    return extended


def build_extended_c_ops(qt, v_low: float, phi_ratio: float, noise_kind: str):
    n = PARAMS.oscillator_levels
    iref = qt.qeye(2)
    io = qt.qeye(n)
    iq = qt.qeye(2)
    a = qt.destroy(n)
    sz = qt.sigmaz()
    phi = gamma_phi_from_ratio(v_low, phi_ratio, noise_kind)
    decay = float(gamma_decay(v_low, params=PARAMS))
    c_ops = []
    if phi > 0:
        c_ops.append(np.sqrt(phi) * qt.tensor(iref, qt.tensor(sz, io)))
    if decay > 0:
        c_ops.append(np.sqrt(decay) * qt.tensor(iref, qt.tensor(iq, a)))
    return c_ops, phi, decay


def bell_state(qt):
    g = qt.basis(2, 0)
    e = qt.basis(2, 1)
    return (qt.tensor(g, g) + qt.tensor(e, e)).unit()


def run_process_point(qt, point: dict, noise_kind: str, phi_ratio: float):
    v_low = point["v_low"]
    g0 = point["g0_over_omega_rabi0"] * PARAMS.omega_rabi0
    total_time = float(gate_time_x(v_low, omega_rabi0=PARAMS.omega_rabi0, v0=PARAMS.v0))
    h_sys = x_gate_drag_hamiltonian(
        v_low,
        total_time,
        leakage_coupling=g0,
        oscillator_ratio=point["oscillator_ratio"],
        leakage_alpha=3.0,
        drag_lambda=0.0,
        qubit_detuning=0.0,
    )
    h_ext = extend_hamiltonian(qt, h_sys)
    c_ops, phi, decay = build_extended_c_ops(qt, v_low, phi_ratio, noise_kind)

    osc0 = qt.basis(PARAMS.oscillator_levels, 0)
    bell = bell_state(qt)
    initial = qt.tensor(bell, osc0)
    ideal_bell = qt.tensor(qt.qeye(2), qt.sigmax()) * bell
    target = qt.tensor(ideal_bell, osc0)
    _, high_sys = oscillator_projectors()
    high_ext = qt.tensor(qt.qeye(2), high_sys)
    times = np.linspace(0.0, total_time, 200)
    result = qt.mesolve(
        h_ext,
        initial.proj(),
        times,
        c_ops=c_ops,
        e_ops=[high_ext],
        options={"store_states": True},
    )
    final_rho = result.states[-1]
    entanglement_fidelity = float((target.dag() * final_rho * target).real)
    average_gate_fidelity = (2.0 * entanglement_fidelity + 1.0) / 3.0
    high = np.asarray(result.expect[0])
    return {
        "F_entanglement": entanglement_fidelity,
        "F_avg": average_gate_fidelity,
        "max_high_population": float(np.max(high)),
        "final_high_population": float(high[-1]),
        "t_gate": total_time,
        "gamma_phi_effective": phi,
        "gamma_decay": decay,
    }


def main():
    ensure_outputs()
    try:
        import qutip as qt
    except Exception as exc:
        print(f"[process_fidelity] skipped: QuTiP unavailable ({exc})")
        return

    points = [
        {"label": "r002_v024_ratio3", "v_low": 0.240, "g0_over_omega_rabi0": 0.020, "oscillator_ratio": 3.0},
        {"label": "r003_v024_ratio45", "v_low": 0.240, "g0_over_omega_rabi0": 0.030, "oscillator_ratio": 4.5},
        {"label": "r004_v024_ratio55", "v_low": 0.240, "g0_over_omega_rabi0": 0.040, "oscillator_ratio": 5.5},
        {"label": "r005_v024_ratio7", "v_low": 0.240, "g0_over_omega_rabi0": 0.050, "oscillator_ratio": 7.0},
    ]
    noise_cases = [
        ("none", 0.0),
        ("white", 1.0e-4),
        ("white", 2.5e-3),
        ("1/f", 1.0e-4),
        ("1/f", 2.5e-3),
    ]
    fidelity_threshold = 0.999
    leakage_threshold = 1e-3

    rows = {
        "point_label": [],
        "noise_kind": [],
        "gamma_phi0_over_omega_rabi0": [],
        "v_low": [],
        "g0_over_omega_rabi0": [],
        "oscillator_ratio": [],
        "t_gate": [],
        "F_entanglement": [],
        "F_avg": [],
        "max_high_population": [],
        "final_high_population": [],
        "gamma_phi_effective": [],
        "gamma_decay": [],
        "passes_average_gate_fidelity": [],
        "passes_leakage": [],
        "passes_both": [],
    }

    total = len(points) * len(noise_cases)
    count = 0
    for point in points:
        for noise_kind, phi_ratio in noise_cases:
            count += 1
            result = run_process_point(qt, point, noise_kind, phi_ratio)
            pass_f = result["F_avg"] > fidelity_threshold
            pass_l = result["max_high_population"] < leakage_threshold
            rows["point_label"].append(point["label"])
            rows["noise_kind"].append(noise_kind)
            rows["gamma_phi0_over_omega_rabi0"].append(phi_ratio)
            rows["v_low"].append(point["v_low"])
            rows["g0_over_omega_rabi0"].append(point["g0_over_omega_rabi0"])
            rows["oscillator_ratio"].append(point["oscillator_ratio"])
            rows["t_gate"].append(result["t_gate"])
            rows["F_entanglement"].append(result["F_entanglement"])
            rows["F_avg"].append(result["F_avg"])
            rows["max_high_population"].append(result["max_high_population"])
            rows["final_high_population"].append(result["final_high_population"])
            rows["gamma_phi_effective"].append(result["gamma_phi_effective"])
            rows["gamma_decay"].append(result["gamma_decay"])
            rows["passes_average_gate_fidelity"].append(int(pass_f))
            rows["passes_leakage"].append(int(pass_l))
            rows["passes_both"].append(int(pass_f and pass_l))
            print(
                f"[{count:02d}/{total}] {point['label']} {noise_kind} gamma/Omega={phi_ratio:.1e} "
                f"F_e={result['F_entanglement']:.6f} F_avg={result['F_avg']:.6f} "
                f"P_high={result['max_high_population']:.3e} pass={pass_f and pass_l}"
            )

    save_csv("standard_process_fidelity_xgate.csv", rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(rows["point_label"]))
    ax.plot(x, rows["F_avg"], marker="o")
    ax.axhline(fidelity_threshold, color="tab:red", linestyle="--", label="0.999 target")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{p}\n{n}:{g:.1e}" for p, n, g in zip(rows["point_label"], rows["noise_kind"], rows["gamma_phi0_over_omega_rabi0"])],
        rotation=65,
        ha="right",
    )
    ax.set_ylabel("Average gate fidelity")
    ax.set_title("Standard X-gate average fidelity via entanglement fidelity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "standard_process_fidelity_xgate.png", dpi=180)


if __name__ == "__main__":
    main()
