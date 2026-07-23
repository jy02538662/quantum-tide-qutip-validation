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


def qubit_state(qt, label: str):
    g = qt.basis(2, 0)
    e = qt.basis(2, 1)
    states = {
        "g": g,
        "e": e,
        "plus": (g + e).unit(),
        "minus": (g - e).unit(),
        "plus_i": (g + 1j * e).unit(),
        "minus_i": (g - 1j * e).unit(),
    }
    return states[label]


def ideal_x_target(qt, label: str):
    return (qt.sigmax() * qubit_state(qt, label)).unit()


def gamma_phi_from_ratio(v_low: float, ratio: float, kind: str):
    phi0 = ratio * PARAMS.omega_rabi0
    if kind == "white":
        return phi0
    if kind == "1/f":
        return phi0 / max(v_low, 1e-12)
    raise ValueError(f"unknown noise kind: {kind}")


def build_c_ops(qt, v_low: float, phi_ratio: float, noise_kind: str):
    n = PARAMS.oscillator_levels
    io = qt.qeye(n)
    iq = qt.qeye(2)
    a = qt.destroy(n)
    sz = qt.sigmaz()
    phi = gamma_phi_from_ratio(v_low, phi_ratio, noise_kind)
    decay = float(gamma_decay(v_low, params=PARAMS))
    c_ops = []
    if phi > 0:
        c_ops.append(np.sqrt(phi) * qt.tensor(sz, io))
    if decay > 0:
        c_ops.append(np.sqrt(decay) * qt.tensor(iq, a))
    return c_ops, phi, decay


def run_input(qt, point: dict, phi_ratio: float, noise_kind: str, input_label: str):
    v_low = point["v_low"]
    g0 = point["g0_over_omega_rabi0"] * PARAMS.omega_rabi0
    total_time = float(gate_time_x(v_low, omega_rabi0=PARAMS.omega_rabi0, v0=PARAMS.v0))
    h = x_gate_drag_hamiltonian(
        v_low,
        total_time,
        leakage_coupling=g0,
        oscillator_ratio=point["oscillator_ratio"],
        leakage_alpha=3.0,
        drag_lambda=0.0,
        qubit_detuning=0.0,
    )
    osc0 = qt.basis(PARAMS.oscillator_levels, 0)
    psi0 = qt.tensor(qubit_state(qt, input_label), osc0)
    target = qt.tensor(ideal_x_target(qt, input_label), osc0)
    _, high_projector = oscillator_projectors()
    c_ops, phi, decay = build_c_ops(qt, v_low, phi_ratio, noise_kind)
    times = np.linspace(0.0, total_time, 180)
    result = qt.mesolve(
        h,
        psi0.proj(),
        times,
        c_ops=c_ops,
        e_ops=[high_projector],
        options={"store_states": True},
    )
    final_rho = result.states[-1]
    fidelity = float((target.dag() * final_rho * target).real)
    high = np.asarray(result.expect[0])
    return fidelity, float(np.max(high)), phi, decay, total_time


def main():
    ensure_outputs()
    try:
        import qutip as qt
    except Exception as exc:
        print(f"[gamma_ratio_scan] skipped: QuTiP unavailable ({exc})")
        return

    points = [
        {"label": "r002_v024_ratio3", "v_low": 0.240, "g0_over_omega_rabi0": 0.020, "oscillator_ratio": 3.0},
        {"label": "r003_v024_ratio45", "v_low": 0.240, "g0_over_omega_rabi0": 0.030, "oscillator_ratio": 4.5},
        {"label": "r004_v024_ratio55", "v_low": 0.240, "g0_over_omega_rabi0": 0.040, "oscillator_ratio": 5.5},
        {"label": "r005_v024_ratio7", "v_low": 0.240, "g0_over_omega_rabi0": 0.050, "oscillator_ratio": 7.0},
    ]
    phi_ratios = np.array([1.0e-4, 3.0e-4, 1.0e-3, 2.5e-3, 1.0e-2])
    noise_kinds = ["white", "1/f"]
    input_labels = ["g", "e", "plus", "minus", "plus_i", "minus_i"]
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
        "gamma_phi_effective": [],
        "gamma_decay": [],
        "F_g": [],
        "mean_six_state_fidelity": [],
        "min_six_state_fidelity": [],
        "max_high_population": [],
        "passes_leakage": [],
        "passes_state_transfer": [],
        "passes_six_state_proxy": [],
    }

    total = len(points) * len(phi_ratios) * len(noise_kinds)
    count = 0
    for point in points:
        for phi_ratio in phi_ratios:
            for noise_kind in noise_kinds:
                count += 1
                fidelities = []
                highs = []
                phi_effective = 0.0
                decay = 0.0
                t_gate = 0.0
                for input_label in input_labels:
                    fidelity, max_high, phi_effective, decay, t_gate = run_input(
                        qt,
                        point,
                        phi_ratio,
                        noise_kind,
                        input_label,
                    )
                    fidelities.append(fidelity)
                    highs.append(max_high)
                f_g = fidelities[0]
                mean_f = float(np.mean(fidelities))
                min_f = float(np.min(fidelities))
                max_high_all = float(np.max(highs))
                pass_leak = max_high_all < leakage_threshold
                pass_g = f_g > fidelity_threshold
                pass_proxy = mean_f > fidelity_threshold and min_f > 0.995

                rows["point_label"].append(point["label"])
                rows["noise_kind"].append(noise_kind)
                rows["gamma_phi0_over_omega_rabi0"].append(phi_ratio)
                rows["v_low"].append(point["v_low"])
                rows["g0_over_omega_rabi0"].append(point["g0_over_omega_rabi0"])
                rows["oscillator_ratio"].append(point["oscillator_ratio"])
                rows["t_gate"].append(t_gate)
                rows["gamma_phi_effective"].append(phi_effective)
                rows["gamma_decay"].append(decay)
                rows["F_g"].append(f_g)
                rows["mean_six_state_fidelity"].append(mean_f)
                rows["min_six_state_fidelity"].append(min_f)
                rows["max_high_population"].append(max_high_all)
                rows["passes_leakage"].append(int(pass_leak))
                rows["passes_state_transfer"].append(int(pass_g))
                rows["passes_six_state_proxy"].append(int(pass_proxy))

                print(
                    f"[{count:02d}/{total}] {point['label']} {noise_kind} "
                    f"gamma/Omega={phi_ratio:.1e} P_high={max_high_all:.3e} "
                    f"F_g={f_g:.6f} F_mean={mean_f:.6f} pass={pass_proxy}"
                )

    save_csv("gamma_phi_ratio_lindblad_scan.csv", rows)

    print("\n[gamma_ratio_scan] summary")
    print("point              noise gamma/Omega  P_high     F_g       F_mean   F_min    pass")
    for i, point_label in enumerate(rows["point_label"]):
        print(
            f"{point_label:18s} {rows['noise_kind'][i]:5s} "
            f"{rows['gamma_phi0_over_omega_rabi0'][i]:.1e}  "
            f"{rows['max_high_population'][i]:.3e}  "
            f"{rows['F_g'][i]:.6f}  "
            f"{rows['mean_six_state_fidelity'][i]:.6f}  "
            f"{rows['min_six_state_fidelity'][i]:.6f}  "
            f"{rows['passes_six_state_proxy'][i]}"
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    for point in points:
        for noise_kind in noise_kinds:
            mask = [
                p == point["label"] and n == noise_kind
                for p, n in zip(rows["point_label"], rows["noise_kind"])
            ]
            x = np.asarray(rows["gamma_phi0_over_omega_rabi0"])[mask]
            y = np.asarray(rows["mean_six_state_fidelity"])[mask]
            ax.semilogx(x, y, marker="o", label=f"{point['label']} {noise_kind}")
    ax.axhline(fidelity_threshold, color="tab:red", linestyle="--", label="0.999 target")
    ax.set_xlabel("gamma_phi0 / Omega_R0")
    ax.set_ylabel("six-state mean fidelity")
    ax.set_title("Lindblad fidelity threshold vs dephasing normalization")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gamma_phi_ratio_lindblad_scan.png", dpi=180)


if __name__ == "__main__":
    main()
