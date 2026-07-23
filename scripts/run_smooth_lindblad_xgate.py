from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.hamiltonians import oscillator_projectors, x_gate_drag_hamiltonian
from src.metrics import FIG_DIR, ensure_outputs, save_csv
from src.noise import gamma_decay, gamma_phi
from src.params import PARAMS
from src.schedules import gate_time_x


def qubit_state(qt, label: str):
    g = qt.basis(2, 0)
    e = qt.basis(2, 1)
    if label == "g":
        return g
    if label == "e":
        return e
    if label == "plus":
        return (g + e).unit()
    if label == "minus":
        return (g - e).unit()
    if label == "plus_i":
        return (g + 1j * e).unit()
    if label == "minus_i":
        return (g - 1j * e).unit()
    raise ValueError(f"unknown state label: {label}")


def ideal_x_target(qt, label: str):
    sx = qt.sigmax()
    return (sx * qubit_state(qt, label)).unit()


def build_c_ops(qt, v_low: float, noise_kind: str):
    n = PARAMS.oscillator_levels
    io = qt.qeye(n)
    iq = qt.qeye(2)
    a = qt.destroy(n)
    sz = qt.sigmaz()
    c_ops = []

    if noise_kind == "none":
        return c_ops

    phi = float(gamma_phi(v_low, kind=noise_kind, params=PARAMS))
    decay = float(gamma_decay(v_low, params=PARAMS))

    if phi > 0:
        c_ops.append(np.sqrt(phi) * qt.tensor(sz, io))
    if decay > 0:
        c_ops.append(np.sqrt(decay) * qt.tensor(iq, a))
    return c_ops


def run_input_state(qt, point: dict, noise_kind: str, input_label: str):
    v_low = point["v_low"]
    coupling_ratio = point["g0_over_omega_rabi0"]
    oscillator_ratio = point["oscillator_ratio"]
    g0 = coupling_ratio * PARAMS.omega_rabi0
    total_time = float(gate_time_x(v_low, omega_rabi0=PARAMS.omega_rabi0, v0=PARAMS.v0))
    h = x_gate_drag_hamiltonian(
        v_low,
        total_time,
        leakage_coupling=g0,
        oscillator_ratio=oscillator_ratio,
        leakage_alpha=3.0,
        drag_lambda=0.0,
        qubit_detuning=0.0,
    )

    osc0 = qt.basis(PARAMS.oscillator_levels, 0)
    psi0 = qt.tensor(qubit_state(qt, input_label), osc0)
    target = qt.tensor(ideal_x_target(qt, input_label), osc0)
    _, high_projector = oscillator_projectors()
    c_ops = build_c_ops(qt, v_low, noise_kind)
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
    return {
        "state_fidelity": fidelity,
        "max_high_population": float(np.max(high)),
        "final_high_population": float(high[-1]),
        "t_gate": total_time,
        "gamma_phi": 0.0 if noise_kind == "none" else float(gamma_phi(v_low, kind=noise_kind, params=PARAMS)),
        "gamma_decay": 0.0 if noise_kind == "none" else float(gamma_decay(v_low, params=PARAMS)),
    }


def main():
    ensure_outputs()
    try:
        import qutip as qt
    except Exception as exc:
        print(f"[smooth_lindblad] skipped: QuTiP unavailable ({exc})")
        return

    operating_points = [
        {"label": "fast_r002_ratio3", "v_low": 0.240, "g0_over_omega_rabi0": 0.020, "oscillator_ratio": 3.0},
        {"label": "fast_r003_ratio45", "v_low": 0.240, "g0_over_omega_rabi0": 0.030, "oscillator_ratio": 4.5},
        {"label": "fast_r004_ratio55", "v_low": 0.240, "g0_over_omega_rabi0": 0.040, "oscillator_ratio": 5.5},
        {"label": "fast_r005_ratio7", "v_low": 0.240, "g0_over_omega_rabi0": 0.050, "oscillator_ratio": 7.0},
        {"label": "near_fail_v018", "v_low": 0.180, "g0_over_omega_rabi0": 0.020, "oscillator_ratio": 7.0},
    ]
    input_labels = ["g", "e", "plus", "minus", "plus_i", "minus_i"]
    noise_kinds = ["none", "white", "1/f"]
    leakage_threshold = 1e-3
    fidelity_threshold = 0.999

    detail_rows = {
        "point_label": [],
        "noise_kind": [],
        "input_state": [],
        "v_low": [],
        "g0_over_omega_rabi0": [],
        "oscillator_ratio": [],
        "t_gate": [],
        "state_fidelity": [],
        "max_high_population": [],
        "final_high_population": [],
        "gamma_phi": [],
        "gamma_decay": [],
    }
    summary_rows = {
        "point_label": [],
        "noise_kind": [],
        "v_low": [],
        "g0_over_omega_rabi0": [],
        "oscillator_ratio": [],
        "t_gate": [],
        "F_X_from_g": [],
        "mean_six_state_fidelity": [],
        "min_six_state_fidelity": [],
        "max_high_population": [],
        "gamma_phi": [],
        "gamma_decay": [],
        "passes_leakage": [],
        "passes_state_transfer": [],
        "passes_six_state_proxy": [],
    }

    total = len(operating_points) * len(noise_kinds) * len(input_labels)
    count = 0
    for point in operating_points:
        for noise_kind in noise_kinds:
            fidelities = []
            high_values = []
            last_result = None
            for input_label in input_labels:
                count += 1
                result = run_input_state(qt, point, noise_kind, input_label)
                last_result = result
                fidelities.append(result["state_fidelity"])
                high_values.append(result["max_high_population"])
                for key in ["point_label", "noise_kind", "input_state"]:
                    if key == "point_label":
                        detail_rows[key].append(point["label"])
                    elif key == "noise_kind":
                        detail_rows[key].append(noise_kind)
                    else:
                        detail_rows[key].append(input_label)
                detail_rows["v_low"].append(point["v_low"])
                detail_rows["g0_over_omega_rabi0"].append(point["g0_over_omega_rabi0"])
                detail_rows["oscillator_ratio"].append(point["oscillator_ratio"])
                detail_rows["t_gate"].append(result["t_gate"])
                detail_rows["state_fidelity"].append(result["state_fidelity"])
                detail_rows["max_high_population"].append(result["max_high_population"])
                detail_rows["final_high_population"].append(result["final_high_population"])
                detail_rows["gamma_phi"].append(result["gamma_phi"])
                detail_rows["gamma_decay"].append(result["gamma_decay"])
                print(
                    f"[{count:03d}/{total}] {point['label']} noise={noise_kind} input={input_label} "
                    f"F={result['state_fidelity']:.6f} P_high={result['max_high_population']:.3e}"
                )

            f_x_from_g = fidelities[0]
            mean_f = float(np.mean(fidelities))
            min_f = float(np.min(fidelities))
            max_high = float(np.max(high_values))
            summary_rows["point_label"].append(point["label"])
            summary_rows["noise_kind"].append(noise_kind)
            summary_rows["v_low"].append(point["v_low"])
            summary_rows["g0_over_omega_rabi0"].append(point["g0_over_omega_rabi0"])
            summary_rows["oscillator_ratio"].append(point["oscillator_ratio"])
            summary_rows["t_gate"].append(last_result["t_gate"])
            summary_rows["F_X_from_g"].append(f_x_from_g)
            summary_rows["mean_six_state_fidelity"].append(mean_f)
            summary_rows["min_six_state_fidelity"].append(min_f)
            summary_rows["max_high_population"].append(max_high)
            summary_rows["gamma_phi"].append(last_result["gamma_phi"])
            summary_rows["gamma_decay"].append(last_result["gamma_decay"])
            summary_rows["passes_leakage"].append(int(max_high < leakage_threshold))
            summary_rows["passes_state_transfer"].append(int(f_x_from_g > fidelity_threshold))
            summary_rows["passes_six_state_proxy"].append(int(mean_f > fidelity_threshold and min_f > 0.995))

    save_csv("smooth_lindblad_state_detail.csv", detail_rows)
    save_csv("smooth_lindblad_summary.csv", summary_rows)

    print("\n[smooth_lindblad] summary")
    print("point                noise   P_high      F_g       F_mean    F_min     pass_leak pass_g pass_proxy")
    for i, label in enumerate(summary_rows["point_label"]):
        print(
            f"{label:20s} {summary_rows['noise_kind'][i]:6s} "
            f"{summary_rows['max_high_population'][i]:.3e}  "
            f"{summary_rows['F_X_from_g'][i]:.6f}  "
            f"{summary_rows['mean_six_state_fidelity'][i]:.6f}  "
            f"{summary_rows['min_six_state_fidelity'][i]:.6f}  "
            f"{summary_rows['passes_leakage'][i]}         "
            f"{summary_rows['passes_state_transfer'][i]}      "
            f"{summary_rows['passes_six_state_proxy'][i]}"
        )

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(summary_rows["point_label"]))
    ax.plot(x, summary_rows["mean_six_state_fidelity"], marker="o", label="six-state mean fidelity")
    ax.plot(x, summary_rows["F_X_from_g"], marker="s", label="|g> -> |e> fidelity")
    ax.axhline(fidelity_threshold, color="tab:red", linestyle="--", label="0.999 target")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{p}\n{n}" for p, n in zip(summary_rows["point_label"], summary_rows["noise_kind"])],
        rotation=60,
        ha="right",
    )
    ax.set_ylabel("fidelity")
    ax.set_title("Smooth-pulse Lindblad fidelity proxy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "smooth_lindblad_fidelity_summary.png", dpi=180)


if __name__ == "__main__":
    main()
