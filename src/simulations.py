"""High-level simulation functions."""
from __future__ import annotations

import numpy as np

from .params import SimParams, PARAMS
from .schedules import omega_rabi, gate_time_x, sta_error_bound
from .noise import analytic_gate_fidelity, gamma_decay

try:
    import qutip as qt
except Exception:  # pragma: no cover
    qt = None


def rabi_scaling(v_values, params: SimParams = PARAMS):
    v_values = np.asarray(v_values, dtype=float)
    omega = omega_rabi(v_values, omega_rabi0=params.omega_rabi0, v0=params.v0)
    t_gate = gate_time_x(v_values, omega_rabi0=params.omega_rabi0, v0=params.v0)
    return {"v_low": v_values, "omega_rabi": omega, "t_gate": t_gate}


def sta_switch_error(t_values, params: SimParams = PARAMS):
    t_values = np.asarray(t_values, dtype=float)
    return {
        "switch_time": t_values,
        "error_bound": np.array([sta_error_bound(t, params.sta_control_error) for t in t_values]),
    }


def optimal_vlow(v_values, kind: str, params: SimParams = PARAMS):
    v_values = np.asarray(v_values, dtype=float)
    fidelity = analytic_gate_fidelity(v_values, kind=kind, params=params)
    decay = gamma_decay(v_values, params=params)
    idx = int(np.argmax(fidelity))
    return {
        "v_low": v_values,
        "fidelity": fidelity,
        "gamma_decay": decay,
        "best_v_low": float(v_values[idx]),
        "best_fidelity": float(fidelity[idx]),
    }


def topological_interferometer(n_values, ports: int = 8, phase_noise: float = 0.0, samples: int = 1, seed: int = 7):
    """DFT multi-port topological readout probabilities.

    Paths acquire phase exp(i 2*pi*n*k/M). The output distribution peaks at
    j = -n mod M. Optional Gaussian path phase noise can be averaged.
    """
    rng = np.random.default_rng(seed)
    n_values = np.asarray(n_values, dtype=int)
    k = np.arange(ports)
    j = np.arange(ports)
    dft = np.exp(2j * np.pi * np.outer(j, k) / ports) / np.sqrt(ports)
    rows = []
    for n in n_values:
        prob_acc = np.zeros(ports)
        for _ in range(samples):
            noise = rng.normal(0.0, phase_noise, size=ports) if phase_noise > 0 else 0.0
            path = np.exp(2j * np.pi * n * k / ports + 1j * noise) / np.sqrt(ports)
            amp = dft @ path
            prob_acc += np.abs(amp) ** 2
        prob = prob_acc / samples
        rows.append(prob / prob.sum())
    return np.array(rows)
