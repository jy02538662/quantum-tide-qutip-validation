"""Noise and decay models for programmable stiffness simulations."""
from __future__ import annotations

import numpy as np

from .params import SimParams, PARAMS
from .schedules import omega_br, gate_time_x


def gamma_decay(v_low: float | np.ndarray, params: SimParams = PARAMS):
    """Three-wave-mixing decay model.

    Gamma_decay = Gamma0 * v_low^-7 when the decay channel is open.
    In the user's paper, the channel closes for v_low > v_gap.
    """
    v = np.asarray(v_low, dtype=float)
    open_channel = v < params.v_gap
    rate = params.decay0 * np.power(np.maximum(v, 1e-9), -7)
    return np.where(open_channel, rate, 0.0)


def noise_spectrum(v_low: float | np.ndarray, kind: str, params: SimParams = PARAMS):
    w = omega_br(v_low, omega0=params.omega0, v0=params.v0)
    if kind == "white":
        return np.ones_like(np.asarray(w), dtype=float)
    if kind in {"1/f", "one_over_f"}:
        return params.omega0 / np.maximum(w, 1e-9)
    raise ValueError(f"unknown noise kind: {kind}")


def gamma_phi(v_low: float | np.ndarray, kind: str = "white", params: SimParams = PARAMS):
    return params.dephasing0 * noise_spectrum(v_low, kind=kind, params=params)


def analytic_gate_fidelity(v_low: float | np.ndarray, kind: str = "white", params: SimParams = PARAMS):
    t_gate = gate_time_x(v_low, omega_rabi0=params.omega_rabi0, v0=params.v0)
    loss_phi = gamma_phi(v_low, kind=kind, params=params) * t_gate
    loss_decay = gamma_decay(v_low, params=params) * t_gate
    return np.clip(1.0 - loss_phi - loss_decay, 0.0, 1.0)
