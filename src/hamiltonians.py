"""Hamiltonian builders for Quantum Tide vortex simulations."""
from __future__ import annotations

import numpy as np

try:
    import qutip as qt
except Exception:  # pragma: no cover
    qt = None

from .params import SimParams, PARAMS
from .schedules import omega_br, omega_rabi, theta_dot_sta


def sin2_envelope(t: float, total_time: float) -> float:
    if t <= 0.0 or t >= total_time:
        return 0.0
    return float(np.sin(np.pi * t / total_time) ** 2)


def sin2_envelope_derivative(t: float, total_time: float) -> float:
    if t <= 0.0 or t >= total_time:
        return 0.0
    return float((np.pi / total_time) * np.sin(2 * np.pi * t / total_time))


def require_qutip():
    if qt is None:
        raise RuntimeError("QuTiP is not installed. Run: pip install -r requirements.txt")


def x_gate_full_space_hamiltonian(
    v_low: float,
    params: SimParams = PARAMS,
    leakage_coupling: float | None = None,
    oscillator_ratio: float = 2.4,
    omega_rabi0: float | None = None,
    leakage_alpha: float = 0.5,
    qubit_detuning: float | None = None,
):
    """Build a qubit + oscillator Hamiltonian for X-gate leakage tests.

    The model intentionally keeps oscillator levels instead of eliminating them:

    H = 0.5 * omega_br(v_low) * sigma_z
      + omega_osc * a^dag a
      + 0.5 * Omega_R(v_low) * sigma_x
      + g * sigma_x (a + a^dag)

    `leakage_coupling`, `oscillator_ratio`, `omega_rabi0`, `leakage_alpha`, and
    `qubit_detuning` are exposed for engineering sweeps. Defaults reproduce the
    baseline run and must not be silently changed to make results look better.
    """
    require_qutip()
    n = params.oscillator_levels
    sx = qt.sigmax()
    sz = qt.sigmaz()
    iq = qt.qeye(2)
    a = qt.destroy(n)
    io = qt.qeye(n)

    coupling = params.leakage_coupling if leakage_coupling is None else leakage_coupling
    rabi0 = params.omega_rabi0 if omega_rabi0 is None else omega_rabi0
    wb = float(omega_br(v_low, omega0=params.omega0, v0=params.v0))
    wr = float(omega_rabi(v_low, omega_rabi0=rabi0, v0=params.v0))
    wosc = oscillator_ratio * wb
    detuning = wb if qubit_detuning is None else qubit_detuning
    g = coupling * (params.v0 / v_low) ** leakage_alpha

    h_qubit = 0.5 * detuning * qt.tensor(sz, io)
    h_drive = 0.5 * wr * qt.tensor(sx, io)
    h_osc = wosc * qt.tensor(iq, a.dag() * a)
    h_coup = g * qt.tensor(sx, a + a.dag())
    return h_qubit + h_drive + h_osc + h_coup


def x_gate_drag_hamiltonian(
    v_low: float,
    total_time: float,
    params: SimParams = PARAMS,
    leakage_coupling: float | None = None,
    oscillator_ratio: float = 2.4,
    leakage_alpha: float = 3.0,
    drag_lambda: float = 0.0,
    qubit_detuning: float = 0.0,
):
    """Time-dependent smooth X gate with a DRAG quadrature.

    The pulse area is normalized so that integral Omega_x(t) dt = pi for
    `sin^2(pi t / T)`. The leakage coupling follows the same microwave envelope,
    while the DRAG quadrature is proportional to dOmega_x/dt divided by the
    leakage detuning scale.
    """
    require_qutip()
    n = params.oscillator_levels
    sx = qt.sigmax()
    sy = qt.sigmay()
    sz = qt.sigmaz()
    iq = qt.qeye(2)
    a = qt.destroy(n)
    io = qt.qeye(n)

    coupling = params.leakage_coupling if leakage_coupling is None else leakage_coupling
    wb = float(omega_br(v_low, omega0=params.omega0, v0=params.v0))
    wosc = oscillator_ratio * wb
    g = coupling * (params.v0 / v_low) ** leakage_alpha
    omega_peak = 2.0 * np.pi / total_time
    leakage_detuning = max(abs(wosc - wb), 1e-9)

    h_static = 0.5 * qubit_detuning * qt.tensor(sz, io) + wosc * qt.tensor(iq, a.dag() * a)
    h_x = 0.5 * qt.tensor(sx, io)
    h_y = 0.5 * qt.tensor(sy, io)
    h_leak_x = qt.tensor(sx, a + a.dag())
    h_leak_y = qt.tensor(sy, a + a.dag())

    def omega_x_coeff(t, **kwargs):
        return omega_peak * sin2_envelope(t, total_time)

    def omega_y_coeff(t, **kwargs):
        d_omega = omega_peak * sin2_envelope_derivative(t, total_time)
        return -drag_lambda * d_omega / leakage_detuning

    def leak_x_coeff(t, **kwargs):
        return g * sin2_envelope(t, total_time)

    def leak_y_coeff(t, **kwargs):
        d_env = sin2_envelope_derivative(t, total_time)
        return -drag_lambda * g * d_env / leakage_detuning

    return [
        h_static,
        [h_x, omega_x_coeff],
        [h_y, omega_y_coeff],
        [h_leak_x, leak_x_coeff],
        [h_leak_y, leak_y_coeff],
    ]


def x_gate_initial_state(params: SimParams = PARAMS):
    require_qutip()
    return qt.tensor(qt.basis(2, 0), qt.basis(params.oscillator_levels, 0))


def oscillator_projectors(params: SimParams = PARAMS):
    require_qutip()
    iq = qt.qeye(2)
    projectors = []
    for k in range(params.oscillator_levels):
        projectors.append(qt.tensor(iq, qt.basis(params.oscillator_levels, k).proj()))
    high = sum(projectors[2:], projectors[0] * 0)
    return projectors, high


def sta_switch_hamiltonian(total_time: float, params: SimParams = PARAMS):
    """Two-level STA switching Hamiltonian for mesolve."""
    require_qutip()
    sz = qt.sigmaz()
    sy = qt.sigmay()

    h0 = 0.5 * params.omega0 * sz
    hy = 0.5 * sy

    def z_coeff(t, _args):
        return 0.0

    def y_coeff(t, _args):
        return -float(theta_dot_sta(t, total_time))

    return [[h0, z_coeff], [hy, y_coeff]]


def sta_initial_state():
    require_qutip()
    return qt.basis(2, 0)


def sta_target_state():
    require_qutip()
    return qt.basis(2, 1)
