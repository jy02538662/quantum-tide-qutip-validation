"""Time schedules and analytic scaling laws."""
from __future__ import annotations

import numpy as np


def omega_br(v: float | np.ndarray, omega0: float = 1.0, v0: float = 1.0):
    return omega0 * np.asarray(v) / v0


def omega_rabi(v_low: float | np.ndarray, omega_rabi0: float = 0.08, v0: float = 1.0):
    return omega_rabi0 * (np.asarray(v_low) / v0) ** (-0.5)


def gate_time_x(v_low: float | np.ndarray, omega_rabi0: float = 0.08, v0: float = 1.0):
    return np.pi / omega_rabi(v_low, omega_rabi0=omega_rabi0, v0=v0)


def smoothstep5(s: float | np.ndarray):
    s = np.asarray(s)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def v_schedule(t: float | np.ndarray, total_time: float, v_start: float, v_end: float):
    s = np.clip(np.asarray(t) / total_time, 0.0, 1.0)
    return v_start + (v_end - v_start) * smoothstep5(s)


def theta_sta(t: float | np.ndarray, total_time: float):
    s = np.clip(np.asarray(t) / total_time, 0.0, 1.0)
    return np.pi - 6 * np.pi * s**5 + 15 * np.pi * s**4 - 10 * np.pi * s**3


def theta_dot_sta(t: float | np.ndarray, total_time: float):
    s = np.clip(np.asarray(t) / total_time, 0.0, 1.0)
    return (-30 * np.pi * s**4 + 60 * np.pi * s**3 - 30 * np.pi * s**2) / total_time


def sta_omega_max(total_time: float):
    return 15 * np.pi / (8 * total_time)


def sta_error_bound(total_time: float, relative_control_error: float = 0.01):
    omega_max = sta_omega_max(total_time)
    return 0.5 * relative_control_error**2 * (1 - np.cos(omega_max * total_time))
