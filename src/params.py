"""Shared parameters for Quantum Tide QuTiP simulations.

Units are normalized unless otherwise stated:
- hbar = 1
- v0 = 1
- angular frequencies are in inverse normalized time
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimParams:
    v0: float = 1.0
    v_high: float = 1.0
    v_low_default: float = 0.1
    omega0: float = 1.0
    omega_rabi0: float = 0.08
    oscillator_levels: int = 8
    leakage_coupling: float = 0.012
    dephasing0: float = 2.0e-4
    decay0: float = 2.0e-8
    v_gap: float = 0.16
    sta_control_error: float = 0.01
    readout_ports: int = 8


PARAMS = SimParams()
