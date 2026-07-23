"""Metrics and output helpers."""
from __future__ import annotations

from pathlib import Path
import csv

import numpy as np

try:
    import qutip as qt
except Exception:  # pragma: no cover
    qt = None

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"
DATA_DIR = ROOT / "outputs" / "data"


def ensure_outputs():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_csv(name: str, rows: dict[str, np.ndarray | list | float]):
    ensure_outputs()
    path = DATA_DIR / name
    keys = list(rows.keys())
    normalized = {}
    max_len = 1
    for key, value in rows.items():
        arr = np.atleast_1d(value)
        normalized[key] = arr
        max_len = max(max_len, len(arr))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(max_len):
            writer.writerow([
                normalized[key][i] if i < len(normalized[key]) else normalized[key][-1]
                for key in keys
            ])
    return path


def state_fidelity(state, target) -> float:
    if qt is None:
        raise RuntimeError("QuTiP is not installed")
    value = qt.metrics.fidelity(state, target)
    return float(value**2)


def loglog_slope(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0)
    coeff = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    return float(coeff[0])
