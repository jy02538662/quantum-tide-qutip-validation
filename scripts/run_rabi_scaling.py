from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.metrics import FIG_DIR, ensure_outputs, loglog_slope, save_csv
from src.simulations import rabi_scaling


def main():
    ensure_outputs()
    v = np.geomspace(0.06, 1.0, 80)
    result = rabi_scaling(v)
    slope = loglog_slope(result["v_low"], result["omega_rabi"])
    save_csv("rabi_scaling.csv", result)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(result["v_low"], result["omega_rabi"], label=f"fit slope = {slope:.3f}")
    ax.set_xlabel("v_low / v0")
    ax.set_ylabel("Omega_R")
    ax.set_title("Quantum Tide Rabi scaling: Omega_R ∝ v_low^(-1/2)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rabi_scaling.png", dpi=180)
    print(f"[rabi_scaling] slope={slope:.4f}")


if __name__ == "__main__":
    main()
