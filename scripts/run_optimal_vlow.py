from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.metrics import FIG_DIR, ensure_outputs, save_csv
from src.simulations import optimal_vlow


def main():
    ensure_outputs()
    v = np.geomspace(0.05, 1.0, 180)
    white = optimal_vlow(v, "white")
    one_f = optimal_vlow(v, "1/f")
    save_csv("optimal_vlow_white.csv", {k: white[k] for k in ["v_low", "fidelity", "gamma_decay"]})
    save_csv("optimal_vlow_1f.csv", {k: one_f[k] for k in ["v_low", "fidelity", "gamma_decay"]})

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(white["v_low"], white["fidelity"], label=f"white best v={white['best_v_low']:.3f}")
    ax.semilogx(one_f["v_low"], one_f["fidelity"], label=f"1/f best v={one_f['best_v_low']:.3f}")
    ax.set_xlabel("v_low / v0")
    ax.set_ylabel("analytic gate fidelity proxy")
    ax.set_title("Optimal programmable stiffness under noise spectra")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "optimal_vlow.png", dpi=180)
    print(f"[optimal_vlow] white best={white['best_v_low']:.4f}, 1/f best={one_f['best_v_low']:.4f}")


if __name__ == "__main__":
    main()
