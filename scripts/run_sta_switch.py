from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.metrics import FIG_DIR, ensure_outputs, save_csv
from src.simulations import sta_switch_error
from src.schedules import sta_omega_max


def main():
    ensure_outputs()
    t_values = np.linspace(20.0, 220.0, 120)
    result = sta_switch_error(t_values)
    result["omega_max"] = np.array([sta_omega_max(t) for t in t_values])
    save_csv("sta_switch_error.csv", result)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(result["switch_time"], result["error_bound"] + 1e-16)
    ax.axhline(1e-4, color="tab:red", linestyle="--", label="1e-4 target")
    ax.set_xlabel("switching time T")
    ax.set_ylabel("STA excitation error bound")
    ax.set_title("STA switching: nonadiabatic excitation bound")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sta_switch_error.png", dpi=180)
    print("[sta_switch] wrote sta_switch_error.csv/png")


if __name__ == "__main__":
    main()
