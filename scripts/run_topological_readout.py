from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.metrics import FIG_DIR, ensure_outputs, save_csv
from src.params import PARAMS
from src.simulations import topological_interferometer


def main():
    ensure_outputs()
    ports = PARAMS.readout_ports
    n_values = np.arange(ports)
    probs = topological_interferometer(n_values, ports=ports, phase_noise=0.03, samples=200)
    rows = {"n": n_values, "predicted_port": (-n_values) % ports, "observed_port": np.argmax(probs, axis=1)}
    for j in range(ports):
        rows[f"P_port_{j}"] = probs[:, j]
    save_csv("topological_readout.csv", rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(probs, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("output port j")
    ax.set_ylabel("winding n")
    ax.set_title("M-port topological readout: peak at j = -n mod M")
    fig.colorbar(im, ax=ax, label="probability")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "topological_readout.png", dpi=180)
    accuracy = np.mean(rows["predicted_port"] == rows["observed_port"])
    print(f"[topological_readout] port accuracy={accuracy:.3f}")


if __name__ == "__main__":
    main()
