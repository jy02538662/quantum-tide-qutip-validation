from pathlib import Path
import csv
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.metrics import DATA_DIR, ensure_outputs, save_csv


def load_smooth_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            drag_lambda = float(row["drag_lambda"])
            if abs(drag_lambda) > 1e-12:
                continue
            rows.append({
                "g0_over_omega_rabi0": float(row["g0_over_omega_rabi0"]),
                "g0": float(row["g0"]),
                "v_low": float(row["v_low"]),
                "oscillator_ratio": float(row["oscillator_ratio"]),
                "drag_lambda": drag_lambda,
                "t_gate": float(row["t_gate"]),
                "max_high_population": float(row["max_high_population"]),
                "final_high_population": float(row["final_high_population"]),
                "F_X": float(row["F_X"]),
                "passes_leakage": int(float(row["passes_leakage"])),
                "passes_fidelity": int(float(row["passes_fidelity"])),
                "passes_both": int(float(row["passes_both"])),
            })
    return rows


def select_min(rows, key):
    return min(rows, key=lambda row: (row[key], row["t_gate"], -row["F_X"]))


def select_max(rows, key):
    return max(rows, key=lambda row: (row[key], -row["t_gate"], -row["max_high_population"]))


def summarize_group(r_value: float, group: list[dict]):
    pass_rows = [row for row in group if row["passes_both"] == 1]
    leak_rows = [row for row in group if row["passes_leakage"] == 1]
    if not pass_rows:
        near = max(group, key=lambda row: row["F_X"] - 10 * max(row["max_high_population"] - 1e-3, 0.0))
        return {
            "g0_over_omega_rabi0": r_value,
            "pass_count": 0,
            "total_count": len(group),
            "min_pass_v_low": np.nan,
            "min_ratio_at_min_v": np.nan,
            "min_v_at_min_ratio": np.nan,
            "min_pass_ratio": np.nan,
            "fastest_pass_t_gate": np.nan,
            "fastest_pass_v_low": np.nan,
            "fastest_pass_ratio": np.nan,
            "fastest_pass_P_high_max": np.nan,
            "fastest_pass_F_X": np.nan,
            "best_fidelity_pass_F_X": np.nan,
            "best_fidelity_pass_v_low": np.nan,
            "best_fidelity_pass_ratio": np.nan,
            "best_leakage_pass_P_high_max": np.nan,
            "best_leakage_only_P_high_max": min(row["max_high_population"] for row in leak_rows) if leak_rows else np.nan,
            "near_pass_v_low": near["v_low"],
            "near_pass_ratio": near["oscillator_ratio"],
            "near_pass_P_high_max": near["max_high_population"],
            "near_pass_F_X": near["F_X"],
            "near_pass_t_gate": near["t_gate"],
        }

    min_v = min(row["v_low"] for row in pass_rows)
    rows_at_min_v = [row for row in pass_rows if row["v_low"] == min_v]
    min_ratio_at_min_v_row = select_min(rows_at_min_v, "oscillator_ratio")
    min_ratio = min(row["oscillator_ratio"] for row in pass_rows)
    rows_at_min_ratio = [row for row in pass_rows if row["oscillator_ratio"] == min_ratio]
    min_v_at_min_ratio_row = select_min(rows_at_min_ratio, "v_low")

    min_v_row = select_min(pass_rows, "v_low")
    min_ratio_row = select_min(pass_rows, "oscillator_ratio")
    fastest_row = select_min(pass_rows, "t_gate")
    best_fidelity_row = select_max(pass_rows, "F_X")
    best_leakage_row = select_min(pass_rows, "max_high_population")

    return {
        "g0_over_omega_rabi0": r_value,
        "pass_count": len(pass_rows),
        "total_count": len(group),
        "min_pass_v_low": min_v_row["v_low"],
        "min_ratio_at_min_v": min_ratio_at_min_v_row["oscillator_ratio"],
        "min_v_at_min_ratio": min_v_at_min_ratio_row["v_low"],
        "min_pass_ratio": min_ratio_row["oscillator_ratio"],
        "fastest_pass_t_gate": fastest_row["t_gate"],
        "fastest_pass_v_low": fastest_row["v_low"],
        "fastest_pass_ratio": fastest_row["oscillator_ratio"],
        "fastest_pass_P_high_max": fastest_row["max_high_population"],
        "fastest_pass_F_X": fastest_row["F_X"],
        "best_fidelity_pass_F_X": best_fidelity_row["F_X"],
        "best_fidelity_pass_v_low": best_fidelity_row["v_low"],
        "best_fidelity_pass_ratio": best_fidelity_row["oscillator_ratio"],
        "best_leakage_pass_P_high_max": best_leakage_row["max_high_population"],
        "best_leakage_only_P_high_max": best_leakage_row["max_high_population"],
        "near_pass_v_low": np.nan,
        "near_pass_ratio": np.nan,
        "near_pass_P_high_max": np.nan,
        "near_pass_F_X": np.nan,
        "near_pass_t_gate": np.nan,
    }


def main():
    ensure_outputs()
    input_path = DATA_DIR / "drag_alpha3_midlow_scan.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"missing input CSV: {input_path}")

    rows = load_smooth_rows(input_path)
    r_values = sorted({row["g0_over_omega_rabi0"] for row in rows})
    summaries = [
        summarize_group(r_value, [row for row in rows if row["g0_over_omega_rabi0"] == r_value])
        for r_value in r_values
    ]
    output_path = save_csv(
        "smooth_alpha3_feasibility_boundary_table.csv",
        {key: [row[key] for row in summaries] for key in summaries[0]},
    )

    print("[smooth_boundary] wrote", output_path)
    print("\nsmooth sin^2 pulse alpha=3 boundary, drag_lambda=0")
    print("r      pass   min_v  ratio@min_v  min_ratio  v@min_ratio  fastest_t  fastest_v  fastest_ratio  P_high      F_X")
    for row in summaries:
        print(
            f"{row['g0_over_omega_rabi0']:.3f}  "
            f"{int(row['pass_count']):02d}/{int(row['total_count']):02d}   "
            f"{row['min_pass_v_low']:.3f}   "
            f"{row['min_ratio_at_min_v']:.1f}          "
            f"{row['min_pass_ratio']:.1f}        "
            f"{row['min_v_at_min_ratio']:.3f}        "
            f"{row['fastest_pass_t_gate']:.3f}      "
            f"{row['fastest_pass_v_low']:.3f}       "
            f"{row['fastest_pass_ratio']:.1f}            "
            f"{row['fastest_pass_P_high_max']:.3e}  "
            f"{row['fastest_pass_F_X']:.6f}"
        )


if __name__ == "__main__":
    main()
