from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

SMOKE_SCRIPTS = [
    "run_rabi_scaling.py",
    "run_topological_readout.py",
    "extract_smooth_alpha3_boundary_table.py",
    "run_standard_process_fidelity_xgate.py",
    "run_truncation_convergence.py",
]


def main():
    for script in SMOKE_SCRIPTS:
        path = ROOT / "scripts" / script
        print(f"\n=== {script} ===")
        completed = subprocess.run([sys.executable, str(path)], cwd=str(ROOT), check=False)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
