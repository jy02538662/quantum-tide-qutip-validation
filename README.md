# Quantum Tide QuTiP Validation Suite

**Recommended GitHub repository name:** `quantum-tide-qutip-validation`

**Short description:** Full-space QuTiP simulations for Quantum Tide vortex stiffness, leakage suppression, Lindblad noise, and X-gate fidelity.

**Release status:** `v0.1.0-alpha` — full-space leakage and fidelity validation for academic research use.

## Abstract

This repository contains a QuTiP-based validation suite for the programmable Quantum Tide vortex-stiffness mechanism. The simulations explicitly retain a qubit fine-structure subspace coupled to a truncated breathing-mode oscillator, allowing high-energy virtual excitations discarded by Schrieffer-Wolff reduction to be directly monitored through their population. Under the strict leakage scaling `g_leak ∝ v_low^-3`, the code identifies both failure regions and feasible smooth-pulse operating windows. It further evaluates Lindblad dephasing and breathing-mode decay channels, scans the physical dephasing normalization `gamma_phi0 / Omega_R0`, and computes standard single-qubit X-gate average fidelity via entanglement fidelity. The main conclusion is that high-energy leakage can be suppressed below `1e-3` in a `v_low ≈ 0.24` smooth-pulse window, while final gate fidelity is ultimately limited by dephasing and low-frequency noise engineering.

## GitHub metadata

```text
Repository name:
  quantum-tide-qutip-validation

Description:
  Full-space QuTiP simulations for Quantum Tide vortex stiffness, leakage suppression, Lindblad noise, and X-gate fidelity.

Topics:
  qutip
  quantum-simulation
  quantum-control
  quantum-computing
  superconducting-qubits
  vortex-dynamics
  lindblad-master-equation
  gate-fidelity
  topological-quantum-computing
  quantum-tide
```

## Citation

```bibtex
@software{quantum_tide_qutip_validation_2026,
  title = {Quantum Tide QuTiP Validation Suite},
  author = {Quantum Tide / TopoCat Research},
  year = {2026},
  url = {https://github.com/YOUR-USERNAME/quantum-tide-qutip-validation},
  note = {Academic research non-commercial software}
}
```

## Overview

Numerical validation code for the Quantum Tide vortex quantum-computing notes:

- `QuantumTide_TopoCat_KnowledgeBase/16_quantum_classical_vortex_readout.md`
- `QuantumTide_TopoCat_KnowledgeBase/17_programmable_quantum_tide_stiffness.md`

The suite converts the analytic claims into reproducible QuTiP simulations. Its purpose is not to beautify results, but to expose hard feasibility windows, failure regions, and engineering limits.

## What this release validates

### 1. Full-space virtual-excitation check

The analytic Schrieffer-Wolff reduction removes high-energy breathing / quasiparticle states. In this suite they are retained explicitly:

```text
qubit two-level fine structure
+
truncated breathing oscillator, N = 5, 8, 10
```

The key monitored quantity is:

```text
P_high,max = max_t population(oscillator levels >= 2)
```

Under the strict leakage scaling:

```text
g_leak(v_low) = g0 * (v0 / v_low)^3
```

and a `sin^2` smooth X pulse, the validated operating window is:

```text
v_low = 0.24
alpha = 3
N = 8, with convergence checked at N = 5, 8, 10
```

Representative hard-pass points:

```text
g0/Omega_R0  omega_osc/omega_br  P_high,max  F_avg(noiseless)
0.020        3.0                 7.83e-04    0.999585
0.030        4.5                 3.90e-04    0.999882
0.040        5.5                 5.95e-04    0.999999
0.050        7.0                 4.44e-04    0.999994
```

The more aggressive point:

```text
v_low = 0.18, g0/Omega_R0 = 0.020, omega_osc/omega_br = 7.0
```

fails the leakage target:

```text
P_high,max = 1.312e-03 > 1e-3
```

This is a retained failure case, not hidden or tuned away.

### 2. Smooth pulse engineering boundary

The `sin^2` smooth pulse pushes the feasible region back into the mid-low stiffness window:

```text
g0/Omega_R0  min v_low  required ratio at min v_low
0.020        0.240      3.0
0.030        0.240      4.5
0.040        0.240      5.5
0.050        0.240      7.0
```

Simple first-order DRAG was tested. In the current model the useful improvement comes from the smooth envelope, not from nonzero DRAG coefficients.

### 3. Lindblad noise and dephasing threshold

The Lindblad simulations include pure dephasing and breathing-mode decay:

```text
c_ops = [sqrt(gamma_phi) * sigma_z, sqrt(gamma_decay) * a]
```

The physical dephasing normalization is scanned by:

```text
gamma_phi0 / Omega_R0 = 1e-4, 3e-4, 1e-3, 2.5e-3, 1e-2
```

Main conclusion:

```text
clean samples, gamma_phi0/Omega_R0 ~ 1e-4:
  F_avg > 0.999 is recovered

typical samples, gamma_phi0/Omega_R0 ~ 2.5e-3:
  white noise: F_avg ~ 0.997
  1/f proxy:   F_avg ~ 0.989
```

Thus the final bottleneck moves from high-energy leakage to dephasing / low-frequency noise.

### 4. Standard average gate fidelity

The release computes standard single-qubit average gate fidelity through entanglement fidelity:

```text
F_avg = (2 F_e + 1) / 3
```

where `F_e` is computed using a Bell state on a reference qubit and the simulated system qubit.

### 5. Truncation convergence

The first-defense conclusion is checked at:

```text
N = 5, 8, 10
```

The key pass/fail conclusions are unchanged across these truncations.

## Repository layout

```text
src/
  params.py          shared normalized simulation parameters
  schedules.py       Rabi, stiffness, and STA schedules
  hamiltonians.py    static, smooth-pulse, and STA Hamiltonian builders
  noise.py           analytic gamma_decay / gamma_phi models
  metrics.py         output and fidelity helpers
  simulations.py     lightweight analytic simulation helpers

scripts/
  run_all.py                              release smoke test
  run_rabi_scaling.py                     Omega_R ∝ v_low^-1/2
  run_topological_readout.py              M-port readout j = -n mod M
  run_x_gate_leakage.py                   baseline full-space leakage run
  run_x_gate_leakage_scan.py              alpha=0.5 vs alpha=3 leakage scan
  run_g0_ratio_alpha3_scan.py             physical g0/Omega_R0 scan
  extract_alpha3_boundary_table.py        static/rectangular boundary table
  run_drag_alpha3_midlow_scan.py          smooth pulse + DRAG scan
  extract_smooth_alpha3_boundary_table.py smooth-pulse boundary table
  run_smooth_lindblad_xgate.py            Lindblad six-state proxy
  run_gamma_phi_ratio_lindblad_scan.py    dephasing normalization threshold
  run_standard_process_fidelity_xgate.py  standard F_e and F_avg
  run_truncation_convergence.py           N=5/8/10 convergence
  run_sta_switch.py                       STA switching bound
  run_optimal_vlow.py                     analytic stiffness/noise optimum

outputs/
  data/       generated CSV result tables
  figures/    generated figures
```

## Install

```bash
pip install -r requirements.txt
```

On Windows, if `python` is not on PATH, use:

```bash
py -m pip install -r requirements.txt
```

## Run release smoke test

```bash
python scripts/run_all.py
```

or on Windows:

```bash
py scripts/run_all.py
```

The smoke test runs a representative subset:

```text
rabi scaling
topological readout
smooth boundary extraction
standard average gate fidelity
N=5/8/10 truncation convergence
```

Heavy scans are kept as individual scripts and are not all run by default.

## Reproduce key heavy scans

```bash
python scripts/run_g0_ratio_alpha3_scan.py
python scripts/run_drag_alpha3_midlow_scan.py
python scripts/run_gamma_phi_ratio_lindblad_scan.py
```

## Scientific boundary

These simulations are not experimental proof. They are numerical stress tests of the theory's internal claims:

```text
analytic derivation + full-space master-equation simulation
```

The strongest supported conclusion is:

```text
The strict alpha=3 leakage law imposes a real engineering boundary. Smooth pulses and detuning can suppress high-energy virtual excitations below 1e-3 in a v_low ≈ 0.24 window, while final gate fidelity under realistic noise is limited by dephasing and low-frequency noise engineering.
```

## License

Academic research use only. Commercial use is not permitted. See `LICENSE`.
