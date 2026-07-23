# gedmd-xxz-hydrodynamics

Reproduction code for

> **Temporal Coarse-Graining as the Origin of Macroscopic Friction in Quantum Spin Chains via Data-Driven Liouvillian Extraction**
> Seiki Saito, submitted to *Physical Review Research* (2026).

The paper extracts Navier–Stokes hydrodynamic coefficients (elasticity `c²`,
friction `γ`, kinematic viscosity `ν`) directly from the exact unitary dynamics
of a chaotic XXZ spin chain, using generalized Extended Dynamic Mode
Decomposition (gEDMD) combined with a Mori–Zwanzig projection, and studies how
these coefficients depend on the temporal coarse-graining scale `Δt_cg`.

A supplementary movie of the extracted macroscopic dynamics is archived at
https://doi.org/10.5281/zenodo.20059728 .

## Method in one paragraph

For a chosen dictionary of Hermitian observables `O`, the code records the
expectation-value time series `X(t) = ⟨ψ(t)|O|ψ(t)⟩` and either the exact time
derivative `Ẋ = i⟨ψ|[H,O]|ψ⟩` (exact-derivative gEDMD) or a finite difference
`[X(t+Δt_cg) − X(t)]/Δt_cg` (temporal coarse-graining). A finite-dimensional
generator `L` is obtained by the least-squares / Galerkin solution
`L = A G⁻¹` with `G = ⟨X, Xᵀ⟩`, `A = ⟨Ẋ, Xᵀ⟩`. Mapping the current rows of `L`
onto a generalized Navier–Stokes equation
`J̇ = −c²∇Z − γJ + ν∇²J` yields the local hydrodynamic coefficients.

## Repository layout

```
src/
  results1_q8_validation/     Fig. 2–4  (8-qubit, full vs. macroscopic dictionary)
  results2_system_environment/Fig. 5–7  (quench / system–environment decoherence)
  results3_hydrodynamics/     Fig. 8–10 (20-qubit coefficients vs. Δt_cg)
  results4_prediction/        Fig. 11   (predictive capability & dictionary size)
  appendixA_scheme/           Fig. 12–13(finite-difference scheme dependence)
data/                         small derived data (CSV / npz) for the lightweight plots
requirements.txt
LICENSE                       MIT
```

Scripts are numbered in execution order within each directory. Filenames
containing `generate`/`gen` produce raw time-series data; `calc`/`analyze`
extract spectra or coefficients; `plot` render figures.

## Environment

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
Tested with Python 3.9. The exact state-vector propagation uses
`scipy.sparse.linalg.expm_multiply` and the Al-Mohy–Higham algorithm; the
Hamiltonian and observables are built with `qiskit.quantum_info.SparsePauliOp`.

## Data

The raw quantum-evolution data are large (the 20-qubit hydrodynamics campaign is
500 random initial states × 149 observables × 2000 time steps, ≈ 2 GB) and are
**not** stored in this repository; the `generate_*` scripts regenerate them. The
8- and 20-qubit exact-diagonalization runs were performed on the NIFS Plasma
Simulator; an 8-qubit validation run fits on a workstation, while the 20-qubit
runs benefit from a multi-core node.

The small, post-processed data needed to redraw Figs. 9, 11, 12 and 13 **are**
included under `data/`, so those figures can be reproduced immediately:

```bash
python src/results4_prediction/2_plot_fig11.py          # Fig. 11
python src/appendixA_scheme/2_plot_scheme_comparison.py # Fig. 12
python src/appendixA_scheme/4_plot_time_resolved.py     # Fig. 13
```

## Reproduction pipeline

### Results I — 8-qubit validation (Figs. 2–4)
1. `1_generate_q8_data.py` — exact evolution of the 8-qubit chain; records `X`,
   `Ẋ` for the macroscopic dictionary (Dict A, 15 observables) and the complete
   Pauli basis (Dict B, `4⁸−1 = 65535` observables).
2. `2_window_spectra.py` — sliding-window gEDMD spectra for Dict A vs. Dict B
   (Fig. 2), and the exact- vs. finite-difference comparison (Fig. 3).
3. `3_reconstruction_dictA_vs_dictB.py` — analytic reconstruction
   `e^{Lt}X(0)` and its error vs. exact Schrödinger dynamics (Fig. 4).

### Results II — system–environment decoherence (Figs. 5–7)
1. `1_generate_q8_quench.py` / `2_generate_q20_quench.py` — two-phase quench
   protocol; the boundary interaction is switched on at `t = t_q`.
2. `3_analyze_q20_spectra.py` — sliding-window spectra for the target (Dict S),
   macroscopic pointer (Dict L) and environment-energy (Dict E) dictionaries
   (Figs. 5, 6).
3. `4_plot_trace.py` — Liouvillian trace `Tr(L) = Σ Re(λ)` tracking the
   synchronized information outflow/inflow at the quench (Fig. 7).

### Results III — hydrodynamics (Figs. 8–10)
1. `1_generate_current_data.py` — 20-qubit chain; records the 149-observable
   hydrodynamic dictionary (`Z_i`, spin currents `J_i`, `Z_iZ_{i+1}`, symmetric
   kinetic terms, and length-3 correlations) over an ensemble of random states.
2. `2_generate_cg_timeseries.py` — coefficients `c²`, `γ`, `ν`, `D`, `D_Z` at a
   set of coarse-graining steps.
3. `3_plot_coefficients_vs_time.py` — time evolution across `Δt_cg` (Fig. 8).
4. `4_calc_markovian_plateau.py` — coefficients vs. `Δt_cg`
   (writes `markovian_plateau_data.csv`); `5_plot_markovian_plateau.py` (Fig. 9).
5. `6_spatial_profiles_sliding_window.py` — spatial profiles of the bulk under
   exact vs. coarse-grained differentiation (Fig. 10).

### Results IV — predictive capability (Fig. 11)
`1_dict_scan_prediction.py` re-runs the extraction on nested sub-dictionaries
(20 / 39 / 77 / 149 observables) and evaluates the forward prediction
`e^{L(t−t_0)}X(t_0)`; `2_plot_fig11.py` renders the figure from `data/`.

### Appendix A — finite-difference scheme dependence (Figs. 12–13)
`1_scheme_comparison.py` compares the forward, backward and central differences;
`3_time_resolved_schemes.py` extracts the time-resolved friction `γ(t)`. The
`2_plot_*` / `4_plot_*` scripts render Figs. 12 and 13 from `data/`.

## Notes

- Several analysis/plot scripts contain hard-coded input filenames and expect to
  be run from a directory containing the corresponding regenerated data; the
  lightweight plot scripts for Figs. 11–13 instead read directly from `data/`.
- Comments in some scripts are in Japanese (the author's working language).

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use this code, please cite the paper above.
