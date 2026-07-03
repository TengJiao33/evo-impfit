# FEMEIF Prototype Showcase

This repository contains a runnable software prototype for the FEMEIF impedance
fitting workflow.

## Current Status

- Implemented six-block RLC-parallel Case I reconstruction from the paper.
- Implemented WBEIF-style single-objective baseline.
- Implemented FEMEIF-style auxiliary objectives: PTC, DTW, LFP, SS, and LMEP.
- Implemented repeated comparison experiments over variants and random seeds.
- Implemented CSV export/import so measured impedance data can be fitted later.

## Terminal Demo

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests
python -m evo_impfit.cli compare --scenario rlc_case1 --variants wbeif,femeif_light,femeif_all --seeds 0,1 --generations 3 --population 12 --out output/comparison-sanity
Get-Content output/comparison-sanity/summary.csv
```

Expected outputs:

- Unit tests should pass.
- `output/comparison-sanity/summary.csv` should contain one row per variant.
- This short run is only a sanity check, not a paper-level reproduction claim.

## Mentor-Facing Plots

```powershell
$env:PYTHONPATH='src'
python -m evo_impfit.cli showcase --scenario rlc_case1 --variants wbeif,femeif_dtw,femeif_all --seeds 0,1,2,3,4 --generations 20 --population 50 --out output/teacher-case1
```

The command writes:

- `convergence.png`: WSMSE convergence over generations.
- `final_fit.png`: target impedance versus the best WBEIF and best FEMEIF fits.
- `summary.csv`, `runs.csv`, `history.csv`, `best_parameters.csv`, and `best_fits.csv`.

## Data Note

Case I can be reconstructed from the paper's public circuit table. The real
motor-drive Case II requires the original measured common-mode impedance trace
for strict reproduction. Until that data is available, `motor_synthetic` is only
for software testing and should not be reported as the paper's industrial case.
