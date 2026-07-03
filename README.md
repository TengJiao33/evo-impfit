# evo-impfit

Research prototype for turning the FEMEIF paper method into runnable software.

The first version focuses on:

- Equivalent-circuit impedance models.
- Synthetic benchmark construction for cases that do not require private data.
- FEMEIF-style multi-objective evolutionary fitting.
- A baseline WBEIF-style single-objective evolutionary fitting mode.
- CSV outputs for later plotting and inspection.

## What can be tested without private data

The paper's Case I can be reconstructed because the generic six-block RLC-parallel
target circuit and component values are printed in the paper. This repository
builds that target synthetically and can run fitting against it.

The real motor-drive experiment depends on measured common-mode impedance from
a physical bench. That measured trace is not public in the paper, so exact
replication is blocked unless the data owner provides it. To keep development
testable, this repository includes a `motor_synthetic` scenario with a
motor-drive-like topology inspired by the paper figure. It is useful for
software and algorithm debugging, but it should not be reported as a replication
of the paper's industrial experiment.

## Quick start

```powershell
$env:PYTHONPATH = "src"
python -m evo_impfit.cli demo --scenario rlc_case1 --generations 30 --population 60 --seed 7 --out output/demo-case1
```

Run the baseline:

```powershell
$env:PYTHONPATH = "src"
python -m evo_impfit.cli demo --scenario rlc_case1 --mode wbeif --generations 30 --population 60 --seed 7 --out output/demo-wbeif
```

Run tests:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
```

Run a repeated comparison:

```powershell
$env:PYTHONPATH = "src"
python -m evo_impfit.cli compare --scenario rlc_case1 --variants wbeif,femeif_light,femeif_all --seeds 0,1,2 --generations 20 --population 50 --out output/compare-case1
```

Generate the two mentor-facing showcase plots:

```powershell
$env:PYTHONPATH = "src"
python -m evo_impfit.cli showcase --scenario rlc_case1 --variants wbeif,femeif_dtw,femeif_all --seeds 0,1,2,3,4 --generations 20 --population 50 --out output/teacher-case1
```

This writes `convergence.png` and `final_fit.png` under the output directory.

Available comparison variants:

- `wbeif`: single-objective baseline.
- `femeif_all`: all implemented auxiliary objectives.
- `femeif_light`: lower-cost PTC + SS + LFP set.
- `femeif_ptc`: PTC-only ablation.
- `femeif_dtw`: DTW-only ablation.
- `femeif_no_dtw`: all implemented objectives except DTW.

Export a synthetic target:

```powershell
$env:PYTHONPATH = "src"
python -m evo_impfit.cli export-scenario --scenario rlc_case1 --out data/rlc_case1_target.csv
```

Fit a measured or exported CSV:

```powershell
$env:PYTHONPATH = "src"
python -m evo_impfit.cli fit-csv --target data/rlc_case1_target.csv --model-preset rlc6 --generations 20 --population 50 --out output/fit-csv-case1
```

## CSV data path

Private or measured data can be added later as CSV with one of these schemas:

- `frequency_hz,real,imag`
- `frequency_hz,magnitude_db,phase_rad`

The fitting software still needs a chosen equivalent-circuit topology and
parameter bounds. The paper's real motor case does not publish enough measured
trace data to exactly reproduce its results from the PDF alone.
