# Showcase smoke experiment

Status: complete.

Purpose:
Verify the mentor-facing `showcase` CLI path that generates convergence and
final-fit plots from the same fitting runs.

Unit:
One small synthetic Case I showcase run over WBEIF and two FEMEIF variants.

Primary contrast:
None for a performance claim. This run only verifies that the plot-producing
software path works.

Command:

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests
python -m evo_impfit.cli showcase --scenario rlc_case1 --variants wbeif,femeif_dtw,femeif_all --seeds 0,1 --generations 3 --population 12 --out output/showcase-smoke
```

Observed output:

- Unit tests: 6 passed.
- Showcase runs: 6.
- Generated artifacts: `convergence.png`, `final_fit.png`, `summary.csv`,
  `runs.csv`, `history.csv`, `best_parameters.csv`, and `best_fits.csv`.

Interpretation:
The two-figure teacher-facing path is functioning. The run is intentionally
small and should not be used as paper-level reproduction evidence.
