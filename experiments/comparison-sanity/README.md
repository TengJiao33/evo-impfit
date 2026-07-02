# Comparison sanity experiment

Status: complete.

Purpose:
Verify the repeated-run comparison harness for FEMEIF variants and WBEIF.

Unit:
One stochastic optimizer run for one `(scenario, variant, seed)` tuple.

Primary contrast:
None for performance claims. The immediate question is whether repeated runs
are materialized and summarized correctly.

Secondary contrasts:
Small-run WBEIF, FEMEIF light, and FEMEIF all-objective variants on synthetic
Case I.

Success signal:
The command writes `runs.csv`, `history.csv`, `summary.csv`, and `summary.json`
with one row per requested run and finite best WSMSE values.

Failure signal:
Missing run rows, non-finite values, variant parsing errors, or inconsistent
seed accounting.

Invalidation conditions:
Treating the tiny sanity run as a performance result, changing scenario data
between variants, or accidentally using different generation/population budgets.

Expected artifact path:
`output/comparison-sanity/`.

Command:

```powershell
$env:PYTHONPATH='src'
python -m evo_impfit.cli compare --scenario rlc_case1 --variants wbeif,femeif_light,femeif_all --seeds 0,1 --generations 3 --population 12 --out output/comparison-sanity
```

Observed output:

- Requested variants: `wbeif`, `femeif_light`, `femeif_all`.
- Seeds: `0`, `1`.
- Run rows written: 6.
- Files written: `runs.csv`, `history.csv`, `summary.csv`, `summary.json`.
- Unit tests after implementation: 4 passed.

Interpretation:
The comparison harness is functioning. The run is deliberately too small for
algorithmic claims, and the summary values should only be used to inspect
format, seed accounting, and runtime bookkeeping.
