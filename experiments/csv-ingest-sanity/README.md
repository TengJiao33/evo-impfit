# CSV ingest sanity experiment

Status: complete.

Purpose:
Verify that a synthetic target can be exported to CSV and then reloaded through
the same path intended for measured or private impedance data.

Unit:
One exported target CSV and one small FEMEIF fitting run over that CSV.

Primary contrast:
None. This is a data-ingestion plumbing check.

Success signal:
The export command writes CSV and metadata JSON, and `fit-csv` writes
`summary.json`, `history.csv`, and `best_fit.csv` with finite objective values.

Failure signal:
CSV schema mismatch, non-finite impedance values, missing output files, or
incorrect model preset selection.

Invalidation conditions:
Treating the tiny fitting run as performance evidence, or assuming that the
synthetic target represents the paper's private motor-bench measurement.

Commands:

```powershell
$env:PYTHONPATH='src'
python -m evo_impfit.cli export-scenario --scenario rlc_case1 --out data/rlc_case1_target.csv
python -m evo_impfit.cli fit-csv --target data/rlc_case1_target.csv --model-preset rlc6 --generations 3 --population 12 --seed 2 --out output/fit-csv-sanity
```

Observed output:

- Exported rows: 180.
- Fit path output: `output/fit-csv-sanity/`.
- Small-run best WSMSE: `259.713`.
- Unit tests after implementation: 5 passed.

Interpretation:
The CSV path is ready for measured data once a model preset and parameter
bounds are chosen. Exact replication of the paper's motor-drive case still
requires the original measured common-mode impedance trace.
