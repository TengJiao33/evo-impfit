# Smoke experiment: FEMEIF software path

Status: complete and local-only.

Purpose:
Verify that the software implementation can construct a synthetic impedance
target, evaluate FEMEIF-style objectives, and run the evolutionary loop without
numerical or file-output failures.

Unit:
One synthetic fitting task instance.

Primary contrast:
None for claim purposes. The run is a plumbing smoke test, not a performance
claim.

Secondary contrast:
WBEIF baseline can be run with the same seed later, but any comparison needs
multiple seeds before it is treated as evidence.

Success signal:
The CLI finishes, writes `summary.json`, `history.csv`, and `best_fit.csv`, and
all reported losses are finite.

Failure signal:
NaN/Inf objectives, crashes, missing outputs, or impedance arrays with invalid
values.

Invalidation conditions:
Wrong target construction, hidden data leakage from true parameters into the
optimizer, unstable CSV output, or using a single stochastic run as a performance
claim.

Expected artifact paths:
`output/smoke-femeif/`, `output/smoke-wbeif/`, and this README.

Commands:

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests
python -m evo_impfit.cli demo --scenario rlc_case1 --generations 5 --population 20 --seed 7 --out output/smoke-femeif
python -m evo_impfit.cli demo --scenario rlc_case1 --mode wbeif --generations 5 --population 20 --seed 7 --out output/smoke-wbeif
```

Observed output:

- Unit tests: 3 passed.
- FEMEIF smoke: `best_wsmse=238.198`.
- WBEIF smoke: `best_wsmse=170.295`.

Interpretation:
The run only verifies plumbing and finite objective values. The stochastic
settings are intentionally tiny and are not suitable for judging algorithmic
performance.
