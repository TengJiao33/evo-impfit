"""Experiment orchestration helpers."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
import time

import numpy as np

from .data import Scenario
from .features import FeatureConfig
from .optimizer import Mode, run_fitting


ALL_OBJECTIVES = ("wsmse", "ptc", "dtw", "lfp", "ss", "lmep")


@dataclass(frozen=True)
class Variant:
    name: str
    mode: Mode
    objectives: tuple[str, ...]
    notes: str


@dataclass(frozen=True)
class RunSummary:
    variant: str
    mode: str
    objectives: tuple[str, ...]
    seed: int
    best_wsmse: float
    runtime_sec: float
    generations: int
    population: int


VARIANTS: dict[str, Variant] = {
    "wbeif": Variant(
        name="wbeif",
        mode="wbeif",
        objectives=("wsmse",),
        notes="Traditional single-objective baseline.",
    ),
    "femeif_all": Variant(
        name="femeif_all",
        mode="femeif",
        objectives=ALL_OBJECTIVES,
        notes="FEMEIF with all implemented auxiliary objectives.",
    ),
    "femeif_light": Variant(
        name="femeif_light",
        mode="femeif",
        objectives=("wsmse", "ptc", "ss", "lfp"),
        notes="Lower-cost FEMEIF set following the paper's time-sensitive guidance.",
    ),
    "femeif_ptc": Variant(
        name="femeif_ptc",
        mode="femeif",
        objectives=("wsmse", "ptc"),
        notes="Single-auxiliary ablation: peak/trough consistency.",
    ),
    "femeif_dtw": Variant(
        name="femeif_dtw",
        mode="femeif",
        objectives=("wsmse", "dtw"),
        notes="Single-auxiliary ablation: dynamic time warping.",
    ),
    "femeif_no_dtw": Variant(
        name="femeif_no_dtw",
        mode="femeif",
        objectives=("wsmse", "ptc", "lfp", "ss", "lmep"),
        notes="All implemented objectives except the expensive DTW objective.",
    ),
}


def parse_variant_names(raw: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not names:
        return ("wbeif", "femeif_all")
    unknown = sorted(name for name in names if name not in VARIANTS)
    if unknown:
        valid = ", ".join(sorted(VARIANTS))
        raise ValueError(f"Unknown variant(s): {', '.join(unknown)}. Valid variants: {valid}")
    return names


def summarize_runs(rows: list[RunSummary]) -> list[dict[str, float | str | int]]:
    grouped: dict[str, list[RunSummary]] = {}
    for row in rows:
        grouped.setdefault(row.variant, []).append(row)

    summary: list[dict[str, float | str | int]] = []
    for variant, values in sorted(grouped.items()):
        losses = np.array([row.best_wsmse for row in values], dtype=float)
        runtimes = np.array([row.runtime_sec for row in values], dtype=float)
        summary.append(
            {
                "variant": variant,
                "runs": int(len(values)),
                "mean_best_wsmse": float(np.mean(losses)),
                "std_best_wsmse": float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
                "min_best_wsmse": float(np.min(losses)),
                "max_best_wsmse": float(np.max(losses)),
                "mean_runtime_sec": float(np.mean(runtimes)),
            }
        )
    return summary


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_comparison(
    scenario: Scenario,
    variant_names: tuple[str, ...],
    seeds: tuple[int, ...],
    generations: int,
    population: int,
    cfg: FeatureConfig,
    out_dir: Path,
) -> list[RunSummary]:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_rows: list[RunSummary] = []
    history_rows: list[dict[str, object]] = []

    for variant_name in variant_names:
        variant = VARIANTS[variant_name]
        for seed in seeds:
            started = time.perf_counter()
            result = run_fitting(
                scenario,
                mode=variant.mode,
                objective_names=variant.objectives,
                generations=generations,
                population_size=population,
                seed=seed,
                cfg=cfg,
            )
            runtime = time.perf_counter() - started
            run_rows.append(
                RunSummary(
                    variant=variant.name,
                    mode=variant.mode,
                    objectives=result.objective_names,
                    seed=seed,
                    best_wsmse=result.best_loss,
                    runtime_sec=runtime,
                    generations=generations,
                    population=population,
                )
            )
            for item in result.history:
                history_rows.append(
                    {
                        "variant": variant.name,
                        "mode": variant.mode,
                        "seed": seed,
                        "generation": int(item["generation"]),
                        "best_wsmse": item["best_wsmse"],
                        "mean_wsmse": item["mean_wsmse"],
                        "best_f_wsmse": item["best_f_wsmse"],
                    }
                )

    run_dicts = [
        {
            "variant": row.variant,
            "mode": row.mode,
            "objectives": ",".join(row.objectives),
            "seed": row.seed,
            "best_wsmse": row.best_wsmse,
            "runtime_sec": row.runtime_sec,
            "generations": row.generations,
            "population": row.population,
        }
        for row in run_rows
    ]
    summary_rows = summarize_runs(run_rows)

    _write_csv(
        out_dir / "runs.csv",
        run_dicts,
        [
            "variant",
            "mode",
            "objectives",
            "seed",
            "best_wsmse",
            "runtime_sec",
            "generations",
            "population",
        ],
    )
    _write_csv(
        out_dir / "history.csv",
        history_rows,
        [
            "variant",
            "mode",
            "seed",
            "generation",
            "best_wsmse",
            "mean_wsmse",
            "best_f_wsmse",
        ],
    )
    _write_csv(
        out_dir / "summary.csv",
        summary_rows,
        [
            "variant",
            "runs",
            "mean_best_wsmse",
            "std_best_wsmse",
            "min_best_wsmse",
            "max_best_wsmse",
            "mean_runtime_sec",
        ],
    )
    metadata = {
        "scenario": scenario.name,
        "notes": scenario.notes,
        "variants": {
            name: {
                "mode": VARIANTS[name].mode,
                "objectives": VARIANTS[name].objectives,
                "notes": VARIANTS[name].notes,
            }
            for name in variant_names
        },
        "seeds": list(seeds),
        "generations": generations,
        "population": population,
        "summary": summary_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return run_rows
