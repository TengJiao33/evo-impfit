"""Showcase runs and plots for mentor-facing demonstrations."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
import time

import numpy as np

from .data import Scenario
from .experiments import RunSummary, VARIANTS, summarize_runs
from .features import FeatureConfig, magnitude_db, phase_rad
from .optimizer import FittingResult, run_fitting


@dataclass(frozen=True)
class ShowcaseRun:
    variant: str
    seed: int
    runtime_sec: float
    generations: int
    population: int
    result: FittingResult


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_rows(runs: list[ShowcaseRun]) -> list[dict[str, object]]:
    return [
        {
            "variant": run.variant,
            "mode": run.result.mode,
            "objectives": ",".join(run.result.objective_names),
            "seed": run.seed,
            "best_wsmse": run.result.best_loss,
            "runtime_sec": run.runtime_sec,
            "generations": run.generations,
            "population": run.population,
        }
        for run in runs
    ]


def _history_rows(runs: list[ShowcaseRun]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        for item in run.result.history:
            rows.append(
                {
                    "variant": run.variant,
                    "mode": run.result.mode,
                    "seed": run.seed,
                    "generation": int(item["generation"]),
                    "best_wsmse": item["best_wsmse"],
                    "mean_wsmse": item["mean_wsmse"],
                    "best_f_wsmse": item["best_f_wsmse"],
                }
            )
    return rows


def _summary_rows(runs: list[ShowcaseRun]) -> list[dict[str, object]]:
    summaries = [
        RunSummary(
            variant=run.variant,
            mode=run.result.mode,
            objectives=run.result.objective_names,
            seed=run.seed,
            best_wsmse=run.result.best_loss,
            runtime_sec=run.runtime_sec,
            generations=run.generations,
            population=run.population,
        )
        for run in runs
    ]
    return summarize_runs(summaries)


def _best_run_by_variant(runs: list[ShowcaseRun]) -> dict[str, ShowcaseRun]:
    best: dict[str, ShowcaseRun] = {}
    for run in runs:
        existing = best.get(run.variant)
        if existing is None or run.result.best_loss < existing.result.best_loss:
            best[run.variant] = run
    return best


def _write_best_artifacts(out_dir: Path, scenario: Scenario, runs: list[ShowcaseRun]) -> None:
    best_runs = _best_run_by_variant(runs)

    parameter_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    target_mag = magnitude_db(scenario.target_z)
    target_phase = phase_rad(scenario.target_z)

    for variant, run in sorted(best_runs.items()):
        for spec, value in zip(scenario.specs, run.result.best_values):
            parameter_rows.append(
                {
                    "variant": variant,
                    "seed": run.seed,
                    "best_wsmse": run.result.best_loss,
                    "parameter": spec.name,
                    "value": value,
                }
            )

        fit_mag = magnitude_db(run.result.best_fit_z)
        fit_phase = phase_rad(run.result.best_fit_z)
        for freq, t_mag, t_phase, f_mag, f_phase in zip(
            scenario.frequencies_hz,
            target_mag,
            target_phase,
            fit_mag,
            fit_phase,
        ):
            fit_rows.append(
                {
                    "variant": variant,
                    "seed": run.seed,
                    "frequency_hz": freq,
                    "target_magnitude_db": t_mag,
                    "target_phase_rad": t_phase,
                    "fit_magnitude_db": f_mag,
                    "fit_phase_rad": f_phase,
                }
            )

    _write_csv(
        out_dir / "best_parameters.csv",
        parameter_rows,
        ["variant", "seed", "best_wsmse", "parameter", "value"],
    )
    _write_csv(
        out_dir / "best_fits.csv",
        fit_rows,
        [
            "variant",
            "seed",
            "frequency_hz",
            "target_magnitude_db",
            "target_phase_rad",
            "fit_magnitude_db",
            "fit_phase_rad",
        ],
    )


def _apply_paper_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    times_font = Path("C:/Windows/Fonts/times.ttf")
    if times_font.exists():
        font_manager.fontManager.addfont(str(times_font))

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 7,
            "axes.linewidth": 0.55,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 5.8,
            "legend.borderpad": 0.25,
            "legend.handlelength": 2.1,
            "legend.handletextpad": 0.45,
            "legend.labelspacing": 0.2,
            "lines.solid_capstyle": "butt",
            "lines.dash_capstyle": "butt",
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def _finish_axes(ax) -> None:
    ax.minorticks_on()
    ax.tick_params(direction="in", top=True, right=True, width=0.55, length=2.6, pad=2.0)
    ax.tick_params(which="minor", direction="in", top=True, right=True, width=0.45, length=1.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)


def _save_figure(fig, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", pad_inches=0.025)


def _plot_convergence(out_dir: Path, runs: list[ShowcaseRun]) -> None:
    _apply_paper_style()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, ax = plt.subplots(figsize=(4.2, 2.55))
    available = {run.variant for run in runs}
    preferred = ("wbeif", "femeif_all", "femeif_dtw")
    variants = [name for name in preferred if name in available]
    variants.extend(sorted(available.difference(variants)))
    styles = {
        "wbeif": {"color": "#d62728", "label": "WBEIF(Mean)"},
        "femeif_all": {"color": "#1f77b4", "label": "FEMEIF(all objectives)(Mean)"},
        "femeif_dtw": {"color": "#9467bd", "label": "FEMEIF(WSMSE+DTW)(Mean)"},
        "femeif_light": {"color": "#2ca02c", "label": "FEMEIF(WSMSE+PTC+SS+LFP)(Mean)"},
        "femeif_ptc": {"color": "#ff7f0e", "label": "FEMEIF(WSMSE+PTC)(Mean)"},
        "femeif_no_dtw": {"color": "#7f7f7f", "label": "FEMEIF(all objectives-DTW)(Mean)"},
    }
    for variant in variants:
        variant_runs = [run for run in runs if run.variant == variant]
        generations = np.array(
            [item["generation"] for item in variant_runs[0].result.history],
            dtype=float,
        )
        losses = np.array(
            [
                [item["best_wsmse"] for item in run.result.history]
                for run in variant_runs
            ],
            dtype=float,
        )
        mean_loss = np.mean(losses, axis=0)
        style = styles.get(variant, {"color": None, "label": variant})
        if len(variant_runs) > 1:
            ci95 = 1.96 * np.std(losses, axis=0, ddof=1) / np.sqrt(len(variant_runs))
            ax.fill_between(
                generations,
                mean_loss - ci95,
                mean_loss + ci95,
                color=style["color"],
                alpha=0.08,
                linewidth=0.0,
            )
        ax.plot(
            generations,
            mean_loss,
            color=style["color"],
            linestyle="-",
            linewidth=0.85,
            label=style["label"],
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel("WSMSE (Reciprocal of fitting precision)")
    ax.set_xlim(left=0, right=max(item["generation"] for run in runs for item in run.result.history))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    _finish_axes(ax)
    ax.legend(loc="upper right", frameon=True, fancybox=False, framealpha=1.0, edgecolor="0.3")
    fig.tight_layout(pad=0.4)
    _save_figure(fig, out_dir / "convergence.png")
    plt.close(fig)


def _pick_final_fit_runs(runs: list[ShowcaseRun]) -> tuple[ShowcaseRun | None, ShowcaseRun | None]:
    wbeif_runs = [run for run in runs if run.result.mode == "wbeif" or run.variant == "wbeif"]
    femeif_runs = [run for run in runs if run.result.mode == "femeif" and run.variant != "wbeif"]
    best_wbeif = min(wbeif_runs, key=lambda run: run.result.best_loss) if wbeif_runs else None
    best_femeif = min(femeif_runs, key=lambda run: run.result.best_loss) if femeif_runs else None
    return best_wbeif, best_femeif


def _plot_final_fit(out_dir: Path, scenario: Scenario, runs: list[ShowcaseRun]) -> None:
    _apply_paper_style()
    import matplotlib.pyplot as plt

    best_wbeif, best_femeif = _pick_final_fit_runs(runs)
    freq = scenario.frequencies_hz
    target_mag = magnitude_db(scenario.target_z)
    target_phase = phase_rad(scenario.target_z)

    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(4.2, 2.9), sharex=True)
    ax_mag.plot(freq, target_mag, color="#0072bd", linewidth=0.9, label="Target", zorder=3)
    ax_phase.plot(freq, target_phase, color="#0072bd", linewidth=0.9, label="Target", zorder=3)

    if best_femeif is not None:
        ax_mag.plot(
            freq,
            magnitude_db(best_femeif.result.best_fit_z),
            color="#d62728",
            linestyle=(0, (4.0, 2.0)),
            linewidth=0.85,
            label="FEMEIF",
            zorder=2,
        )
        ax_phase.plot(
            freq,
            phase_rad(best_femeif.result.best_fit_z),
            color="#d62728",
            linestyle=(0, (4.0, 2.0)),
            linewidth=0.85,
            label="FEMEIF",
            zorder=2,
        )

    if best_wbeif is not None:
        ax_mag.plot(
            freq,
            magnitude_db(best_wbeif.result.best_fit_z),
            color="#edb120",
            linestyle=(0, (1.1, 1.7)),
            linewidth=0.95,
            label="WBEIF",
            zorder=4,
        )
        ax_phase.plot(
            freq,
            phase_rad(best_wbeif.result.best_fit_z),
            color="#edb120",
            linestyle=(0, (1.1, 1.7)),
            linewidth=0.95,
            label="WBEIF",
            zorder=4,
        )

    ax_mag.set_ylabel("Magnitude (dB)")
    ax_phase.set_ylabel("Phase (rad)")
    ax_phase.set_xlabel("Frequency(Hz)")
    for ax in (ax_mag, ax_phase):
        ax.set_xscale("log")
        _finish_axes(ax)

    ax_mag.set_xlim(freq.min(), freq.max())
    ax_mag.set_ylim(-40, 60)
    ax_phase.set_ylim(-2, 2)
    ax_phase.legend(loc="lower left", frameon=True, fancybox=False, framealpha=1.0, edgecolor="0.3")

    fig.tight_layout(pad=0.35, h_pad=0.35)
    _save_figure(fig, out_dir / "final_fit.png")
    plt.close(fig)


def run_showcase(
    scenario: Scenario,
    variant_names: tuple[str, ...],
    seeds: tuple[int, ...],
    generations: int,
    population: int,
    cfg: FeatureConfig,
    out_dir: Path,
) -> list[ShowcaseRun]:
    out_dir.mkdir(parents=True, exist_ok=True)
    runs: list[ShowcaseRun] = []

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
            runs.append(
                ShowcaseRun(
                    variant=variant.name,
                    seed=seed,
                    runtime_sec=time.perf_counter() - started,
                    generations=generations,
                    population=population,
                    result=result,
                )
            )

    summary_rows = _summary_rows(runs)
    _write_csv(
        out_dir / "runs.csv",
        _run_rows(runs),
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
        _history_rows(runs),
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
    _write_best_artifacts(out_dir, scenario, runs)
    _plot_convergence(out_dir, runs)
    _plot_final_fit(out_dir, scenario, runs)

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
        "plots": ["convergence.png", "final_fit.png"],
        "caveat": "Software-chain showcase on reconstructed Case I synthetic data; not a paper-level reproduction claim.",
    }
    (out_dir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return runs
