"""Command-line interface for the impedance fitting prototype."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .data import make_scenario, scenario_from_csv
from .experiments import parse_variant_names, run_comparison
from .features import FeatureConfig, magnitude_db, phase_rad
from .optimizer import run_fitting
from .showcase import run_showcase


DEFAULT_OBJECTIVES = ("wsmse", "ptc", "dtw", "lfp", "ss", "lmep")


def _parse_objectives(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    if not values:
        return DEFAULT_OBJECTIVES
    if "wsmse" not in values:
        values = ("wsmse",) + values
    return values


def _write_history(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["generation", "best_wsmse", "mean_wsmse", "best_f_wsmse"])
        writer.writeheader()
        writer.writerows(history)


def _write_fit(path: Path, scenario, fit_z) -> None:
    target_mag = magnitude_db(scenario.target_z)
    target_phase = phase_rad(scenario.target_z)
    fit_mag = magnitude_db(fit_z)
    fit_phase = phase_rad(fit_z)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frequency_hz",
                "target_magnitude_db",
                "target_phase_rad",
                "fit_magnitude_db",
                "fit_phase_rad",
            ]
        )
        for row in zip(scenario.frequencies_hz, target_mag, target_phase, fit_mag, fit_phase):
            writer.writerow([f"{value:.12g}" for value in row])


def _write_target(path: Path, scenario) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mag = magnitude_db(scenario.target_z)
    phase = phase_rad(scenario.target_z)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frequency_hz", "real", "imag", "magnitude_db", "phase_rad"])
        for freq, z_value, mag_value, phase_value in zip(
            scenario.frequencies_hz,
            scenario.target_z,
            mag,
            phase,
        ):
            writer.writerow(
                [
                    f"{freq:.12g}",
                    f"{z_value.real:.12g}",
                    f"{z_value.imag:.12g}",
                    f"{mag_value:.12g}",
                    f"{phase_value:.12g}",
                ]
            )


def run_demo(args: argparse.Namespace) -> int:
    scenario = make_scenario(
        args.scenario,
        noise_mag_db=args.noise_mag_db,
        noise_phase_rad=args.noise_phase_rad,
        seed=args.seed,
    )
    cfg = FeatureConfig(alpha=args.alpha)
    objectives = _parse_objectives(args.objectives)
    result = run_fitting(
        scenario,
        mode=args.mode,
        objective_names=objectives,
        generations=args.generations,
        population_size=args.population,
        seed=args.seed,
        cfg=cfg,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_history(out_dir / "history.csv", result.history)
    _write_fit(out_dir / "best_fit.csv", scenario, result.best_fit_z)
    summary = {
        "scenario": result.scenario_name,
        "mode": result.mode,
        "objectives": result.objective_names,
        "generations": args.generations,
        "population": args.population,
        "seed": args.seed,
        "best_wsmse": result.best_loss,
        "best_objectives": result.best_objectives,
        "parameters": {
            spec.name: float(value)
            for spec, value in zip(scenario.specs, result.best_values)
        },
        "notes": scenario.notes,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"scenario={result.scenario_name}")
    print(f"mode={result.mode}")
    print(f"objectives={','.join(result.objective_names)}")
    print(f"best_wsmse={result.best_loss:.6g}")
    print(f"out={out_dir}")
    return 0


def run_export_scenario(args: argparse.Namespace) -> int:
    scenario = make_scenario(
        args.scenario,
        noise_mag_db=args.noise_mag_db,
        noise_phase_rad=args.noise_phase_rad,
        seed=args.seed,
    )
    out_path = Path(args.out)
    _write_target(out_path, scenario)
    metadata = {
        "scenario": scenario.name,
        "rows": int(len(scenario.frequencies_hz)),
        "notes": scenario.notes,
        "true_parameters": (
            {
                spec.name: float(value)
                for spec, value in zip(scenario.specs, scenario.true_values)
            }
            if scenario.true_values is not None
            else None
        ),
    }
    metadata_path = out_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"scenario={scenario.name}")
    print(f"rows={len(scenario.frequencies_hz)}")
    print(f"out={out_path}")
    print(f"metadata={metadata_path}")
    return 0


def run_fit_csv(args: argparse.Namespace) -> int:
    scenario = scenario_from_csv(args.target, args.model_preset, name=args.name)
    cfg = FeatureConfig(alpha=args.alpha)
    objectives = _parse_objectives(args.objectives)
    result = run_fitting(
        scenario,
        mode=args.mode,
        objective_names=objectives,
        generations=args.generations,
        population_size=args.population,
        seed=args.seed,
        cfg=cfg,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_history(out_dir / "history.csv", result.history)
    _write_fit(out_dir / "best_fit.csv", scenario, result.best_fit_z)
    summary = {
        "target": str(args.target),
        "scenario": result.scenario_name,
        "model_preset": args.model_preset,
        "mode": result.mode,
        "objectives": result.objective_names,
        "generations": args.generations,
        "population": args.population,
        "seed": args.seed,
        "best_wsmse": result.best_loss,
        "best_objectives": result.best_objectives,
        "parameters": {
            spec.name: float(value)
            for spec, value in zip(scenario.specs, result.best_values)
        },
        "notes": scenario.notes,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"target={args.target}")
    print(f"model_preset={args.model_preset}")
    print(f"mode={result.mode}")
    print(f"best_wsmse={result.best_loss:.6g}")
    print(f"out={out_dir}")
    return 0


def _parse_seeds(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def run_compare(args: argparse.Namespace) -> int:
    scenario = make_scenario(
        args.scenario,
        noise_mag_db=args.noise_mag_db,
        noise_phase_rad=args.noise_phase_rad,
        seed=args.scenario_seed,
    )
    cfg = FeatureConfig(alpha=args.alpha)
    variants = parse_variant_names(args.variants)
    seeds = _parse_seeds(args.seeds)
    out_dir = Path(args.out)
    rows = run_comparison(
        scenario=scenario,
        variant_names=variants,
        seeds=seeds,
        generations=args.generations,
        population=args.population,
        cfg=cfg,
        out_dir=out_dir,
    )
    print(f"scenario={scenario.name}")
    print(f"variants={','.join(variants)}")
    print(f"runs={len(rows)}")
    print(f"out={out_dir}")
    return 0


def run_showcase_command(args: argparse.Namespace) -> int:
    scenario = make_scenario(
        args.scenario,
        noise_mag_db=args.noise_mag_db,
        noise_phase_rad=args.noise_phase_rad,
        seed=args.scenario_seed,
    )
    cfg = FeatureConfig(alpha=args.alpha)
    variants = parse_variant_names(args.variants)
    seeds = _parse_seeds(args.seeds)
    out_dir = Path(args.out)
    rows = run_showcase(
        scenario=scenario,
        variant_names=variants,
        seeds=seeds,
        generations=args.generations,
        population=args.population,
        cfg=cfg,
        out_dir=out_dir,
    )
    print(f"scenario={scenario.name}")
    print(f"variants={','.join(variants)}")
    print(f"runs={len(rows)}")
    print(f"convergence={out_dir / 'convergence.png'}")
    print(f"final_fit={out_dir / 'final_fit.png'}")
    print(f"out={out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evo-impfit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run a synthetic fitting demo.")
    demo.add_argument("--scenario", choices=["rlc_case1", "motor_synthetic"], default="rlc_case1")
    demo.add_argument("--mode", choices=["femeif", "wbeif"], default="femeif")
    demo.add_argument("--objectives", default=",".join(DEFAULT_OBJECTIVES))
    demo.add_argument("--generations", type=int, default=40)
    demo.add_argument("--population", type=int, default=80)
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--alpha", type=float, default=25.12)
    demo.add_argument("--noise-mag-db", type=float, default=0.0)
    demo.add_argument("--noise-phase-rad", type=float, default=0.0)
    demo.add_argument("--out", default="output/demo")
    demo.set_defaults(func=run_demo)

    export_scenario = subparsers.add_parser("export-scenario", help="Export a synthetic target CSV.")
    export_scenario.add_argument("--scenario", choices=["rlc_case1", "motor_synthetic"], default="rlc_case1")
    export_scenario.add_argument("--seed", type=int, default=0)
    export_scenario.add_argument("--noise-mag-db", type=float, default=0.0)
    export_scenario.add_argument("--noise-phase-rad", type=float, default=0.0)
    export_scenario.add_argument("--out", default="data/rlc_case1_target.csv")
    export_scenario.set_defaults(func=run_export_scenario)

    fit_csv = subparsers.add_parser("fit-csv", help="Fit a CSV target with a preset model.")
    fit_csv.add_argument("--target", required=True)
    fit_csv.add_argument("--model-preset", choices=["rlc6", "motor_like"], default="rlc6")
    fit_csv.add_argument("--name", default=None)
    fit_csv.add_argument("--mode", choices=["femeif", "wbeif"], default="femeif")
    fit_csv.add_argument("--objectives", default=",".join(DEFAULT_OBJECTIVES))
    fit_csv.add_argument("--generations", type=int, default=40)
    fit_csv.add_argument("--population", type=int, default=80)
    fit_csv.add_argument("--seed", type=int, default=0)
    fit_csv.add_argument("--alpha", type=float, default=25.12)
    fit_csv.add_argument("--out", default="output/fit-csv")
    fit_csv.set_defaults(func=run_fit_csv)

    compare = subparsers.add_parser("compare", help="Run repeated synthetic comparisons.")
    compare.add_argument("--scenario", choices=["rlc_case1", "motor_synthetic"], default="rlc_case1")
    compare.add_argument("--variants", default="wbeif,femeif_all")
    compare.add_argument("--seeds", default="0,1,2")
    compare.add_argument("--generations", type=int, default=40)
    compare.add_argument("--population", type=int, default=80)
    compare.add_argument("--scenario-seed", type=int, default=0)
    compare.add_argument("--alpha", type=float, default=25.12)
    compare.add_argument("--noise-mag-db", type=float, default=0.0)
    compare.add_argument("--noise-phase-rad", type=float, default=0.0)
    compare.add_argument("--out", default="output/compare")
    compare.set_defaults(func=run_compare)

    showcase = subparsers.add_parser("showcase", help="Run showcase experiments and generate mentor-facing plots.")
    showcase.add_argument("--scenario", choices=["rlc_case1", "motor_synthetic"], default="rlc_case1")
    showcase.add_argument("--variants", default="wbeif,femeif_dtw,femeif_all")
    showcase.add_argument("--seeds", default="0,1,2,3,4")
    showcase.add_argument("--generations", type=int, default=20)
    showcase.add_argument("--population", type=int, default=50)
    showcase.add_argument("--scenario-seed", type=int, default=0)
    showcase.add_argument("--alpha", type=float, default=25.12)
    showcase.add_argument("--noise-mag-db", type=float, default=0.0)
    showcase.add_argument("--noise-phase-rad", type=float, default=0.0)
    showcase.add_argument("--out", default="output/teacher-case1")
    showcase.set_defaults(func=run_showcase_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
