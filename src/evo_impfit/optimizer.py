"""Evolutionary optimizers for impedance fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .data import Scenario
from .features import FeatureConfig, evaluate_objectives, weighted_mse


Mode = Literal["femeif", "wbeif"]


@dataclass(frozen=True)
class Candidate:
    genome: np.ndarray
    values: np.ndarray
    objectives: dict[str, float]
    loss: float


@dataclass(frozen=True)
class FittingResult:
    scenario_name: str
    mode: Mode
    objective_names: tuple[str, ...]
    best_values: np.ndarray
    best_loss: float
    best_objectives: dict[str, float]
    history: list[dict[str, float]]
    best_fit_z: np.ndarray


def _bounds_arrays(scenario: Scenario) -> tuple[np.ndarray, np.ndarray]:
    lower = np.array([spec.lower for spec in scenario.specs], dtype=float)
    upper = np.array([spec.upper for spec in scenario.specs], dtype=float)
    return lower, upper


def _decode(genomes: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    log_lower = np.log10(lower)
    log_upper = np.log10(upper)
    return 10.0 ** (log_lower + genomes * (log_upper - log_lower))


def _evaluate_population(
    genomes: np.ndarray,
    scenario: Scenario,
    objective_names: tuple[str, ...],
    cfg: FeatureConfig,
) -> list[Candidate]:
    lower, upper = _bounds_arrays(scenario)
    decoded = _decode(genomes, lower, upper)
    population: list[Candidate] = []
    for genome, values in zip(genomes, decoded):
        fit_z = scenario.model(scenario.frequencies_hz, values)
        objectives = evaluate_objectives(
            objective_names,
            scenario.frequencies_hz,
            scenario.target_z,
            fit_z,
            cfg,
        )
        loss = weighted_mse(scenario.target_z, fit_z, cfg.alpha)
        population.append(
            Candidate(
                genome=np.array(genome, copy=True),
                values=np.array(values, copy=True),
                objectives=objectives,
                loss=loss,
            )
        )
    return population


def _dominates(left: Candidate, right: Candidate, objectives: tuple[str, ...]) -> bool:
    left_values = np.array([left.objectives[name] for name in objectives])
    right_values = np.array([right.objectives[name] for name in objectives])
    return bool(np.all(left_values >= right_values) and np.any(left_values > right_values))


def _nondominated_sort(
    population: list[Candidate],
    objectives: tuple[str, ...],
) -> list[list[int]]:
    dominates_map: list[list[int]] = [[] for _ in population]
    dominated_count = [0 for _ in population]
    fronts: list[list[int]] = [[]]

    for i, left in enumerate(population):
        for j, right in enumerate(population):
            if i == j:
                continue
            if _dominates(left, right, objectives):
                dominates_map[i].append(j)
            elif _dominates(right, left, objectives):
                dominated_count[i] += 1
        if dominated_count[i] == 0:
            fronts[0].append(i)

    cursor = 0
    while cursor < len(fronts) and fronts[cursor]:
        next_front: list[int] = []
        for i in fronts[cursor]:
            for j in dominates_map[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        if next_front:
            fronts.append(next_front)
        cursor += 1

    return fronts


def _crowding_distance(
    population: list[Candidate],
    front: list[int],
    objectives: tuple[str, ...],
) -> dict[int, float]:
    if not front:
        return {}
    distance = {index: 0.0 for index in front}
    if len(front) <= 2:
        return {index: float("inf") for index in front}

    for objective in objectives:
        sorted_front = sorted(front, key=lambda index: population[index].objectives[objective])
        distance[sorted_front[0]] = float("inf")
        distance[sorted_front[-1]] = float("inf")
        min_value = population[sorted_front[0]].objectives[objective]
        max_value = population[sorted_front[-1]].objectives[objective]
        span = max(max_value - min_value, 1e-12)
        for pos in range(1, len(sorted_front) - 1):
            prev_value = population[sorted_front[pos - 1]].objectives[objective]
            next_value = population[sorted_front[pos + 1]].objectives[objective]
            distance[sorted_front[pos]] += (next_value - prev_value) / span

    return distance


def _select_nsga2(
    population: list[Candidate],
    size: int,
    objectives: tuple[str, ...],
) -> list[Candidate]:
    selected: list[Candidate] = []
    for front in _nondominated_sort(population, objectives):
        if len(selected) + len(front) <= size:
            selected.extend(population[index] for index in front)
            continue
        distances = _crowding_distance(population, front, objectives)
        remaining = size - len(selected)
        selected.extend(
            population[index]
            for index in sorted(front, key=lambda i: distances[i], reverse=True)[:remaining]
        )
        break
    return selected


def _select_wbeif(population: list[Candidate], size: int) -> list[Candidate]:
    return sorted(population, key=lambda candidate: candidate.loss)[:size]


def _make_offspring(
    parents: list[Candidate],
    rng: np.random.Generator,
    population_size: int,
    mutation_rate: float,
    mutation_scale: float,
) -> np.ndarray:
    genomes = np.array([candidate.genome for candidate in parents])
    offspring: list[np.ndarray] = []
    while len(offspring) < population_size:
        parent_a, parent_b = genomes[rng.integers(0, len(genomes), size=2)]
        blend = rng.random(parent_a.shape)
        child_a = blend * parent_a + (1.0 - blend) * parent_b
        child_b = blend * parent_b + (1.0 - blend) * parent_a
        for child in (child_a, child_b):
            mask = rng.random(child.shape) < mutation_rate
            child = np.array(child, copy=True)
            child[mask] += rng.normal(0.0, mutation_scale, size=int(np.sum(mask)))
            offspring.append(np.clip(child, 0.0, 1.0))
            if len(offspring) >= population_size:
                break
    return np.array(offspring)


def run_fitting(
    scenario: Scenario,
    mode: Mode = "femeif",
    objective_names: tuple[str, ...] = ("wsmse", "ptc", "dtw", "lfp", "ss", "lmep"),
    generations: int = 60,
    population_size: int = 100,
    seed: int = 0,
    cfg: FeatureConfig | None = None,
    mutation_rate: float | None = None,
    mutation_scale: float = 0.08,
) -> FittingResult:
    if cfg is None:
        cfg = FeatureConfig()
    if "wsmse" not in objective_names:
        objective_names = ("wsmse",) + tuple(objective_names)
    if mode == "wbeif":
        objective_names = ("wsmse",)
    if mutation_rate is None:
        mutation_rate = 1.0 / len(scenario.specs)

    rng = np.random.default_rng(seed)
    genomes = rng.random((population_size, len(scenario.specs)))
    population = _evaluate_population(genomes, scenario, objective_names, cfg)
    history: list[dict[str, float]] = []

    for generation in range(generations + 1):
        best = min(population, key=lambda candidate: candidate.loss)
        history.append(
            {
                "generation": float(generation),
                "best_wsmse": best.loss,
                "mean_wsmse": float(np.mean([candidate.loss for candidate in population])),
                "best_f_wsmse": best.objectives["wsmse"],
            }
        )
        if generation == generations:
            break

        offspring_genomes = _make_offspring(
            population,
            rng,
            population_size,
            mutation_rate,
            mutation_scale,
        )
        offspring = _evaluate_population(offspring_genomes, scenario, objective_names, cfg)
        combined = population + offspring
        if mode == "femeif":
            population = _select_nsga2(combined, population_size, objective_names)
        elif mode == "wbeif":
            population = _select_wbeif(combined, population_size)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    best = min(population, key=lambda candidate: candidate.loss)
    best_fit_z = scenario.model(scenario.frequencies_hz, best.values)
    return FittingResult(
        scenario_name=scenario.name,
        mode=mode,
        objective_names=objective_names,
        best_values=best.values,
        best_loss=best.loss,
        best_objectives=best.objectives,
        history=history,
        best_fit_z=best_fit_z,
    )
