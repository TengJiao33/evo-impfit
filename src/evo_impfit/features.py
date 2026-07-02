"""Feature objectives from the FEMEIF paper, implemented for fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class FeatureConfig:
    alpha: float = 25.12
    slope_threshold: float = 8.0
    shape_beta: float = 0.4
    shape_mu: float = 0.5
    dtw_points: int = 60
    lfp_threshold: float = 1.6
    ptc_gamma: float = 0.41


def magnitude_db(z: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(z), EPS))


def phase_rad(z: np.ndarray) -> np.ndarray:
    return np.unwrap(np.angle(z))


def weighted_mse(target_z: np.ndarray, fit_z: np.ndarray, alpha: float) -> float:
    mag_err = magnitude_db(target_z) - magnitude_db(fit_z)
    phase_err = phase_rad(target_z) - phase_rad(fit_z)
    return float(np.mean(mag_err * mag_err + alpha * phase_err * phase_err))


def wsmse_fitness(target_z: np.ndarray, fit_z: np.ndarray, cfg: FeatureConfig) -> float:
    return 1.0 / (weighted_mse(target_z, fit_z, cfg.alpha) + EPS)


def _shape_pattern(values: np.ndarray, log_freq: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    slopes = np.diff(values) / np.maximum(np.diff(log_freq), EPS)
    if slopes.size < 2:
        return slopes

    signs = np.where(
        slopes > cfg.slope_threshold,
        1.0,
        np.where(slopes < -cfg.slope_threshold, -1.0, 0.0),
    )
    accel = np.diff(slopes, prepend=slopes[0])
    smooth_accel = np.tanh(cfg.shape_mu * accel / (abs(cfg.slope_threshold) + EPS))
    return signs * (1.0 + cfg.shape_beta * smooth_accel)


def shape_similarity_fitness(
    frequencies_hz: np.ndarray,
    target_z: np.ndarray,
    fit_z: np.ndarray,
    cfg: FeatureConfig,
) -> float:
    log_freq = np.log10(frequencies_hz)
    target_pattern = _shape_pattern(magnitude_db(target_z), log_freq, cfg)
    fit_pattern = _shape_pattern(magnitude_db(fit_z), log_freq, cfg)
    return 1.0 / (float(np.mean(np.abs(target_pattern - fit_pattern))) + EPS)


def _local_extrema(values: np.ndarray) -> list[tuple[int, int]]:
    extrema: list[tuple[int, int]] = []
    for index in range(1, len(values) - 1):
        prev_value = values[index - 1]
        value = values[index]
        next_value = values[index + 1]
        if value > prev_value and value > next_value:
            extrema.append((index, 1))
        elif value < prev_value and value < next_value:
            extrema.append((index, -1))
    return extrema


def _neighbor_bounds(log_freq: np.ndarray, target_extrema: list[tuple[int, int]]) -> list[tuple[float, float]]:
    centers = [log_freq[index] for index, _ in target_extrema]
    bounds: list[tuple[float, float]] = []
    for pos, center in enumerate(centers):
        low = log_freq[0] if pos == 0 else 0.5 * (centers[pos - 1] + center)
        high = log_freq[-1] if pos == len(centers) - 1 else 0.5 * (center + centers[pos + 1])
        bounds.append((low, high))
    return bounds


def peak_trough_consistency_fitness(
    frequencies_hz: np.ndarray,
    target_z: np.ndarray,
    fit_z: np.ndarray,
    cfg: FeatureConfig,
) -> float:
    log_freq = np.log10(frequencies_hz)
    target_extrema = _local_extrema(magnitude_db(target_z))
    fit_extrema = _local_extrema(magnitude_db(fit_z))
    if not target_extrema:
        return 1.0

    bounds = _neighbor_bounds(log_freq, target_extrema)
    domain = max(float(log_freq[-1] - log_freq[0]), EPS)
    penalty = 0.0
    for (target_index, target_kind), (low, high) in zip(target_extrema, bounds):
        candidates = [
            index
            for index, kind in fit_extrema
            if kind == target_kind and low <= log_freq[index] <= high
        ]
        if not candidates:
            penalty += cfg.ptc_gamma
            continue

        best_distance = min(abs(log_freq[index] - log_freq[target_index]) for index in candidates)
        penalty += float(np.exp(cfg.ptc_gamma * best_distance / domain) - 1.0)

    return 1.0 / (penalty + EPS)


def _downsample(values: np.ndarray, n_points: int) -> np.ndarray:
    if values.size <= n_points:
        return values
    x_old = np.linspace(0.0, 1.0, values.size)
    x_new = np.linspace(0.0, 1.0, n_points)
    return np.interp(x_new, x_old, values)


def dtw_fitness(target_z: np.ndarray, fit_z: np.ndarray, cfg: FeatureConfig) -> float:
    target = _downsample(magnitude_db(target_z), cfg.dtw_points)
    fit = _downsample(magnitude_db(fit_z), cfg.dtw_points)
    target = (target - np.mean(target)) / (np.std(target) + EPS)
    fit = (fit - np.mean(fit)) / (np.std(fit) + EPS)

    n_target = target.size
    n_fit = fit.size
    dp = np.full((n_target + 1, n_fit + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n_target + 1):
        for j in range(1, n_fit + 1):
            cost = abs(target[i - 1] - fit[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return 1.0 / (float(dp[n_target, n_fit]) + EPS)


def _zones_from_target(frequencies_hz: np.ndarray, target_z: np.ndarray) -> list[np.ndarray]:
    log_freq = np.log10(frequencies_hz)
    extrema = _local_extrema(magnitude_db(target_z))
    if not extrema:
        return [np.arange(len(frequencies_hz))]

    bounds = _neighbor_bounds(log_freq, extrema)
    zones: list[np.ndarray] = []
    for low, high in bounds:
        indices = np.flatnonzero((log_freq >= low) & (log_freq <= high))
        if indices.size:
            zones.append(indices)
    return zones


def low_frequency_priority_fitness(
    frequencies_hz: np.ndarray,
    target_z: np.ndarray,
    fit_z: np.ndarray,
    cfg: FeatureConfig,
) -> float:
    zones = _zones_from_target(frequencies_hz, target_z)
    score = 0.0
    for indices in zones:
        zone_loss = weighted_mse(target_z[indices], fit_z[indices], cfg.alpha)
        if zone_loss <= cfg.lfp_threshold:
            score += 1.0
        else:
            break
    return score + EPS


def low_max_error_preference_fitness(
    target_z: np.ndarray,
    fit_z: np.ndarray,
    cfg: FeatureConfig,
) -> float:
    mag_err = magnitude_db(target_z) - magnitude_db(fit_z)
    phase_err = phase_rad(target_z) - phase_rad(fit_z)
    point_error = np.sqrt(mag_err * mag_err + cfg.alpha * phase_err * phase_err)
    return 1.0 / (float(np.max(point_error)) + EPS)


def evaluate_objectives(
    objective_names: tuple[str, ...],
    frequencies_hz: np.ndarray,
    target_z: np.ndarray,
    fit_z: np.ndarray,
    cfg: FeatureConfig,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in objective_names:
        if name == "wsmse":
            values[name] = wsmse_fitness(target_z, fit_z, cfg)
        elif name == "ss":
            values[name] = shape_similarity_fitness(frequencies_hz, target_z, fit_z, cfg)
        elif name == "ptc":
            values[name] = peak_trough_consistency_fitness(frequencies_hz, target_z, fit_z, cfg)
        elif name == "dtw":
            values[name] = dtw_fitness(target_z, fit_z, cfg)
        elif name == "lfp":
            values[name] = low_frequency_priority_fitness(frequencies_hz, target_z, fit_z, cfg)
        elif name == "lmep":
            values[name] = low_max_error_preference_fitness(target_z, fit_z, cfg)
        else:
            raise ValueError(f"Unknown objective: {name}")
    return values
