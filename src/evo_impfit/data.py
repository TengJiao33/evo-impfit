"""Scenario and target-data construction."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Callable

import numpy as np

from .circuits import motor_drive_common_mode_like, rlc_parallel_series


ModelFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    lower: float
    upper: float


@dataclass(frozen=True)
class Scenario:
    name: str
    frequencies_hz: np.ndarray
    target_z: np.ndarray
    specs: tuple[ParameterSpec, ...]
    model: ModelFn
    true_values: np.ndarray | None = None
    notes: str = ""


def rlc6_specs() -> tuple[ParameterSpec, ...]:
    specs: list[ParameterSpec] = []
    for i in range(1, 7):
        specs.extend(
            [
                ParameterSpec(f"R{i}", 1.0, 1e5),
                ParameterSpec(f"C{i}", 1e-9, 1e-4),
                ParameterSpec(f"L{i}", 1e-7, 1e-2),
            ]
        )
    return tuple(specs)


def motor_like_specs() -> tuple[ParameterSpec, ...]:
    return tuple(
        [ParameterSpec(f"R{i}", 1e-2, 10**4.5) for i in range(1, 8)]
        + [ParameterSpec(f"C{i}", 1e-12, 1e-7) for i in range(1, 5)]
        + [ParameterSpec(f"L{i}", 1e-9, 1e-4) for i in range(1, 5)]
    )


def _maybe_add_noise(
    target_z: np.ndarray,
    noise_mag_db: float,
    noise_phase_rad: float,
    seed: int,
) -> np.ndarray:
    if noise_mag_db <= 0 and noise_phase_rad <= 0:
        return target_z

    rng = np.random.default_rng(seed)
    mag_db = 20.0 * np.log10(np.abs(target_z))
    phase = np.unwrap(np.angle(target_z))
    mag_db = mag_db + rng.normal(0.0, noise_mag_db, size=mag_db.shape)
    phase = phase + rng.normal(0.0, noise_phase_rad, size=phase.shape)
    return 10.0 ** (mag_db / 20.0) * np.exp(1j * phase)


def rlc_case1(
    n_points: int = 180,
    noise_mag_db: float = 0.0,
    noise_phase_rad: float = 0.0,
    seed: int = 0,
) -> Scenario:
    """Reconstruct the paper's generic six-block RLC-parallel case."""

    frequencies = np.logspace(2.0, 6.0, n_points)
    true_values = np.array(
        [
            2000.0,
            10e-6,
            1585e-6,
            3000.0,
            3.98e-6,
            10e-6,
            500.0,
            1e-6,
            100e-6,
            1200.0,
            0.631e-6,
            2.512e-6,
            800.0,
            0.501e-6,
            1.259e-6,
            50.0,
            0.158e-6,
            0.251e-6,
        ],
        dtype=float,
    )
    target = _maybe_add_noise(
        rlc_parallel_series(frequencies, true_values),
        noise_mag_db,
        noise_phase_rad,
        seed,
    )
    return Scenario(
        name="rlc_case1",
        frequencies_hz=frequencies,
        target_z=target,
        specs=rlc6_specs(),
        model=rlc_parallel_series,
        true_values=true_values,
        notes="Synthetic reconstruction of the paper's generic RLC case.",
    )


def motor_synthetic(
    n_points: int = 96,
    noise_mag_db: float = 0.0,
    noise_phase_rad: float = 0.0,
    seed: int = 0,
) -> Scenario:
    """Motor-drive-like synthetic scenario for private-data-free testing."""

    frequencies = np.logspace(4.0, 7.4, n_points)
    true_values = np.array(
        [
            4.7,
            160.0,
            14.0,
            75.0,
            33.0,
            820.0,
            12.0,
            160e-12,
            47e-12,
            18e-12,
            9.1e-12,
            180e-9,
            820e-9,
            3.3e-6,
            95e-9,
        ],
        dtype=float,
    )

    target = _maybe_add_noise(
        motor_drive_common_mode_like(frequencies, true_values),
        noise_mag_db,
        noise_phase_rad,
        seed,
    )
    return Scenario(
        name="motor_synthetic",
        frequencies_hz=frequencies,
        target_z=target,
        specs=motor_like_specs(),
        model=motor_drive_common_mode_like,
        true_values=true_values,
        notes="Synthetic motor-drive-like target; not a replication of the private bench trace.",
    )


def make_scenario(name: str, **kwargs: object) -> Scenario:
    if name == "rlc_case1":
        return rlc_case1(**kwargs)
    if name == "motor_synthetic":
        return motor_synthetic(**kwargs)
    raise ValueError(f"Unknown scenario: {name}")


def load_target_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load measured target data from CSV.

    Supported headers:
      frequency_hz,real,imag
      frequency_hz,magnitude_db,phase_rad
    """

    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {path}.")

    frequencies = np.array([float(row["frequency_hz"]) for row in rows], dtype=float)
    headers = set(rows[0])
    if {"real", "imag"} <= headers:
        z = np.array([complex(float(row["real"]), float(row["imag"])) for row in rows])
    elif {"magnitude_db", "phase_rad"} <= headers:
        mag = np.array([float(row["magnitude_db"]) for row in rows])
        phase = np.array([float(row["phase_rad"]) for row in rows])
        z = 10.0 ** (mag / 20.0) * np.exp(1j * phase)
    else:
        raise ValueError("CSV must contain real/imag or magnitude_db/phase_rad columns.")

    order = np.argsort(frequencies)
    return frequencies[order], z[order]


def scenario_from_csv(
    path: str | Path,
    model_preset: str,
    name: str | None = None,
) -> Scenario:
    frequencies, target_z = load_target_csv(path)
    if model_preset == "rlc6":
        model = rlc_parallel_series
        specs = rlc6_specs()
        notes = "CSV target fitted with six-block RLC-parallel model."
    elif model_preset == "motor_like":
        model = motor_drive_common_mode_like
        specs = motor_like_specs()
        notes = "CSV target fitted with motor-drive-like model preset."
    else:
        raise ValueError("model_preset must be one of: rlc6, motor_like")

    return Scenario(
        name=name or Path(path).stem,
        frequencies_hz=frequencies,
        target_z=target_z,
        specs=specs,
        model=model,
        true_values=None,
        notes=notes,
    )
