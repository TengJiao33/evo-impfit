"""Equivalent-circuit impedance models."""

from __future__ import annotations

import numpy as np


def rlc_parallel_series(frequencies_hz: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Series connection of parallel RLC blocks.

    The value vector is ordered as R1, C1, L1, R2, C2, L2, ...
    """

    values = np.asarray(values, dtype=float)
    if values.size % 3 != 0:
        raise ValueError("RLC series model expects values ordered as R,C,L blocks.")

    blocks = values.reshape((-1, 3))
    omega = 2.0 * np.pi * np.asarray(frequencies_hz, dtype=float)
    z_total = np.zeros_like(omega, dtype=complex)
    eps = np.finfo(float).tiny

    for resistance, capacitance, inductance in blocks:
        resistance = max(float(resistance), eps)
        capacitance = max(float(capacitance), eps)
        inductance = max(float(inductance), eps)
        admittance = (
            1.0 / resistance
            + 1.0 / (1j * omega * inductance)
            + 1j * omega * capacitance
        )
        z_total += 1.0 / admittance

    return z_total


def motor_drive_common_mode_like(
    frequencies_hz: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Motor-drive-like common-mode equivalent circuit.

    This follows the topology visible in the paper's Fig. 15 at a software
    prototype level: three parallel branches followed by a series C-R-L tail.
    It is not a substitute for the private measured motor-drive dataset.

    Value order:
      R1..R7, C1..C4, L1..L4
    """

    values = np.asarray(values, dtype=float)
    if values.size != 15:
        raise ValueError("Motor-like model expects R1..R7, C1..C4, L1..L4.")

    r = np.maximum(values[:7], np.finfo(float).tiny)
    c = np.maximum(values[7:11], np.finfo(float).tiny)
    l = np.maximum(values[11:15], np.finfo(float).tiny)
    omega = 2.0 * np.pi * np.asarray(frequencies_hz, dtype=float)
    z_total = np.zeros_like(omega, dtype=complex)

    block_specs = [
        (r[0], r[1], c[0], l[0]),
        (r[2], r[3], c[1], l[1]),
        (r[4], r[5], c[2], l[2]),
    ]

    for series_r, shunt_r, capacitance, inductance in block_specs:
        z_cap = 1.0 / (1j * omega * capacitance)
        z_rl = series_r + 1j * omega * inductance
        y_block = 1.0 / z_cap + 1.0 / z_rl + 1.0 / shunt_r
        z_total += 1.0 / y_block

    z_total += 1.0 / (1j * omega * c[3]) + r[6] + 1j * omega * l[3]
    return z_total
