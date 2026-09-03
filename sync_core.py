"""AetherSync deterministic compensation core.

Stdlib-only. JAX/Cirq paths in run_engine.py and quantum_accel.py remain
optional accelerators. This module is the Chromebook-runnable contract:

- Euclidean delay at 343 m/s
- Inverse-square attenuation with epsilon floor
- EWMA latency tracker matching math_core.zig semantics
- Deterministic PRNG via hash seed (no wall-clock)
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

SPEED_OF_SOUND_MPS = 343.0
EPS = 1e-6


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def delay_s(player: Sequence[float], event: Sequence[float]) -> float:
    return _dist(player, event) / SPEED_OF_SOUND_MPS


def attenuation(intensity: float, player: Sequence[float], event: Sequence[float]) -> float:
    d = _dist(player, event)
    return intensity / (d * d + EPS)


def batch_compensation(
    players: Sequence[Sequence[float]],
    events: Sequence[Sequence[float]],
    intensities: Sequence[float] | None = None,
) -> Tuple[List[List[float]], List[List[float]]]:
    if intensities is None:
        intensities = [1.0] * len(events)
    if len(intensities) != len(events):
        raise ValueError("intensities length must match events")
    delays: List[List[float]] = []
    atts: List[List[float]] = []
    for p in players:
        row_d: List[float] = []
        row_a: List[float] = []
        for e, inten in zip(events, intensities):
            row_d.append(delay_s(p, e))
            row_a.append(attenuation(inten, p, e))
        delays.append(row_d)
        atts.append(row_a)
    return delays, atts


@dataclass
class LatencyTracker:
    ewma_ms: float = 50.0
    jitter_ms: float = 0.0
    alpha: float = 0.18
    last_measured: float = 50.0
    predicted_next: float = 50.0

    def update(self, measured: float) -> None:
        if measured <= 0:
            return
        err = abs(measured - self.ewma_ms)
        self.jitter_ms = 0.25 * err + 0.75 * self.jitter_ms
        cr = abs(measured - self.last_measured)
        da = 0.18 * (1.0 + cr / 25.0)
        da = max(0.06, min(0.42, da))
        self.alpha = da
        self.predicted_next = 0.7 * self.ewma_ms + 0.3 * measured
        self.ewma_ms = da * measured + (1.0 - da) * self.ewma_ms
        self.last_measured = measured


def seeded_unit_points(n: int, dim: int, salt: bytes) -> List[List[float]]:
    """Deterministic points in [0,1)^dim from SHA-256 stream."""
    out: List[List[float]] = []
    counter = 0
    while len(out) < n:
        block = hashlib.sha256(salt + counter.to_bytes(8, "big")).digest()
        counter += 1
        # consume 4-byte little chunks as U32 / 2^32
        i = 0
        pt: List[float] = []
        while i + 4 <= len(block) and len(pt) < dim:
            u = int.from_bytes(block[i : i + 4], "big") / 2**32
            pt.append(u)
            i += 4
        if len(pt) == dim:
            out.append(pt)
    return out


def receipt_hash(delays: Iterable[Iterable[float]], atts: Iterable[Iterable[float]]) -> str:
    h = hashlib.sha256()
    h.update(b"AETHERSYNC_V1")
    for row in delays:
        for v in row:
            h.update(f"{v:.12f}".encode("ascii"))
    for row in atts:
        for v in row:
            h.update(f"{v:.12f}".encode("ascii"))
    return h.hexdigest()
