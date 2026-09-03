import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sync_core import (
    SPEED_OF_SOUND_MPS,
    LatencyTracker,
    batch_compensation,
    delay_s,
    receipt_hash,
    seeded_unit_points,
)


def test_delay_zero_when_colocated():
    assert delay_s((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)) == 0.0


def test_delay_343m_is_one_second():
    d = delay_s((0.0, 0.0, 0.0), (SPEED_OF_SOUND_MPS, 0.0, 0.0))
    assert math.isclose(d, 1.0, rel_tol=1e-12)


def test_batch_shape_and_determinism():
    players = seeded_unit_points(4, 3, b"players")
    events = seeded_unit_points(6, 3, b"events")
    d1, a1 = batch_compensation(players, events)
    d2, a2 = batch_compensation(players, events)
    assert d1 == d2 and a1 == a2
    assert len(d1) == 4 and len(d1[0]) == 6
    assert receipt_hash(d1, a1) == receipt_hash(d2, a2)


def test_latency_tracker_ignores_nonpositive():
    t = LatencyTracker()
    before = t.ewma_ms
    t.update(0.0)
    t.update(-3.0)
    assert t.ewma_ms == before


def test_latency_tracker_moves_toward_measurement():
    t = LatencyTracker(ewma_ms=50.0, last_measured=50.0)
    t.update(80.0)
    assert t.ewma_ms > 50.0
    assert t.ewma_ms < 80.0
    assert 0.06 <= t.alpha <= 0.42
