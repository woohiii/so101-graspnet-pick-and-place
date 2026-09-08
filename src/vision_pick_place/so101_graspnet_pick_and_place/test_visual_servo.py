"""Tests for bounded XY visual-servo correction."""

from __future__ import annotations

import numpy as np
import pytest

from custom_scripts.vision_pick_place.so101_graspnet_pick_and_place.visual_servo import (
    compute_xy_nudge,
)


def test_compute_xy_nudge_transforms_target_minus_current() -> None:
    transform = np.eye(4)
    current = np.array([0.1, 0.2, 0.6])
    target = np.array([0.13, 0.16, 0.6])

    assert compute_xy_nudge(current, target, transform, max_nudge_m=0.05) == pytest.approx((0.03, -0.04))


def test_compute_xy_nudge_clamps_large_correction() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        compute_xy_nudge(np.zeros(3), np.array([0.1, 0.0, 0.0]), np.eye(4), max_nudge_m=0.04)
