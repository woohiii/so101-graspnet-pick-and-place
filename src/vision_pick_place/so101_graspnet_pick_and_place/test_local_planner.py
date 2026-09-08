from __future__ import annotations

import numpy as np
import pytest

from .local_planner import plan_local_task
from .models import CameraIntrinsics, DetectedObject, RgbdFrame, TaskCommand
from .routing import ArmRoute, WorkspaceBounds


def _frame() -> RgbdFrame:
    return RgbdFrame(
        rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        depth=np.full((10, 10), 500, dtype=np.uint16),
        intrinsics=CameraIntrinsics(100, 100, 0, 0),
    )


def _command() -> TaskCommand:
    return TaskCommand(
        source=DetectedObject("red block", (1, 1, 5, 5), 0.9, "object"),
        destination=DetectedObject("blue tray", (5, 5, 9, 9), 0.9, "destination"),
    )


def test_plan_local_task_backprojects_both_regions_and_routes_arm() -> None:
    transform = np.eye(4)
    transform[0, 3] = -0.2
    arms = [ArmRoute("left", WorkspaceBounds(-1, -0.05, -1, 1, 0, 1), (0, 0, 0))]

    plan = plan_local_task(
        _frame(), _command(), transform, arms, center_exclusion_half_width=0.02, min_confidence=0.6
    )

    assert plan.arm.name == "left"
    assert plan.source_xyz[2] == pytest.approx(0.5)
    assert plan.destination_xyz[2] == pytest.approx(0.5)


def test_plan_local_task_rejects_low_confidence_detection() -> None:
    command = TaskCommand(
        source=DetectedObject("block", (1, 1, 5, 5), 0.5),
        destination=DetectedObject("tray", (5, 5, 9, 9), 0.9, "destination"),
    )
    arms = [ArmRoute("left", WorkspaceBounds(-1, -0.05, -1, 1, 0, 1), (0, 0, 0))]

    with pytest.raises(ValueError, match="confidence"):
        plan_local_task(_frame(), command, np.eye(4), arms, center_exclusion_half_width=0.02, min_confidence=0.6)

