from __future__ import annotations

from .local_cli import format_preview
from .local_planner import LocalPickPlacePlan
from .models import DetectedObject, TaskCommand
from .routing import ArmRoute, WorkspaceBounds


def test_format_preview_contains_plan_and_explicit_confirmation() -> None:
    source = DetectedObject("red block", (1, 2, 4, 5), 0.9)
    destination = DetectedObject("blue tray", (5, 6, 9, 9), 0.8, "destination")
    plan = LocalPickPlacePlan(
        TaskCommand(source, destination),
        (0.1, 0.2, 0.3),
        (0.2, 0.1, 0.4),
        ArmRoute("left", WorkspaceBounds(-1, -0.05, -1, 1, 0, 1), (0, 0, 0)),
    )

    preview = format_preview(plan)

    assert "red block" in preview
    assert "blue tray" in preview
    assert "left" in preview
    assert "yes" in preview
    assert "motor" in preview.lower()
