"""Depth-backed planning for the local VLM command path."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .geometry import build_roi_point_cloud, transform_points
from .models import RgbdFrame, TaskCommand
from .routing import ArmRoute, route_point


@dataclass(frozen=True)
class LocalPickPlacePlan:
    command: TaskCommand
    source_xyz: tuple[float, float, float]
    destination_xyz: tuple[float, float, float]
    arm: ArmRoute


def _roi_center_point(frame: RgbdFrame, roi, *, min_valid_ratio: float) -> np.ndarray:
    cloud = build_roi_point_cloud(frame, roi)
    area = (roi.x1 - roi.x0) * (roi.y1 - roi.y0)
    if len(cloud) < area * min_valid_ratio:
        raise ValueError("ROI has insufficient valid depth")
    return np.median(cloud[:, :3], axis=0)


def plan_local_task(
    frame: RgbdFrame,
    command: TaskCommand,
    camera_to_base: np.ndarray,
    arms: Sequence[ArmRoute],
    *,
    center_exclusion_half_width: float,
    min_confidence: float = 0.6,
    min_valid_depth_ratio: float = 0.3,
) -> LocalPickPlacePlan:
    if command.source.confidence < min_confidence or command.destination.confidence < min_confidence:
        raise ValueError("source and destination confidence are below the configured threshold")
    source_camera = _roi_center_point(frame, command.source.roi, min_valid_ratio=min_valid_depth_ratio)
    destination_camera = _roi_center_point(
        frame, command.destination.roi, min_valid_ratio=min_valid_depth_ratio
    )
    source = transform_points(source_camera[None, :], camera_to_base)[0]
    destination = transform_points(destination_camera[None, :], camera_to_base)[0]
    arm = next((item for item in arms if item.name == command.arm), None) if command.arm else None
    if arm is None:
        arm = route_point(source, arms, center_exclusion_half_width)
    elif not arm.workspace.contains(source):
        raise ValueError(f"source point is outside requested arm workspace: {arm.name}")
    if not arm.workspace.contains(destination):
        raise ValueError(f"destination point is outside selected arm workspace: {arm.name}")
    return LocalPickPlacePlan(
        command=command,
        source_xyz=tuple(float(value) for value in source),
        destination_xyz=tuple(float(value) for value in destination),
        arm=arm,
    )
