"""Tests for markerless depth-pixel selection."""

from __future__ import annotations

import numpy as np
import pytest

from custom_scripts.vision_pick_place.so101_graspnet_pick_and_place.collect_handeye_points import (
    auto_select_depth_pixel,
    build_target_depth_pixels,
    find_nearest_valid_depth_pixel,
    resolve_depth_pixel,
    validate_joint_points,
)


def test_auto_select_depth_pixel_returns_nearest_valid_cluster_center() -> None:
    depth = np.full((10, 10), 800, dtype=np.uint16)
    depth[2:4, 6:8] = 400

    u, v = auto_select_depth_pixel(depth)

    assert (u, v) == (6, 2)


def test_auto_select_depth_pixel_rejects_empty_depth() -> None:
    with pytest.raises(RuntimeError, match="valid depth"):
        auto_select_depth_pixel(np.zeros((4, 4), dtype=np.uint16))


def test_find_nearest_valid_depth_pixel_expands_search_for_occlusion() -> None:
    depth = np.zeros((40, 40), dtype=np.uint16)
    depth[20, 30] = 500

    assert find_nearest_valid_depth_pixel(depth, 20, 20, radius=6, max_radius=16) == (30, 20)


def test_resolve_depth_pixel_rejects_occluded_tcp_by_default() -> None:
    depth = np.zeros((40, 40), dtype=np.uint16)
    depth[20, 30] = 500

    assert resolve_depth_pixel(depth, 20, 20) is None
    assert resolve_depth_pixel(depth, 20, 20, allow_fallback=True) == (30, 20)


def test_build_target_depth_pixels_supports_more_than_eight_points() -> None:
    points = build_target_depth_pixels(12)

    assert len(points) == 12
    assert all(0 <= u < 320 and 0 <= v < 240 for u, v in points)


def test_validate_joint_points_requires_finite_matrix() -> None:
    joints = validate_joint_points(np.zeros((6, 5)), expected_count=6)

    assert joints.shape == (6, 5)
    with pytest.raises(ValueError, match="joint points"):
        validate_joint_points(np.zeros((5, 5)), expected_count=6)
