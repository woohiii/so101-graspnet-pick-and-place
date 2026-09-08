"""Shared, measured safe-start pose for the left SO-101 follower arm."""

import numpy as np

HOME_JOINTS_DEG = np.array([-8.44, 10.29, 4.27, 97.67, 3.74, 97.37], dtype=float)


def home_pose_error_deg(joints_deg: np.ndarray) -> float:
    """Return the largest absolute joint error from the recorded safe pose."""
    joints = np.asarray(joints_deg, dtype=float)
    if joints.shape != HOME_JOINTS_DEG.shape or not np.isfinite(joints).all():
        raise ValueError("expected six finite joint positions")
    return float(np.max(np.abs(joints - HOME_JOINTS_DEG)))


def is_at_home(joints_deg: np.ndarray, tolerance_deg: float) -> bool:
    if tolerance_deg <= 0:
        raise ValueError("home tolerance must be positive")
    return home_pose_error_deg(joints_deg) <= tolerance_deg
