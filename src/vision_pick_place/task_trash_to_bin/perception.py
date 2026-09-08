"""Click-based coordinate estimation for an arbitrary (color/shape-agnostic)
trash object.

2026-09-08: trimmed copy of ../task_red_cube_to_bin_new_gripper/perception.py.
No HSV color/shape detection here - there is no generic detector for an
unknown-shaped trash object, so this task only supports a user-clicked
pixel (click_grasp_trash.py), not a wrist-cam closed-loop servo.

Coordinate estimation is NOT a full pixel+depth -> camera-frame 3D point ->
base-frame transform via a proper 6-DOF T_cam_to_base extrinsic - see the
sibling task's perception.py docstring for the full reasoning. What's used:
  - xy: a 2D homography (Astra RGB pixel -> robot-base xy on the table
    plane), from calibrate_camera.py's touch-point calibration
    (homography.json). Only valid for objects sitting on the table plane.
  - z: a HEIGHT DELTA at the clicked pixel (table depth reading minus the
    clicked patch's depth reading, both from Astra), added on top of the
    independently-measured TABLE_Z. No camera-to-base extrinsic needed.
This is a coarse guess only - task_state_machine.descend_and_grasp()'s
contact detection during descent, not this estimate's absolute accuracy, is
what actually decides when the gripper has reached the object.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np

try:
    from . import config
except ImportError:
    import config


RGBD_PREVIEW_SCALE = 2


@dataclass
class Detection:
    cx: float  # pixel x of the object's centroid
    cy: float  # pixel y of the object's centroid
    area: float
    bbox: tuple[int, int, int, int]  # x, y, w, h


class PublishedFrameSource:
    """Reads whatever camera_hub.py/astra_s_live.py last published to `path`
    instead of opening the camera device itself (two processes can't both
    hold a UVC/OpenNI2 device open for streaming). Same cv2.VideoCapture-
    shaped isOpened()/read()/release() so it can substitute for one."""

    def __init__(self, path: str, stale_timeout_s: float = config.FRAME_STALE_TIMEOUT_S):
        self.path = path
        self.stale_timeout_s = stale_timeout_s

    def _fresh(self) -> bool:
        return os.path.exists(self.path) and (time.time() - os.path.getmtime(self.path)) < self.stale_timeout_s

    def isOpened(self) -> bool:
        return self._fresh()

    def read(self):
        if not self._fresh():
            return False, None
        frame = cv2.imread(self.path)
        return (frame is not None), frame

    def release(self) -> None:
        pass


def load_fresh_depth(depth_path: str = config.ASTRA_DEPTH_MM_PATH) -> np.ndarray | None:
    if not os.path.exists(depth_path) or (time.time() - os.path.getmtime(depth_path)) >= config.FRAME_STALE_TIMEOUT_S:
        return None
    try:
        depth_mm = np.load(depth_path)
    except (OSError, ValueError):
        return None
    return depth_mm if depth_mm.ndim == 2 else None


def rgbd_overlay(bgr: np.ndarray, depth_mm: np.ndarray | None) -> np.ndarray | None:
    if depth_mm is None or bgr.ndim != 3 or not np.any(depth_mm):
        return None
    depth_mm = _apply_registered_depth_offset(depth_mm)
    depth_vis = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
    if bgr.shape[:2] != depth_color.shape[:2]:
        bgr = cv2.resize(bgr, (depth_color.shape[1], depth_color.shape[0]))
    return cv2.addWeighted(bgr, 0.55, depth_color, 0.45, 0)


def _apply_registered_depth_offset(depth_mm: np.ndarray) -> np.ndarray:
    offset_x, offset_y = config.ASTRA_DEPTH_REGISTERED_OFFSET_PX
    if offset_x == 0 and offset_y == 0:
        return depth_mm
    transform = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    return cv2.warpAffine(
        depth_mm,
        transform,
        (depth_mm.shape[1], depth_mm.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def depth_valid_mask(depth_mm: np.ndarray | None, output_shape: tuple[int, ...]) -> np.ndarray | None:
    if depth_mm is None or depth_mm.ndim != 2:
        return None
    valid = _apply_registered_depth_offset(depth_mm) > 0
    # 구조광의 고립된 1~2픽셀 dropout을 메우되 큰 FOV 무효 영역은 보존한다.
    valid = cv2.morphologyEx(
        valid.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8)
    ).astype(bool)
    output_h, output_w = output_shape[:2]
    if valid.shape != (output_h, output_w):
        valid = cv2.resize(valid.astype(np.uint8), (output_w, output_h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return valid


def _depth_registration_active() -> bool:
    status_path = config.ASTRA_DEPTH_REGISTRATION_STATUS_PATH
    if os.path.exists(status_path) and (time.time() - os.path.getmtime(status_path)) < config.FRAME_STALE_TIMEOUT_S:
        try:
            with open(status_path) as f:
                return bool(json.load(f)["registered"])
        except (OSError, ValueError, KeyError):
            pass
    return config.ASTRA_DEPTH_REGISTERED_TO_COLOR


def rgb_px_to_homography_px(px: float, py: float, frame_shape: tuple[int, ...]) -> tuple[float, float]:
    frame_h, frame_w = frame_shape[:2]
    return px * config.FRAME_W / frame_w, py * config.FRAME_H / frame_h


def is_frame_corrupted(bgr_frame: np.ndarray) -> bool:
    """Flags USB frame-tearing (a real, validated issue on the cheap wrist
    UVC camera specifically): (a) a noisy multicolor band - several rows in
    a row with an abnormally large jump from the row above, and (b) a solid
    anomalous color block - a tall run of near-identical rows (an all-
    zero/garbage USB transfer decodes to a flat, often greenish block) whose
    color sits far from the rest of the frame's average."""
    row_means = bgr_frame.mean(axis=1)
    diffs = np.abs(np.diff(row_means, axis=0)).sum(axis=1)
    noisy_band = int((diffs > 25).sum()) >= 4 or bool((diffs > 100).any())

    flat = diffs < 3
    max_run = run = best_start = 0
    for i, f in enumerate(flat):
        if f:
            run += 1
            if run > max_run:
                max_run, best_start = run, i - run + 1
        else:
            run = 0
    h = bgr_frame.shape[0]
    block_color_far = False
    if max_run >= h * 0.12:
        block_mean = row_means[best_start : best_start + max_run].mean(axis=0)
        block_color_far = float(np.abs(block_mean - row_means.mean(axis=0)).sum()) > 60
    return bool(noisy_band or block_color_far)


def _load_homography() -> np.ndarray | None:
    if not config.HOMOGRAPHY_PATH.exists():
        return None
    with open(config.HOMOGRAPHY_PATH) as f:
        data = json.load(f)
    return np.array(data["homography"], dtype=float)


_HOMOGRAPHY = _load_homography()


def _load_calib_robot_bounds() -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Bounding box (x_range, y_range) of homography.json's own touch-taught
    robot_points - see is_xy_within_safe_workspace's docstring for why this
    exists. None if no homography/calib file yet."""
    if not config.HOMOGRAPHY_PATH.exists():
        return None
    with open(config.HOMOGRAPHY_PATH) as f:
        pts = json.load(f).get("robot_points")
    if not pts:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (min(xs), max(xs)), (min(ys), max(ys))


_CALIB_ROBOT_BOUNDS = _load_calib_robot_bounds()

# homography.json only has 4 touch-taught points, so pixel_to_xy's linear
# map is already extrapolating for anything outside that small patch. A
# click far outside the taught region could send the arm toward an
# uncalibrated, possibly joint-limit-straining part of the workspace with no
# other guard catching it before motion starts. Margin is generous (not the
# exact taught box) specifically so real objects near-but-outside the 4
# taught points still get through - this is a sanity gate against a badly
# wrong result, not a tight reachability fence.
WORKSPACE_MARGIN_M = 0.08


def is_xy_within_safe_workspace(x: float, y: float) -> bool:
    """False if (x, y) falls outside the calibrated touch-points' bounding
    box + WORKSPACE_MARGIN_M - callers should refuse to move there rather
    than trust a click that extrapolated this far past anything ever
    physically verified. True (no opinion) if no calibration exists yet,
    same permissive default pixel_to_xy itself falls back to."""
    if _CALIB_ROBOT_BOUNDS is None:
        return True
    (x_lo, x_hi), (y_lo, y_hi) = _CALIB_ROBOT_BOUNDS
    return (
        x_lo - WORKSPACE_MARGIN_M <= x <= x_hi + WORKSPACE_MARGIN_M
        and y_lo - WORKSPACE_MARGIN_M <= y <= y_hi + WORKSPACE_MARGIN_M
    )


def pixel_to_xy(px: float, py: float, homography: np.ndarray | None = _HOMOGRAPHY) -> tuple[float, float] | None:
    """Coarse robot-base (x, y) for an arbitrary Astra-RGB pixel via the
    table-plane homography - see this module's docstring for why this isn't
    a full 3D backprojection. None if there's no homography yet."""
    if homography is None:
        return None
    mapped = homography @ np.array([px, py, 1.0])
    if abs(mapped[2]) < 1e-9:
        return None
    return float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])


def estimate_height_at_px_m(
    px: float,
    py: float,
    rgb_path: str = config.ASTRA_RGB_FRAME_PATH,
    depth_path: str = config.ASTRA_DEPTH_MM_PATH,
) -> float | None:
    """Astra-depth height DELTA at an RGB click, independent of object color/shape.

    The clicked RGB-pixel patch is mapped into Astra depth's native resolution;
    the whole-frame table median is subtracted from the patch median and
    implausible readings return None.
    """
    ret, color = PublishedFrameSource(rgb_path).read()
    if not ret or color is None:
        return None
    if not os.path.exists(depth_path) or (time.time() - os.path.getmtime(depth_path)) >= config.FRAME_STALE_TIMEOUT_S:
        return None
    try:
        depth_mm = np.load(depth_path)
    except (OSError, ValueError):
        return None

    if depth_mm.ndim != 2 or color.ndim < 2:
        return None

    depth_h, depth_w = depth_mm.shape
    color_h, color_w = color.shape[:2]
    if _depth_registration_active():
        # OpenNI2 DEPTH_TO_COLOR registration puts both frames in color-camera
        # coordinates; only the published resolutions still differ.
        offset_x, offset_y = config.ASTRA_DEPTH_REGISTERED_OFFSET_PX
        center_x = px * depth_w / color_w - offset_x
        center_y = py * depth_h / color_h - offset_y
    else:
        # Astra S nominal FOV: color 62.0 x 48.6 deg, depth 58.4 x 45.5 deg.
        # Map about each optical center using tan(FOV/2), not a naive aspect
        # resize, when a driver cannot expose OpenNI2 image registration.
        x_scale = np.tan(np.deg2rad(62.0 / 2)) / np.tan(np.deg2rad(58.4 / 2))
        y_scale = np.tan(np.deg2rad(48.6 / 2)) / np.tan(np.deg2rad(45.5 / 2))
        center_x = depth_w * (0.5 + (px / color_w - 0.5) * x_scale)
        center_y = depth_h * (0.5 + (py / color_h - 0.5) * y_scale)

    radius = config.HEIGHT_SAMPLE_RADIUS_PX
    radius_x = radius * depth_w / color_w
    radius_y = radius * depth_h / color_h
    x0, y0 = max(0, int(np.floor(center_x - radius_x))), max(0, int(np.floor(center_y - radius_y)))
    x1, y1 = min(depth_w, int(np.ceil(center_x + radius_x))), min(depth_h, int(np.ceil(center_y + radius_y)))
    if x1 <= x0 or y1 <= y0:
        return None

    patch_valid = depth_mm[y0:y1, x0:x1]
    patch_valid = patch_valid[patch_valid > 0]
    table_valid = depth_mm[depth_mm > 0]
    if patch_valid.size < config.HEIGHT_SAMPLE_MIN_VALID_PIXELS or table_valid.size < 100:
        return None

    height_m = (float(np.median(table_valid)) - float(np.median(patch_valid))) / 1000.0
    if not (config.OBJECT_HEIGHT_MIN_M <= height_m <= config.OBJECT_HEIGHT_MAX_M):
        return None
    return height_m
