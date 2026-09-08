"""RGB-D backprojection and rigid-transform helpers."""

from __future__ import annotations

import numpy as np

from .models import ObjectRoi, RgbdFrame


def build_roi_point_cloud(
    frame: RgbdFrame,
    roi: ObjectRoi,
    *,
    min_z: float | None = None,
    max_z: float | None = None,
) -> np.ndarray:
    """Backproject an image ROI to an ``N x 6`` XYZRGB point cloud.

    Depth values are interpreted as millimetres and output in metres.  The
    bounding box uses half-open coordinates and must be wholly inside the
    image; invalid boxes are rejected instead of silently changing the target.
    """

    height, width = frame.depth.shape
    if (
        roi.x0 < 0
        or roi.y0 < 0
        or roi.x1 > width
        or roi.y1 > height
        or roi.x1 <= roi.x0
        or roi.y1 <= roi.y0
    ):
        raise ValueError("ROI must be within the image bounds")
    if min_z is not None and max_z is not None and min_z > max_z:
        raise ValueError("min_z must not exceed max_z")

    depth_mm = frame.depth[roi.y0 : roi.y1, roi.x0 : roi.x1]
    rgb = frame.rgb[roi.y0 : roi.y1, roi.x0 : roi.x1]
    valid = depth_mm > 0
    depth_m = depth_mm.astype(np.float64) / 1000.0
    if min_z is not None:
        valid &= depth_m >= min_z
    if max_z is not None:
        valid &= depth_m <= max_z

    rows, cols = np.nonzero(valid)
    z = depth_m[rows, cols]
    u = cols + roi.x0
    v = rows + roi.y0
    intrinsics = frame.intrinsics
    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy
    colors = rgb[rows, cols].astype(np.uint8, copy=False)
    return np.column_stack((x, y, z, colors)).astype(np.float64, copy=False)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to an ``N x 3`` point array."""

    points = np.asarray(points)
    transform = np.asarray(transform)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if transform.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")

    homogeneous = np.concatenate((points, np.ones((len(points), 1), dtype=points.dtype)), axis=1)
    transformed = homogeneous @ transform.T
    scale = transformed[:, 3:4]
    if np.any(np.isclose(scale, 0.0)):
        raise ValueError("transform produced a point at infinity")
    return transformed[:, :3] / scale
