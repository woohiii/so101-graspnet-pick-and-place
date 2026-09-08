"""Load SOArm101 with the left-arm-only configuration namespace."""

from __future__ import annotations

import sys
from pathlib import Path


def _load_left_arm_module():
    root = Path(__file__).resolve().parent
    vision_dir = root.parent
    runtime_dir = root / "left_arm_runtime"
    task_dir = vision_dir / "task_trash_to_bin"
    for path in reversed((str(runtime_dir), str(vision_dir), str(task_dir))):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    # The shared module uses ``import config``.  Fail closed if another arm's
    # config was imported first in this Python process.
    imported = sys.modules.get("config")
    if imported is not None and Path(imported.__file__).resolve().parent != runtime_dir:
        raise RuntimeError("right-arm config is already loaded; start a fresh process for the left arm")
    from task_trash_to_bin import kinematics

    return kinematics


_kinematics = _load_left_arm_module()
_SharedSOArm101 = _kinematics.SOArm101


class LeftSOArm101(_SharedSOArm101):
    """SOArm101 bound to the left arm's independent LeRobot calibration ID."""

    def __init__(self, port: str = "/dev/ttyACM1", robot=None):
        super().__init__(
            port=port,
            robot=robot,
            robot_id="so101_left_follower",
            disable_torque_on_disconnect=False,
        )


def build_left_kinematics():
    """Construct FK/IK using the isolated left-arm configuration."""
    return _kinematics.build_kinematics()
