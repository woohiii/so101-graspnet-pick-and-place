"""Read-only FK/IK sensitivity check for the currently held SO-101 pose."""

from __future__ import annotations

import numpy as np


def main() -> int:
    from .left_arm import LeftSOArm101

    arm = LeftSOArm101(port="/dev/ttyACM1")
    connected = False
    try:
        arm.connect()
        connected = True
        joints = arm.get_joint_deg()
        xyz = arm.gripper_xyz()
        print("current joints:", np.round(joints, 2))
        print("current FK TCP:", np.round(xyz, 4))
        for name, target in (
            ("+X 1cm", (xyz[0] + 0.01, xyz[1], xyz[2])),
            ("+Y 1cm", (xyz[0], xyz[1] + 0.01, xyz[2])),
        ):
            plan = arm.preview_move(target)
            print(f"{name} max joint delta: {plan['max_abs_delta_deg']:.2f} deg")
    finally:
        if connected:
            arm.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
