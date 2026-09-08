"""CLI for local Ollama RGB-D detection and text-directed pick/place preview."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import load_pipeline_config
from .frame_io import load_rgbd_frame
from .local_planner import LocalPickPlacePlan, plan_local_task
from .local_vlm import OllamaVlmClient
from .snapshot import capture_published_snapshot


def format_preview(plan: LocalPickPlacePlan) -> str:
    return "\n".join(
        [
            "=== Pick-and-place preview (no motor command sent yet) ===",
            f"source: {plan.command.source.label} box={plan.command.source.box_xyxy} "
            f"confidence={plan.command.source.confidence:.2f}",
            f"destination: {plan.command.destination.label} box={plan.command.destination.box_xyxy} "
            f"confidence={plan.command.destination.confidence:.2f}",
            f"arm: {plan.arm.name}",
            f"source base xyz: {tuple(round(value, 4) for value in plan.source_xyz)} m",
            f"destination base xyz: {tuple(round(value, 4) for value in plan.destination_xyz)} m",
            "The next step may send motor commands. Type 'yes' only after checking the plan.",
        ]
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--command", required=True, help="e.g. 'red block를 blue tray에 가져다줘'")
    parser.add_argument("--snapshot", type=Path, help="previously saved RGB-D snapshot directory")
    parser.add_argument("--rgb-path", type=Path, help="published Astra RGB PNG")
    parser.add_argument("--depth-path", type=Path, help="published Astra depth .npy")
    parser.add_argument("--snapshot-root", type=Path, default=Path("captures"))
    parser.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-model", default="qwen2.5vl:7b")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--yes", action="store_true", help="confirm the printed preview")
    return parser


def _load_frame(args, config):
    if args.snapshot is not None:
        return load_rgbd_frame(args.snapshot)
    if args.rgb_path is None or args.depth_path is None:
        raise ValueError("provide --snapshot or both --rgb-path and --depth-path")
    directory = capture_published_snapshot(
        args.rgb_path, args.depth_path, config.camera_intrinsics, args.snapshot_root
    )
    print(f"snapshot: {directory}")
    return load_rgbd_frame(directory)


def _execute(plan: LocalPickPlacePlan) -> None:
    if not plan.arm.port:
        raise ValueError(f"arm {plan.arm.name!r} has no configured port")
    # This adapter intentionally reuses the repository's tested, interpolated
    # SO-101 controller. It is imported lazily so dry-run remains hardware-free.
    from custom_scripts.vision_pick_place.task_trash_to_bin.kinematics import SOArm101
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower

    from .adapters import SO101ArmAdapter
    from .orchestrator import PickAndPlaceExecutor

    # Give each physical arm its own LeRobot calibration id. The legacy task
    # wrapper defaults to a single "follower" id, which is unsafe for a pair.
    robot = SOFollower(
        SOFollowerRobotConfig(
            port=plan.arm.port,
            id=plan.arm.name,
            use_degrees=True,
            max_relative_target=15.0,
        )
    )
    robot.connect(calibrate=False)
    arm = SOArm101(port=plan.arm.port, robot=robot)
    arm._protect_arm_motors()
    try:
        home = tuple(float(value) for value in arm.gripper_xyz())
        adapter = SO101ArmAdapter(arm)
        executor = PickAndPlaceExecutor(
            arm=adapter,
            verify_gripper=adapter.verify_grasp,
            verify_wrist=adapter.verify_pose,
            home=home,
            bin_pose=plan.destination_xyz,
            approach_height_m=0.08,
            lift_height_m=0.10,
        )
        executor.execute(np.asarray(plan.source_xyz, dtype=float))
    finally:
        robot.disconnect()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.dry_run == args.execute:
        parser.error("choose exactly one of --dry-run or --execute")
    try:
        config = load_pipeline_config(args.config)
        frame = _load_frame(args, config)
        client = OllamaVlmClient(endpoint=args.ollama_endpoint, model=args.ollama_model)
        detections = client.detect_objects(
            frame.rgb,
            "Detect every physical object and every destination tray/bin/region visible in the image. "
            "Use kind=object for pickable objects and kind=destination for places. "
            "Ignore robots, cables, furniture, and the whole background. "
            "If no clear pickable object or destination is visible, return an empty objects array.",
        )
        command = client.parse_task(args.command, detections)
        plan = plan_local_task(
            frame,
            command,
            config.camera_to_base,
            config.arms,
            center_exclusion_half_width=config.center_exclusion_half_width_m,
        )
        print(format_preview(plan))
        if args.execute:
            config.require_execution_ready()
            if not args.yes:
                print("execution blocked: rerun with --yes after reviewing the preview")
                return 2
            _execute(plan)
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
