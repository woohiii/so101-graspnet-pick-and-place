# SO-101 Astra S Local VLM Pick-and-Place

Left SO-101 follower-arm pick-and-place tooling using an Astra S RGB-D camera,
local Qwen2.5-VL object detection, hand-eye calibration, and guarded visual
servoing.

## Safety status

This project is intentionally fail-closed.  The current left-arm hand-eye
calibration must be recreated after a follower motor calibration.  Do not use
`--execute` until a fresh calibration has passed validation and dry-run checks.

## Included components

- `src/vision_pick_place/so101_graspnet_pick_and_place/`: calibration,
  left-arm isolation, home-pose checks, Qwen visual-servo, tests, and pipeline
  code.
- `src/vision_pick_place/task_trash_to_bin/`: shared SO-101 kinematics used by
  the isolated left-arm wrapper.
- `src/vision_pick_place/task_red_cube_to_bin_new_gripper/`: local Qwen2.5-VL
  detector and its supporting configuration.
- `src/vision_pick_place/orbbec_color_camera.py`: Astra S OpenNI2 RGB-D helper.
- `src/vision_pick_place/so101_urdf/`: SO-101 URDF and mesh assets.

## Prerequisites

This project is extracted from a LeRobot workspace.  Run it from a compatible
LeRobot checkout with the hardware dependencies, OpenNI2/Orbbec runtime,
PyTorch CUDA, and Qwen2.5-VL dependencies installed.

Calibrate the left follower with its own ID before collecting hand-eye data:

```bash
uv run lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=so101_left_follower
```

## Current safe workflow

1. Check the left arm without commanding motion:

   ```bash
   uv run python -m custom_scripts.vision_pick_place.so101_graspnet_pick_and_place.ik_home_diagnostic
   ```

2. Collect fresh hand-eye points into a new directory after motor calibration.

3. Run `calibrate_joint_tcp` and use the resulting JSON only for dry-runs.

4. Use `visual_servo --preview` and verify the Qwen marker before any physical
   command.  The visual-servo defaults to one bounded correction per run.

## Important local files not committed

Hand-eye JSON files, point captures, model caches, camera recordings, and
device-specific calibration artifacts are deliberately excluded.  They can
contain machine-specific geometry and must be regenerated for each arm.
