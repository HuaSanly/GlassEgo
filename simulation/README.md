# MuJoCo Simulation Setup

This folder contains the BRX20260825 MuJoCo task scene and the GlassEgo
inference adapter.

## Target

- Robot model: `/home/huasan/BRX20260825/urdf/BRX20260825.urdf`, with the full
  BRX body, head, wheels, dual 7-DoF arms, and grippers loaded.
- First project task: Block Jamming on a compact gray matte tabletop, with a
  movable red square block and a small open blue tray.
- Conda environment: `GlassEgo`.
- External mesh source: `/home/huasan/BRX20260825/meshes`.

## Install

```bash
conda activate GlassEgo
pip install -r requirements.txt
```

## Verify

From the GlassEgo repo root:

```bash
conda run -n GlassEgo python simulation/scripts/build_block_jamming_brx_scene.py
MUJOCO_GL=egl conda run -n GlassEgo python simulation/scripts/run_brx_oracle_grasp.py --grasp-mode physical
MUJOCO_GL=egl conda run -n GlassEgo python simulation/scripts/run_block_jamming_env.py --render-check
MUJOCO_GL=egl conda run -n GlassEgo python simulation/scripts/run_block_jamming_env.py --render-check --camera task_overview
conda run -n GlassEgo python simulation/scripts/run_block_jamming_env.py
```

## Run policy inference

The simulation entrypoint uses MuJoCo ground truth for object poses and renders
the clean RGB input expected by the checkpoint. It currently drives the right
arm for the single-hand Block Jamming checkpoint:

The model's `ego_rgbd` camera is mounted on `Head03_Link`, 50 mm above and
15 mm forward of the middle eye. It looks down 55 degrees with a 60-degree
vertical FOV. Both head joints have position servos to prevent gravity sag.
This head-mounted view supplies RGB-D and clean-image policy input;
`task_overview` and the interactive viewer's free camera are display-only.

```bash
conda run -n GlassEgo \
  python inference/run_inference_sim.py --max-steps 100
```

Use `--device cpu` when running in an environment without CUDA. Add
`--result-json /tmp/sim_result.json` to persist episode metrics. On a desktop,
the MuJoCo viewer opens automatically; use `--headless` to force offscreen
execution. Without `--max-steps`, the entrypoint loops over new episodes until
`Ctrl-C`; use `--once` for one episode or `--loop --max-steps N` for repeated
bounded episodes. The same window stays open and motion plays at real time;
each episode ends with a 2-second pause before resetting the scene. Change
that pause with `--loop-delay SECONDS`. Headless runs remain unpaced.
Both modes use the real-world template's command loop: 8 action commands at
10 Hz, then a new observation and inference. `--policy-only` follows model
motion/grasp outputs without geometry intervention, stopping on model done or
the episode budget. The default hybrid mode gates closure at 45 mm but keeps
executing model motion. It allows at least 8 policy cycles before takeover on
sustained IK failure, repeated unsafe closure without progress, prolonged
stalling, false done, or budget exhaustion (default policy budget: 40 cycles).
The physical grasp/place supervisor then works from the current scene. Results
report policy command counts and the explicit takeover reason separately from
the geometry outcome; object perception remains MuJoCo ground truth.

`run_block_jamming_env.py` loads the BRX scene by default. The red block is a
3 cm cube sampled in the right-arm workspace (approximately `x=0.60..0.65 m`,
`y=-0.23..-0.15 m`); the checked-in default is `(0.64, -0.165, 0.565) m`.
The blue tray stays at `(0.64, 0.02, 0.555) m`, shifted 6 cm toward the robot's
left (+Y) from the previous layout to leave more grasp clearance beside the
red block. RGB render checks use `640x480`. Use `--goal-state` to start
with the red block inside the blue tray, or pass `--target-position X Y Z` and
`--block-position X Y Z` for a controlled placement. The Oracle script
validates the complete no-model grasp/place path; `--grasp-mode weld` selects
an explicitly marked simulation-only weld fallback.

## Notes

- Keep the BRX vendor meshes outside this repo. Do not vendor STL assets into
  GlassEgo.
- Use this setup first to validate the MuJoCo robot/task stack. The next phase is
  to add adapters for `inference.interfaces.Camera`, `RobotArm`, and
  `Perception`.
- If GLFW rendering fails but model loading works, prefer `MUJOCO_GL=egl` for
  automated offscreen checks.
