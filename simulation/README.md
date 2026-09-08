# MuJoCo Simulation Setup

This folder contains the minimal project-side glue for bringing up the official
Trossen MuJoCo task environment. It does not connect to GlassEgo inference yet.

## Target

- Robot model: two Trossen official WXAI follower arms, matching the
  HumanEgo example hardware class.
- First project task: Block Jamming on a white tabletop, with a movable block
  and a yellow target box.
- Conda environment: `glassego-mujoco`.
- External source checkout: `/home/huasan/trossen_arm_mujoco`.

## Install

```bash
conda create -n glassego-mujoco python=3.11 -y
conda activate glassego-mujoco

pip install --upgrade pip
pip install mujoco gymnasium numpy scipy opencv-python pyyaml

git clone https://github.com/TrossenRobotics/trossen_arm_mujoco.git /home/huasan/trossen_arm_mujoco
cd /home/huasan/trossen_arm_mujoco
pip install -e .
```

## Verify

From the GlassEgo repo root:

```bash
conda run -n glassego-mujoco python simulation/scripts/check_mujoco_env.py
MUJOCO_GL=egl conda run -n glassego-mujoco python simulation/scripts/check_mujoco_env.py --render-check
conda run -n glassego-mujoco python simulation/scripts/run_trossen_stationary_demo.py --headless-steps 200
conda run -n glassego-mujoco python simulation/scripts/run_trossen_stationary_demo.py
conda run -n glassego-mujoco python simulation/scripts/build_block_jamming_wxai_scene.py
conda run -n glassego-mujoco python simulation/scripts/run_block_jamming_env.py --headless-steps 200
MUJOCO_GL=egl conda run -n glassego-mujoco python simulation/scripts/run_block_jamming_env.py --render-check
MUJOCO_GL=egl conda run -n glassego-mujoco python simulation/scripts/run_block_jamming_env.py --render-check --camera task_overview
conda run -n glassego-mujoco python simulation/scripts/run_block_jamming_env.py
```

`check_mujoco_env.py` verifies imports and loads a Stationary XML model. The
optional `--render-check` renders one offscreen frame and is useful for debugging
OpenGL/EGL issues.

`run_trossen_stationary_demo.py --headless-steps N` runs the official task logic
without a viewer for a quick smoke test. Without `--headless-steps`, it launches
the official Trossen Stationary viewer demo. Close the MuJoCo viewer to stop it.

`build_block_jamming_wxai_scene.py` builds
`simulation/tasks/block_jamming_wxai.xml` from the official Trossen
`wxai_follower.xml` model. The generated task scene uses two WXAI follower arms
mounted side by side, a white task tabletop, a movable block, a yellow target
box, and a fixed `ego_rgbd` camera for HumanEgo-style camera-frame simulation.

`run_block_jamming_env.py` loads the WXAI Block Jamming scene by default. Use
`--goal-state` to start with the block already sitting on the yellow box for
checking the desired final pose. Use `--camera task_overview` for an overview
render instead of the default `ego_rgbd` view. The generated XML still points to
the external Trossen checkout for mesh and texture assets instead of vendoring
STL/PNG files into this repository.

## Notes

- Keep the official Trossen repository outside this repo. Do not vendor its
  assets into GlassEgo.
- Use this setup first to validate the MuJoCo robot/task stack. The next phase is
  to add adapters for `inference.interfaces.Camera`, `RobotArm`, and
  `Perception`.
- If GLFW rendering fails but model loading works, prefer `MUJOCO_GL=egl` for
  automated offscreen checks.
