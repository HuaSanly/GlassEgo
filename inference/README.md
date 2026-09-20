# Inference — Real-World Deployment

This folder is a **reference template** for deploying a trained HumanEgo policy on
a real dual-arm robot. It is intentionally clean and hardware-agnostic: it shows
the *standard structure* of a HumanEgo inference stack so you can wire in **your
own** camera, robot, and perception and reuse everything else.

> ⚠️ **It will not run out of the box.** It depends on physical hardware
> (camera + arms), a hand-eye calibration, and heavy perception models. Treat it
> as the blueprint to build your own deployment from — not a turn-key script. The
> example uses the same hardware as the paper: **Intel RealSense + Trossen arms**,
> with **DINO-SAM + LaMa** perception.

---

## The idea in one picture

```
 camera ─▶ perception ─▶ clean image + ICT ─▶ policy ─▶ EE trajectory ─▶ robot
    ▲                                                                      │
    └──────────────────────────── close the loop ◀─────────────────────────┘
```

A HumanEgo policy consumes two things every step and predicts a future
end-effector trajectory:

1. **A clean, embodiment-agnostic RGB image** — the real arm is inpainted out and
   a virtual gripper is rendered in its place (this closes the *visual gap* between
   human-video training and robot deployment).
2. **Interaction-Centric Tokens (ICT)** — a compact, viewpoint- and
   embodiment-invariant encoding of every hand and object as a 6DoF entity plus
   each hand's pose *relative to* that entity (this closes the *kinematic gap*).

Because the ICT and clean image are built **identically** at train and test time,
a policy trained purely on human egocentric video transfers to the robot.

---

## Files

| File | Role |
|------|------|
| [`interfaces.py`](interfaces.py) | The 3 abstractions you implement: `Camera`, `RobotArm`, `Perception`. The loop is written entirely against these. |
| [`policy.py`](policy.py) | `ICTPolicy`: load the checkpoint, `prepare_image`, `build_ict` (dual-arm), flow-matching `infer`, decode to camera-frame EE targets. |
| [`controller.py`](controller.py) | `TrajectoryController`: smooth (EMA + Slerp) + rate-limit predictions and servo the arms (receding horizon). |
| [`run_inference.py`](run_inference.py) | The main loop wiring it all together, plus example hardware adapters and a reference perception. |
| [`../cfg/inference/example_dualarm.yaml`](../cfg/inference/example_dualarm.yaml) | One annotated config for the whole stack. |
| `CamRS.py`, `RobotArmTrossen.py` | **Example** drivers (RealSense / Trossen) you can adapt to your own hardware. |

---

## Read this first — frame & unit conventions

Most deployment bugs are frame bugs. The contract (see `interfaces.py`):

- **Poses are 4×4 SE(3) matrices, positions in meters, rotations proper.**
- **`_in_cam` = the camera optical frame** (OpenCV: +x right, +y down, +z forward).
  This is the single shared "world" frame for one episode.
- **`T_base_in_cam`** (per arm) — the **hand-eye extrinsic** placing the robot base
  in the camera frame. Get it from a hand-eye calibration. A wrong extrinsic is the
  #1 cause of "the robot moves to the wrong place".
- **`T_align`** — bridges *your* end-effector frame to the **"hand" frame the model
  was trained on**. Identity if you trained on robot/teleop data in the same EE
  convention; for the released Aria-MPS checkpoints use the fixed rotation provided
  in [`example_dualarm.yaml`](../cfg/inference/example_dualarm.yaml) (the `T_align`
  block). Wrong `T_align` ⇒ orientations are systematically off.

---

## The pipeline, step by step

**One-time, at episode start** (`run_inference.py: run()`):

1. **Estimate object poses** — `Perception.estimate_objects()` detects+segments each
   object, lifts mask pixels to 3D via depth, and fits a 6DoF pose. The **anchor**
   object (`obj1`) defines the object-centric reference frame the ICTs use.
2. **Home** the arms and open grippers.

**Every step, closed-loop** (~5–10 Hz):

| # | What | Where |
|---|------|-------|
| 3 | Grab an RGB-D frame | `Camera.get_frame()` |
| 4 | Read each arm's EE pose (FK) + gripper, bridge by `T_align` → **hand poses** | `RobotArm.get_T_ee_in_cam()` |
| 5 | **Latch** grasped objects so their pose tracks the gripper | `run_inference.py: latch_objects()` |
| 6 | Build the **clean image** (inpaint arm + render virtual gripper) | `Perception.make_clean_image()` |
| 7 | Build the **ICT** from hand + object poses | `ICTPolicy.build_ict()` |
| 8 | **Flow-matching inference** → future EE trajectory + done prob | `ICTPolicy.infer()` |
| 9 | **Decode** reference-frame prediction → camera-frame EE targets | `ICTPolicy.decode_ee_in_cam()` |
| 10 | Smooth + execute the first `exec_horizon` steps, then re-plan | `TrajectoryController.execute_chunk()` |
| 11 | Stop when `done_prob > done_threshold` | `run_inference.py: run()` |

### How the ICT is built (the core)

`ICTPolicy.build_ict()` mirrors `training/FlowMatchingDataloader._build_ict()` exactly.
Each entity (hand or object) becomes one token:

```
[ type_id(1) | pose_in_ref(9) | hand-in-entity(9 single / 18 dual) | flag(1) ]
```

- `pose_in_ref` — the entity's 6DoF pose in the reference frame, encoded as
  `[normalized position(3), 6D rotation(6)]`.
- `hand-in-entity` — each hand's pose **expressed in that entity's frame**. This
  relative encoding is what makes the representation *interaction*-centric and
  invariant to where the camera/robot is.
- Token order is fixed: **hand(s) first, then the anchor object, then the rest** —
  it must match training.

### How the robot moves

The policy predicts the future **hand** trajectory in the reference frame. To
command the arm, each step is mapped back (`decode_ee_in_cam`):

```
pred (hand pose in REF) --T_ref_in_cam--> hand pose in CAM --inv(T_align)--> EE pose in CAM
```

`TrajectoryController` then EMA-smooths position, Slerp-smooths rotation, clamps
the per-step motion (safety cage), and calls `RobotArm.move_ee_in_cam(...)`
non-blocking, plus opens/closes the gripper by thresholding the predicted grasp
probability. Only the first `exec_horizon` steps run before the loop re-plans on a
fresh observation — this **receding horizon** keeps the policy reactive.

---

## How to run

### MuJoCo simulation

`run_inference_sim.py` wires the existing policy and trajectory controller to
the Block Jamming MuJoCo scene. `interface_sim.py` provides an RGB-D Ego camera,
damped-least-squares site IK, gripper control, and deterministic Oracle
perception. This keeps the first simulation bring-up independent of DINO/SAM or
LaMa while preserving the `Frame`, `ObjectState`, ICT, and clean-image
contracts.

The model input camera is `ego_rgbd`, attached to `Head03_Link`, 50 mm above
and 15 mm in front of the middle eye, pitched down 55 degrees (vertical FOV
60 degrees). Head position servos hold the neutral pose against gravity.
The viewer's free camera and `task_overview` do not feed the model. The policy
still receives this head camera's clean image with robot meshes hidden and
virtual gripper/object overlays, resized to the checkpoint's image size.
Camera intrinsics use square pixels (`fx = fy`); object anchors and robot-base
extrinsics are evaluated in the current head-camera frame each observation.

```bash
conda run -n GlassEgo python inference/run_inference_sim.py --device cpu
```

For a repeatable scene layout, the BRX XML contains a tray at `(0.64, 0.02,
0.555)` and a red block at `(0.64, -0.165, 0.565)` (meters). Both can be
overridden without editing the XML:

```bash
conda run -n GlassEgo python inference/run_inference_sim.py --device cpu \
  --block-position 0.64 -0.165 0.565 \
  --target-position 0.64 0.02 0.555
```

The simulation entrypoint prints per-cycle inputs, policy output, each executed
substep, IK status, scene state, and stage timings by default. Use `--quiet` to
keep only the termination summary and final JSON result.
Pass `--seed N` together with the position overrides to make the policy noise
and the episode repeatable.

Both modes share `TrajectoryController.execute_chunk` with the real-world
template: predict a trajectory, execute its first 8 commands at 10 Hz, then
capture a fresh observation and replan. Position/rotation smoothing, the
grasp threshold, failed-IK handling and first-command behavior are identical.
The simulation sends the nonblocking arm and gripper targets before advancing
one control interval, including when IK fails; it no longer moves first and
then adds separate gripper settling time. Nonblocking motion uses MuJoCo joint
position servos, so this matches command ordering rather than a particular
hardware SDK's internal Cartesian interpolation. Both modes use the same IK
settings during policy control and feed back the measured gripper value.

`--policy-only` disables all geometry gating and takeover. It stops after a
chunk when the model's done probability exceeds 0.8, or when the simulation
episode budget (`--max-steps`, default 100 inference cycles) is exhausted.
IK failures and geometric success are recorded but do not stop policy control.
The final `success` is measured from the physical scene, independently of the
model's done prediction. Perception still uses MuJoCo object ground truth;
the hardware template's approximate object-latching heuristic is not used.

The default hybrid mode retains the same policy loop. It only suppresses close
commands when the current finger midpoint is more than 45 mm from the current
block center; it never cancels the motion chunk based on a future close target.
The model gets at least 8 complete inference/execution cycles before takeover.
After that, the geometry supervisor takes over on 5 consecutive all-IK-failed
cycles, 5 consecutive gated-close cycles accompanied by 5 cycles without
progress, 12 cycles without progress, or a model done signal without placement.
Continuing improvement delays takeover, up to a 40-cycle policy budget (or the
episode limit if it is reached after the minimum 8 cycles). Progress means at
least 5 mm improvement in approach distance, lift up to 50 mm, or tray alignment
after lifting; merely pushing the block toward the tray is not placement
progress. Geometry then uses its own IK settings and attempts a physical
grasp/place from the current scene, without resetting the block. Takeover opens
the jaws and approaches directly without a joint-space home move across the
table. The grasp orientation is derived from the finger pad geometry rather
than a fixed correction to the policy's final wrist pose.

JSON results separate `policy_execution_steps` (attempted action commands),
`policy_ik_failures`, `policy_max_lift` (sampled after each chunk),
`policy_final_scene`, `gated_close_count`, `fallback_trigger`, and
`geometry_fallback` diagnostics. A fallback success is not a policy-only success.

### Inspect predictions independently of robot execution

```bash
conda run -n GlassEgo python inference/run_inference_sim.py --viewer --seed 7 \
  --preview-prediction
```

After opening/homing the arms, this mode freezes physics, runs the model once,
and displays its **complete decoded EE trajectory** in MuJoCo world coordinates.
It bypasses smoothing, step limits, IK, gripper commands, and geometry takeover.
The window stays open until closed or interrupted; the moving marker repeats
the same prediction at `control_hz`, without advancing the robot or requesting
new predictions. This is a visualization of one predicted horizon, not a
closed-loop task rollout or the flow-matching ODE integration process.
Preview takes precedence over `--loop`, `--max-steps`, and the model done signal.

Cyan points/lines indicate predicted open grasps; orange indicates closure
above the configured grasp threshold. Bright points are the first `exec_horizon`
commands (normally 8); the remaining prediction is translucent. Sparse red,
green, and blue arrows show the decoded robot EE's local X/Y/Z axes, after
`T_align` conversion. The animated marker is labeled with its zero-based index
and predicted closure probability. Rotate/zoom the viewer to inspect depth.

Use `--show-prediction` during normal policy or hybrid execution to replace the
overlay after each inference while the robot continues moving. Each overlay
is frozen using that observation's camera pose, so head movement does not move
the already predicted world targets. All markers are display-only geometry;
they have no collisions and are absent from model RGB-D input.

For offscreen inspection, save a world-pose/probability JSON and observer PNG:

```bash
conda run -n GlassEgo python inference/run_inference_sim.py --headless --seed 7 \
  --preview-prediction --prediction-output runs/prediction-visualization/seed7.json
```

`--prediction-output` also works with normal execution, overwriting the latest
prediction and PNG each cycle. The JSON records the checkpoint, observation
camera transform, full world poses and grasp probabilities. Preview reports
zero policy commands; its scene `success` field is not a grasp evaluation.

On a desktop with `DISPLAY`, the inference entrypoint opens the MuJoCo viewer
automatically; pass `--headless` to force EGL/offscreen execution. On machines
without `DISPLAY`, headless mode is selected automatically. The simulation
adapter selects the NVIDIA GLX vendor automatically when `libGLX_nvidia` is
available. A desktop launch can therefore use the same command without
manually exporting `__GLX_VENDOR_LIBRARY_NAME`:

```bash
DISPLAY=:0 python inference/run_inference_sim.py --device cpu --seed 7
```

Without `--max-steps`, the command keeps starting new episodes until the viewer
is closed or `Ctrl-C` is pressed. Use `--once` for one episode, or use
`--max-steps N` for a bounded diagnostic run. `--loop --max-steps N` explicitly
enables repeated bounded episodes. The model and viewer stay open across
episodes; the scene and controller state reset each time. Viewer motion plays
at real time with continuous refresh, including geometry fallback. The final
pose stays visible for 2 seconds before reset; `--loop-delay SECONDS` changes
that interval. Headless execution remains unpaced. `--result-json` is updated
after each episode with its latest metrics.

If a wrapper hides the host GLX libraries, export
`__GLX_VENDOR_LIBRARY_NAME=nvidia` before launching instead.

Hybrid takeover reasons are printed with the number of policy cycles and action
commands already attempted. Policy-only runs continue replanning after failed
IK commands, just like the real-world template.

Defaults live in [`sim_config.yaml`](sim_config.yaml). `ego_rgbd` is the policy
camera; `task_overview` remains a viewer/debug camera. MuJoCo camera coordinates
are converted to the OpenCV optical convention with `(x, -y, -z)`. The configured
`T_align` applies the BRX TCP-to-training-hand axis bridge; keep it consistent
with the checkpoint's hand-frame convention when changing the robot model.

**Prerequisites**

1. A trained checkpoint with `config.json` **and** `dataset_stats.json` next to it
   (the trainer writes these; they carry the architecture + normalization).
2. Hardware drivers installed: `SKIP_HARDWARE=0 bash setup.sh` (RealSense + Trossen).
3. Your perception models (the reference uses DINO-SAM + LaMa from `preprocess/`).
4. A **hand-eye calibration** → `T_base_in_cam` for each arm.

**Configure** — edit [`../cfg/inference/example_dualarm.yaml`](../cfg/inference/example_dualarm.yaml):
set `policy.ckpt`, the `perception.object_prompts` / `erase_prompt`, the camera and
robot `cfg_path`s, and `robot.T_align`.

**Run**

```bash
python inference/run_inference.py cfg/inference/example_dualarm.yaml
```

---

## Configuration & tuning

All knobs live in `example_dualarm.yaml`. The ones you will actually tune:

| Knob | Section | Effect |
|------|---------|--------|
| `num_inference_steps` | `policy` | Flow ODE steps. 10–20. ↑ = smoother actions, slower. |
| `exec_horizon` | `control` | Predicted steps run before re-planning. ↓ = more reactive/closed-loop. |
| `control_hz` | `control` | Control-loop rate (sets `dt`). Match to your arm's servo rate. |
| `alpha_pos` / `alpha_rot` | `control` | EMA / Slerp smoothing. ↑ = smoother but laggier. |
| `max_pos_step` | `control` | Safety cage: max EE move per step (m). Keep small at first. |
| `grasp_threshold` | `control` | Predicted grasp prob above which the gripper closes. |
| `done_threshold` | `control` | Done prob above which the episode stops. |
| `safe_z_min` | `control` | Intended base-frame Z floor (table protection). **Note:** this template does not yet wire the value through to the driver — the Trossen example enforces its own −0.12 m floor — so rely on your driver's floor + e-stop and set conservatively. |

**Bring-up tip:** start with a *low* `control_hz`, *small* `max_pos_step`, and a
high `safe_z_min`, hand on the e-stop. Loosen once the motion looks right.

---

## Single-arm vs dual-arm

This template defaults to **dual-arm** (`robot.sides: ["left", "right"]`,
`single_hand: false` in the training config → `ict_dim = 29`, both hand
trajectories predicted in one forward pass). For **single-arm**, set
`robot.sides: ["right"]` and use a checkpoint trained with `single_hand: true`
(`ict_dim = 20`). `ICTPolicy` reads `single_hand` from the checkpoint's
`config.json`, so the token layout and trajectory unpacking follow automatically.

---

## Writing your own camera / robot / perception

Implement the three interfaces in `interfaces.py` — that's the whole porting job:

- **`Camera`** → return `Frame(rgb, depth_m, K)`. See `RealSenseCamera` in
  `run_inference.py` wrapping `CamRS`.
- **`RobotArm`** → FK (`get_T_ee_in_cam`), Cartesian servo (`move_ee_in_cam`),
  gripper, `go_home`, and the `T_base_in_cam` extrinsic. See `TrossenArm` wrapping
  `RobotArmTrossen`. Cartesian IK is your driver's job.
- **`Perception`** → object 6DoF poses + a clean image. This is the heaviest part.
  Any source of object poses works (open-vocab detector + PCA on depth keypoints, an
  AprilTag, FoundationPose, known CAD + ICP, …). The clean image must match how your
  model was trained (same inpainting + gripper rendering as `preprocess/`).

---

## What this template leaves out

To keep the idea legible, this reference template intentionally omits several pieces
from our internal production loop (not part of this release):

- **Async control + temporal ensembling** — a worker thread servoing at a fixed rate
  decoupled from (slower) inference, averaging overlapping predictions for smoother
  motion.
- **Delta action mode**, PCD features, region attention, object-dynamics & visual-
  foresight auxiliary heads — all supported by the model; here we run the common
  absolute-action path.
- **Robustness/UX**: keyboard tele-override, live visualization, post-grasp forced
  lift, IK-failure escape, interactive extrinsic calibration, grasp latching across
  occlusion, checkpoint architecture auto-detection.

Start from this template, get a single arm reaching to an object, then layer these on
as you need them.
