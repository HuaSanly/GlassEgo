"""Run a trained HumanEgo policy in the MuJoCo Block Jamming scene."""

from __future__ import annotations

import argparse
import json
import os
import sys
from time import perf_counter, sleep
from pathlib import Path

# Select EGL before importing MuJoCo so the policy camera's offscreen renderer
# does not share a GLFW/GLX context with the optional interactive viewer.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import yaml
import mujoco
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.controller import TrajectoryController
from simulation.geometry_grasp import GeometryGraspSupervisor, PolicyProgressMonitor
from simulation.interface_sim import (
    OracleSimPerception,
    SimCamera,
    SimRobotArm,
    SimWorld,
    ViewerClosedError,
)
from simulation.prediction_visualization import PredictionOverlay


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message, flush=True)


def _fmt_vector(values: np.ndarray, decimals: int = 4) -> str:
    return str(np.round(np.asarray(values, dtype=np.float64), decimals).tolist())


def _pose_summary(pose: np.ndarray) -> tuple[str, str]:
    position = _fmt_vector(pose[:3, 3])
    euler = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
    return position, _fmt_vector(euler, 1)


def _set_target_position(world: SimWorld, position: np.ndarray) -> None:
    body_id = int(mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, "target_container"))
    if body_id < 0:
        raise ValueError("MuJoCo scene is missing target_container")
    target = np.asarray(position, dtype=float).reshape(3)
    if not np.all(np.isfinite(target)):
        raise ValueError("target position must contain three finite values")
    world.model.body_pos[body_id] = target
    mujoco.mj_forward(world.model, world.data)


def _set_block_position(world: SimWorld, position: np.ndarray) -> None:
    joint_id = int(mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_JOINT, "block_freejoint"))
    if joint_id < 0:
        raise ValueError("MuJoCo scene is missing block_freejoint")
    block = np.asarray(position, dtype=float).reshape(3)
    if not np.all(np.isfinite(block)):
        raise ValueError("block position must contain three finite values")
    qpos = int(world.model.jnt_qposadr[joint_id])
    qvel = int(world.model.jnt_dofadr[joint_id])
    world.data.qpos[qpos : qpos + 3] = block
    world.data.qpos[qpos + 3 : qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    world.data.qvel[qvel : qvel + 6] = 0.0
    mujoco.mj_forward(world.model, world.data)


def _scene_status(world: SimWorld, cfg: dict) -> dict:
    model, data = world.model, world.data
    target_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_container_bounds"))
    target_floor_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_container_floor"))
    block_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "block_geom"))
    block_joint = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "block_freejoint"))
    target_center, target_size = data.geom_xpos[target_id], model.geom_size[target_id]
    block_center, block_size = data.geom_xpos[block_id], model.geom_size[block_id]
    xy_delta = block_center[:2] - target_center[:2]
    target_floor_z = data.geom_xpos[target_floor_id][2] + model.geom_size[target_floor_id][2]
    clearance = float(block_center[2] - block_size[2] - target_floor_z)
    linear_velocity = data.qvel[model.jnt_dofadr[block_joint] : model.jnt_dofadr[block_joint] + 3]
    inner_half_size = target_size[:2] - cfg.get("target_wall_thickness", 0.008)
    xy_ok = np.all(np.abs(xy_delta) <= inner_half_size - block_size[:2])
    z_ok = abs(clearance) <= cfg.get("success_z_tolerance", 0.008)
    speed = float(np.linalg.norm(linear_velocity))
    return {
        "target_position": target_center.copy(),
        "block_position": block_center.copy(),
        "xy_distance": float(np.linalg.norm(xy_delta)),
        "clearance": clearance,
        "linear_speed": speed,
        "success": bool(xy_ok and z_ok and speed < cfg.get("success_linear_speed", 0.08)),
    }


def _block_world_position(world: SimWorld) -> np.ndarray:
    """Return the current red block center for the geometric grasp gate."""
    geom_id = int(mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_GEOM, "block_geom"))
    if geom_id < 0:
        raise ValueError("MuJoCo scene is missing block_geom")
    return world.data.geom_xpos[geom_id].copy()


def _policy_progress(world: SimWorld, arm: SimRobotArm, initial_block_z: float) -> np.ndarray:
    """Lower is better: approach distance, negative lift, then tray XY distance."""
    block = _block_world_position(world)
    distance = np.linalg.norm(arm.get_gripper_midpoint_world() - block)
    lift = max(0.0, float(block[2] - initial_block_z))
    target = world.data.geom("target_container_bounds").xpos
    # Pushing the block toward the box on the table is not pick/place progress.
    place_distance = np.linalg.norm(block[:2] - target[:2]) if lift >= 0.05 else np.inf
    return np.array([distance, -min(lift, 0.05), place_distance])


class SimTrajectoryController(TrajectoryController):
    """Share the real-world command loop, advancing MuJoCo instead of sleeping."""

    def __init__(self, arms: dict, cfg: dict, grasp_gate_distance: float | None = None) -> None:
        super().__init__(arms, cfg)
        self.world = next(iter(arms.values())).world
        self.grasp_gate_distance = grasp_gate_distance
        self.last_gate_distance: float | None = None

    def _wait(self, dt: float) -> None:
        self.world.advance(dt)

    def _gripper_command(self, side: str, grasp: float) -> bool:
        closed = super()._gripper_command(side, grasp)
        self.last_gate_distance = None
        if closed and self.grasp_gate_distance is not None:
            # Read the block at each action, including while it follows a grasp.
            self.last_gate_distance = float(np.linalg.norm(
                self.arms[side].get_gripper_midpoint_world()
                - _block_world_position(self.world)
            ))
            if self.last_gate_distance > self.grasp_gate_distance:
                return False
        return closed

    def _command_step(self, side: str, target: np.ndarray, grasp: float, dt: float) -> dict:
        item = super()._command_step(side, target, grasp, dt)
        arm = self.arms[side]
        item.update({
            "grasp_gated": item["predicted_gripper_closed"] and not item["gripper_closed"],
            "grasp_gate_distance": self.last_gate_distance,
            "ik_position_error": arm.last_ik_position_error,
            "ik_rotation_error_deg": arm.last_ik_rotation_error_deg,
            "ik_target_world": arm.last_ik_target_world.copy(),
        })
        return item


def _device(value: str) -> str:
    if value == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    if value == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
    return value


def run(
    config_path: str | Path,
    device: str = "auto",
    headless: bool = False,
    max_steps: int | None = None,
    verbose: bool = True,
    block_position: np.ndarray | None = None,
    target_position: np.ndarray | None = None,
    seed: int | None = None,
    policy_only: bool = False,
    loop: bool = False,
    loop_delay: float = 2.0,
    result_json: Path | None = None,
    show_prediction: bool = False,
    preview_prediction: bool = False,
    prediction_output: Path | None = None,
) -> dict | None:
    with open(config_path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    sim_cfg = cfg["simulation"]
    control_cfg = cfg["control"]
    geometry_cfg = dict(sim_cfg.get("geometry_fallback", {}))
    geometry_enabled = bool(geometry_cfg.get("enabled", True)) and not policy_only and not preview_prediction
    policy_cfg = dict(cfg["policy"])
    xml_path = (ROOT / sim_cfg["xml"]).resolve() if not os.path.isabs(sim_cfg["xml"]) else Path(sim_cfg["xml"])
    checkpoint = policy_cfg.get("ckpt")
    if not checkpoint:
        raise ValueError("simulation policy.ckpt must point to a checkpoint")
    checkpoint = str((ROOT / checkpoint).resolve()) if not os.path.isabs(checkpoint) else checkpoint
    policy_cfg["ckpt"] = checkpoint

    if seed is not None:
        np.random.seed(int(seed))
        import torch

        torch.manual_seed(int(seed))

    selected_device = _device(device)
    from simulation.policy import ICTPolicy

    world = SimWorld(xml_path, sim_cfg.get("width", 640), sim_cfg.get("height", 480), hide_robot_visuals=True)
    if target_position is not None:
        _set_target_position(world, target_position)
    if block_position is not None:
        _set_block_position(world, block_position)
    camera = SimCamera(world, sim_cfg.get("camera", "ego_rgbd"))
    ik_position_tolerance = float(sim_cfg.get("ik_position_tolerance", 0.015))
    ik_rotation_tolerance_deg = float(sim_cfg.get("ik_rotation_tolerance_deg", 8.0))
    interpolation_steps = int(sim_cfg.get("interpolation_steps", 4))
    arms = {
        side: SimRobotArm(
            world,
            camera,
            side,
            ik_damping=sim_cfg.get("ik_damping", 0.04),
            ik_iterations=sim_cfg.get("ik_iterations", 40),
            ik_rotation_weight=sim_cfg.get("ik_rotation_weight", 0.35),
            ik_position_tolerance=ik_position_tolerance,
            ik_rotation_tolerance_deg=ik_rotation_tolerance_deg,
            interpolation_steps=interpolation_steps,
        )
        for side in ("left", "right")
    }
    perception = OracleSimPerception(world, camera, sim_cfg.get("anchor_key", "obj_1"))
    policy = ICTPolicy(policy_cfg, device=selected_device)
    if policy.sides != ["right"]:
        raise ValueError(f"This simulation entrypoint currently expects a right-arm checkpoint, got {policy.sides}")
    gate_distance = float(geometry_cfg.get("distance", 0.045)) if geometry_enabled else None
    controller = SimTrajectoryController(
        {"right": arms["right"]}, control_cfg, grasp_gate_distance=gate_distance
    )
    max_steps = int(max_steps if max_steps is not None else sim_cfg.get("max_steps", 100))
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    dt = 1.0 / float(control_cfg.get("control_hz", 10.0))
    anchor_key = sim_cfg.get("anchor_key", "obj_1")
    static_objects = perception.estimate_objects([])
    T_align = np.asarray(sim_cfg.get("T_align", np.eye(4).tolist()), dtype=np.float32)
    if T_align.shape != (4, 4):
        raise ValueError("simulation.T_align must be a 4x4 hand-to-EE transform")
    _log(verbose, "[sim:init] ------------------------------------------------------------")
    _log(verbose, f"[sim:init] xml={xml_path} camera={camera.name} resolution={camera.world.width}x{camera.world.height}")
    _log(verbose, f"[sim:init] mode={'headless' if headless else 'viewer'} device={selected_device} max_steps={max_steps}")
    _log(verbose, f"[sim:init] policy_only={policy_only} geometry_fallback={geometry_enabled} gate_distance={gate_distance}")
    if seed is not None:
        _log(verbose, f"[sim:init] seed={int(seed)}")
    _log(verbose, f"[sim:init] control_hz={control_cfg.get('control_hz', 10)} exec_horizon={control_cfg.get('exec_horizon', 8)} dt={dt:.3f}s")
    _log(verbose, f"[sim:init] done_threshold={control_cfg.get('done_threshold', 0.8)}")
    _log(verbose, f"[sim:init] checkpoint={checkpoint} sides={policy.sides} pred_horizon={policy.pred_horizon} ode_steps={policy.num_inference_steps}")
    _log(verbose, f"[sim:init] anchor={anchor_key} T_align={_fmt_vector(T_align.reshape(-1), 3)}")
    if target_position is not None:
        _log(verbose, f"[sim:init] target_position_override={_fmt_vector(target_position)}")
    if block_position is not None:
        _log(verbose, f"[sim:init] block_position_override={_fmt_vector(block_position)}")
    anchor = static_objects.get(anchor_key)
    if anchor is not None:
        _log(verbose, f"[sim:init] anchor_position={_fmt_vector(anchor.T_in_cam[:3, 3])}")
    initial_qpos = world.data.qpos.copy()

    def run_episode() -> dict:
        run_started = perf_counter()
        world.reset()
        if world.viewer is not None:
            with world.viewer.lock():
                world.viewer.user_scn.ngeom = 0
        world.data.qpos[:] = initial_qpos
        mujoco.mj_forward(world.model, world.data)
        for arm in arms.values():
            arm.ik_failure_count = 0
        controller.reset()
        done_probability = 0.0
        reason = "max_steps"
        success_streak = 0
        result = None
        steps_completed = 0
        execution_steps_total = 0
        execution_failures_total = 0
        gated_close_total = 0
        fallback_result = None
        fallback_trigger = None
        policy_final_scene = None
        progress = None
        policy_max_lift = 0.0
        timing_totals_ms = {"capture": 0.0, "input": 0.0, "infer": 0.0, "decode": 0.0, "execute": 0.0, "total": 0.0}

        def make_result(steps: int) -> dict:
            final_scene = _scene_status(world, sim_cfg)
            failure_rate = execution_failures_total / execution_steps_total if execution_steps_total else 0.0
            return {
                "success": bool(fallback_result["success"] if fallback_result is not None else final_scene["success"]),
                "termination_reason": reason,
                "steps": steps,
                "sim_time": world.sim_time,
                "done_probability": done_probability,
                "ik_failures": arms["right"].ik_failure_count + (
                    fallback_result["ik_failures"] if fallback_result is not None else 0
                ),
                "ik_failure_rate": failure_rate,
                "device": selected_device,
                "final_ee_in_cam": arms["right"].get_T_ee_in_cam().tolist(),
                "final_scene": {
                    "success": final_scene["success"],
                    "block_position": final_scene["block_position"].tolist(),
                    "target_position": final_scene["target_position"].tolist(),
                    "xy_distance": final_scene["xy_distance"],
                    "clearance": final_scene["clearance"],
                    "linear_speed": final_scene["linear_speed"],
                },
                "timing_ms": dict(timing_totals_ms),
                "policy_only": policy_only,
                "preview_prediction": preview_prediction,
                "policy_execution_steps": execution_steps_total,
                "policy_ik_failures": execution_failures_total,
                "policy_max_lift": policy_max_lift,
                "policy_final_scene": policy_final_scene,
                "gated_close_count": gated_close_total,
                "fallback_trigger": fallback_trigger,
                "geometry_fallback": fallback_result,
                "wall_time_s": perf_counter() - run_started,
            }

        try:
            for arm in arms.values():
                arm.set_gripper(0.0, blocking=True)
                arm.go_home(blocking=True)
            initial_block_z = float(_block_world_position(world)[2])
            progress = PolicyProgressMonitor(
                geometry_cfg, _policy_progress(world, arms["right"], initial_block_z)
            ) if geometry_enabled else None
            current_position, _ = _pose_summary(arms["right"].get_T_ee_in_cam())
            _log(verbose, f"[sim:init] right_ee_position={current_position} gripper={arms['right'].get_gripper():.3f}")
            for step in range(max_steps):
                step_started = perf_counter()
                step_number = step + 1
                _log(verbose, f"\n[sim:step {step_number}] sim_time={world.sim_time:.3f}s")
                phase_started = perf_counter()
                frame = camera.get_frame()
                capture_ms = (perf_counter() - phase_started) * 1000.0
                hands = {"right": arms["right"].get_T_ee_in_cam() @ T_align}
                grippers = {"right": arms["right"].get_gripper()}
                ee_position, ee_euler = _pose_summary(hands["right"])
                valid_depth = int(np.count_nonzero(np.isfinite(frame.depth_m) & (frame.depth_m > 0)))
                _log(verbose, f"[sim:step {step_number}] frame={frame.rgb.shape[1]}x{frame.rgb.shape[0]} depth_valid={valid_depth} capture={capture_ms:.1f}ms")
                _log(verbose, f"[sim:step {step_number}] ee_position={ee_position} ee_rpy_deg={ee_euler} gripper={grippers['right']:.3f}")
                phase_started = perf_counter()
                objects = perception.estimate_objects([])
                clean = perception.make_clean_image(frame, hands, grippers)
                ict, ict_mask = policy.build_ict(hands, grippers, objects, anchor_key)
                image = policy.prepare_image(clean)
                anchor = objects.get(anchor_key)
                anchor_uv = policy.compute_anchor_uv(anchor, frame.K, frame.rgb.shape[1], frame.rgb.shape[0])
                input_ms = (perf_counter() - phase_started) * 1000.0
                ict_tokens = int(ict_mask.sum().item())
                block_position = objects.get("obj_2").T_in_cam[:3, 3] if objects.get("obj_2") else None
                block_text = f" block_position={_fmt_vector(block_position)}" if block_position is not None else ""
                anchor_uv_text = f" anchor_uv={_fmt_vector(anchor_uv[0].detach().cpu().numpy(), 3)}" if anchor_uv is not None else ""
                _log(verbose, f"[sim:step {step_number}] input ict_tokens={ict_tokens} image={tuple(image.shape)}{anchor_uv_text}{block_text} prep={input_ms:.1f}ms")
                phase_started = perf_counter()
                trajectory, done_probability = policy.infer(image, ict, ict_mask, anchor_uv)
                inference_ms = (perf_counter() - phase_started) * 1000.0
                positions, rotations, grasps = trajectory["right"]
                exec_horizon = int(control_cfg.get("exec_horizon", 8))
                _log(verbose, f"[sim:step {step_number}] policy horizon={len(positions)} done_probability={done_probability:.3f} infer={inference_ms:.1f}ms")
                if len(positions):
                    _log(verbose, f"[sim:step {step_number}] predicted_normalized_position_range={_fmt_vector(positions.min(axis=0))}..{_fmt_vector(positions.max(axis=0))} grasp_range={float(grasps.min()):.3f}..{float(grasps.max()):.3f}")
                phase_started = perf_counter()
                targets = {"right": [policy.decode_ee_in_cam(positions[i], rotations[i], anchor, T_align) for i in range(len(positions))]}
                decode_ms = (perf_counter() - phase_started) * 1000.0
                if show_prediction or preview_prediction or prediction_output is not None:
                    # Snapshot the observation's camera pose before any command or physics step.
                    T_cam_in_world = np.linalg.inv(camera.pose_world_to_camera(np.eye(3), np.zeros(3)))
                    overlay = PredictionOverlay(
                        T_cam_in_world @ np.asarray(targets["right"]), grasps,
                        controller.grasp_threshold, exec_horizon,
                    )
                    if world.viewer is not None:
                        if preview_prediction:
                            with world.viewer.lock():
                                overlay.frame_camera(world.viewer.cam, world)
                        overlay.show(world.viewer)
                    if prediction_output is not None:
                        overlay.save(world, prediction_output, {
                            "checkpoint": checkpoint, "seed": seed, "cycle": step_number,
                            "observation_sim_time": world.sim_time, "done_probability": done_probability,
                            "camera": camera.name, "T_camera_in_world": T_cam_in_world.tolist(),
                        })
                if preview_prediction:
                    steps_completed = step_number
                    reason = "prediction_preview"
                    timing_totals_ms.update(capture=capture_ms, input=input_ms, infer=inference_ms,
                                            decode=decode_ms, total=(perf_counter() - step_started) * 1000.0)
                    _log(True, "[sim:preview] Physics frozen after initialization; no predicted commands or IK. "
                         "Cyan=open, orange=close; RGB arrows=EE XYZ. Animated marker repeats the same prediction. "
                         "Close the viewer or press Ctrl-C to exit.")
                    started = perf_counter()
                    while world.viewer is not None and world.viewer.is_running():
                        overlay.show(world.viewer, int((perf_counter() - started) / dt) % len(positions))
                        sleep(0.02)
                    break
                execution = controller.execute_chunk(
                    targets, {"right": grasps}, dt, exec_horizon,
                )
                execute_ms = sum(item["elapsed_ms"] for item in execution)
                execution_steps_total += len(execution)
                execution_failures_total += sum(not item["ik_ok"] for item in execution)
                gated_close_total += sum(bool(item.get("grasp_gated")) for item in execution)
                for item in execution:
                    target_position, target_euler = _pose_summary(item["target"])
                    achieved_position, _ = _pose_summary(item["achieved"])
                    status = "OK" if item["ik_ok"] else "IK_UNREACHABLE"
                    target_world = _fmt_vector(item["ik_target_world"])
                    achieved_world = _fmt_vector(arms[item["side"]].world_position_from_camera(item["achieved"]))
                    tracking_error = np.linalg.norm(item["target"][:3, 3] - item["achieved"][:3, 3])
                    gate_distance_text = (
                        ""
                        if item.get("grasp_gate_distance") is None
                        else f" gate_dist={item['grasp_gate_distance']:.3f}m"
                        + (" GATED" if item.get("grasp_gated") else "")
                    )
                    _log(verbose, f"[sim:exec {step_number}.{item['index']}] {status} target={target_position} rpy_deg={target_euler} achieved={achieved_position} target_world={target_world} achieved_world={achieved_world} ik_pos_err={item['ik_position_error']:.4f}m tracking_err={tracking_error:.4f}m rot_err={item['ik_rotation_error_deg']:.1f}deg grasp={item['grasp_probability']:.3f} gripper={'CLOSE' if item['gripper_closed'] else 'OPEN'}{gate_distance_text} elapsed={item['elapsed_ms']:.1f}ms")
                chunk_ik_failures = sum(not item["ik_ok"] for item in execution)
                ik_failures_now = arms["right"].ik_failure_count
                scene = _scene_status(world, sim_cfg)
                policy_max_lift = max(policy_max_lift, float(scene["block_position"][2] - initial_block_z))
                success_streak = success_streak + 1 if scene["success"] else 0
                total_ms = (perf_counter() - step_started) * 1000.0
                timing_totals_ms["capture"] += capture_ms
                timing_totals_ms["input"] += input_ms
                timing_totals_ms["infer"] += inference_ms
                timing_totals_ms["decode"] += decode_ms
                timing_totals_ms["execute"] += execute_ms
                timing_totals_ms["total"] += total_ms
                _log(verbose, f"[sim:step {step_number}] scene block={_fmt_vector(scene['block_position'])} target={_fmt_vector(scene['target_position'])} xy_distance={scene['xy_distance']:.4f}m clearance={scene['clearance']:.4f}m speed={scene['linear_speed']:.4f}m/s success={scene['success']} streak={success_streak}")
                _log(verbose, f"[sim:step {step_number}] timing capture={capture_ms:.1f}ms input={input_ms:.1f}ms infer={inference_ms:.1f}ms decode={decode_ms:.1f}ms execute={execute_ms:.1f}ms total={total_ms:.1f}ms ik_failures={ik_failures_now}")
                steps_completed = step_number
                policy_done = done_probability > control_cfg.get("done_threshold", 0.8)
                if policy_done and (not geometry_enabled or scene["success"]):
                    reason = "policy_done"
                    break
                if not world.sync_viewer():
                    reason = "viewer_closed"
                    break
                if progress is not None:
                    if success_streak >= int(sim_cfg.get("success_consecutive_steps", 3)):
                        reason = "geometric_success"
                        break
                    fallback_trigger = progress.update(
                        _policy_progress(world, arms["right"], initial_block_z),
                        gated_close=any(item["grasp_gated"] for item in execution),
                        all_ik_failed=bool(execution) and chunk_ik_failures == len(execution),
                        policy_done=policy_done,
                    )
                    _log(verbose, f"[sim:progress] cycles={progress.steps} no_progress={progress.no_progress_steps} gated_streak={progress.gated_streak} ik_streak={progress.ik_failure_streak} fallback={fallback_trigger}")
                    if fallback_trigger:
                        break
            else:
                if progress is not None and progress.steps >= progress.min_steps:
                    fallback_trigger = "episode_budget_exhausted"
            policy_final_scene = make_result(steps_completed)["final_scene"]
            if fallback_trigger:
                _log(True, f"[sim:fallback] reason={fallback_trigger} after {steps_completed} policy cycles / {execution_steps_total} action commands")
                # Apply geometry-specific IK settings only after policy control ends.
                geometry_arm = SimRobotArm(
                    world, camera, "right",
                    ik_damping=sim_cfg.get("ik_damping", 0.04),
                    ik_iterations=sim_cfg.get("ik_iterations", 40),
                    ik_rotation_weight=sim_cfg.get("ik_rotation_weight", 0.35),
                    ik_position_tolerance=geometry_cfg.get("ik_position_tolerance", 0.05),
                    ik_rotation_tolerance_deg=geometry_cfg.get("ik_rotation_tolerance_deg", 10.0),
                    interpolation_steps=geometry_cfg.get("interpolation_steps", 12),
                )
                fallback_result = GeometryGraspSupervisor(
                    world, camera, geometry_arm, geometry_cfg
                ).execute(reset_home=False)
                reason = "geometry_fallback_success" if fallback_result["success"] else "geometry_fallback_failed"
            result = make_result(steps_completed)
        except ViewerClosedError:
            reason = "viewer_closed"
        except KeyboardInterrupt:
            reason = "interrupted"
            _log(True, "[sim:stop] interrupted by user")
        except Exception as exc:
            _log(True, f"[sim:stop] error={type(exc).__name__}: {exc}")
            raise
        if result is None:
            result = make_result(steps_completed)
        _log(True, f"[sim:stop] success={result['success']} reason={result['termination_reason']} steps={result['steps']} sim_time={result['sim_time']:.3f}s done_probability={result['done_probability']:.3f} ik_failures={result['ik_failures']} ik_failure_rate={result['ik_failure_rate']:.1%}")
        _log(True, f"[sim:stop] final_block={_fmt_vector(result['final_scene']['block_position'])} final_target={_fmt_vector(result['final_scene']['target_position'])} xy_distance={result['final_scene']['xy_distance']:.4f}m clearance={result['final_scene']['clearance']:.4f}m wall_time={result['wall_time_s']:.2f}s")
        _log(True, f"[sim:stop] cumulative_timing_ms={_fmt_vector([result['timing_ms'][key] for key in ('capture', 'input', 'infer', 'decode', 'execute', 'total')], 1)}")
        return result

    result = None
    episode_number = 0
    try:
        if not headless:
            world.launch_viewer()
        while True:
            episode_number += 1
            _log(True, f"[sim:episode {episode_number}] starting; Ctrl-C or close the viewer to stop")
            result = run_episode()
            if result_json is not None:
                result_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            if preview_prediction or not loop or result["termination_reason"] in {"viewer_closed", "interrupted"}:
                break
            # Hold the final pose for inspection without advancing episode physics.
            deadline = perf_counter() + loop_delay
            while perf_counter() < deadline:
                if not world.sync_viewer():
                    return result
                sleep(min(0.05, max(0.0, deadline - perf_counter())))
    except KeyboardInterrupt:
        _log(True, "[sim:stop] interrupted by user")
    finally:
        for arm in arms.values():
            arm.close()
        camera.close()
        world.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "sim_config.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    display_mode = parser.add_mutually_exclusive_group()
    display_mode.add_argument(
        "--headless",
        dest="headless",
        action="store_true",
        help="Run without a GLFW window (also the automatic default without DISPLAY).",
    )
    display_mode.add_argument(
        "--viewer",
        dest="headless",
        action="store_false",
        help="Open the GLFW viewer.",
    )
    parser.set_defaults(headless=not bool(os.environ.get("DISPLAY")))
    parser.add_argument("--max-steps", type=int, help="Maximum inference/execution cycles per episode (default: 100).")
    loop_mode = parser.add_mutually_exclusive_group()
    loop_mode.add_argument(
        "--loop",
        action="store_true",
        help="Keep starting new episodes until the viewer is closed or Ctrl-C is pressed.",
    )
    loop_mode.add_argument(
        "--once",
        action="store_true",
        help="Run exactly one episode (the default when --max-steps is supplied).",
    )
    parser.add_argument(
        "--loop-delay",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="Hold the final scene between looped episodes (default: 2).",
    )
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step simulation diagnostics.")
    parser.add_argument("--show-prediction", action="store_true",
                        help="Overlay the full raw decoded model trajectory during normal execution.")
    parser.add_argument("--preview-prediction", action="store_true",
                        help="Infer once after homing, freeze physics and display predictions until the viewer closes; no policy execution.")
    parser.add_argument("--prediction-output", type=Path, metavar="JSON_PATH",
                        help="Save the latest world-frame prediction as JSON and an overview PNG with the same stem.")
    parser.add_argument(
        "--target-position",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Override the blue tray body position in meters.",
    )
    parser.add_argument(
        "--block-position",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Override the red block position in meters.",
    )
    parser.add_argument("--seed", type=int, help="Seed NumPy and policy noise for a repeatable episode.")
    parser.add_argument(
        "--policy-only",
        action="store_true",
        help="Use the real-world reference command loop without geometric gating or early IK/success stops.",
    )
    args = parser.parse_args()
    if not np.isfinite(args.loop_delay) or args.loop_delay < 0:
        parser.error("--loop-delay must be finite and non-negative")
    if args.prediction_output is not None and args.prediction_output.suffix.lower() != ".json":
        parser.error("--prediction-output must end in .json (the preview PNG uses the same stem)")
    if args.headless and args.preview_prediction and args.prediction_output is None:
        parser.error("headless --preview-prediction requires --prediction-output to inspect the prediction")

    # Supplying --max-steps is commonly used for a bounded diagnostic run.
    # An explicit --loop still takes precedence when repeated episodes are wanted.
    should_loop = bool(args.loop) or (not args.once and args.max_steps is None)
    run_kwargs = {
        "config_path": args.config,
        "device": args.device,
        "headless": args.headless,
        "max_steps": args.max_steps,
        "verbose": not args.quiet,
        "target_position": np.asarray(args.target_position) if args.target_position is not None else None,
        "block_position": np.asarray(args.block_position) if args.block_position is not None else None,
        "seed": args.seed,
        "policy_only": args.policy_only,
        "loop": should_loop,
        "loop_delay": args.loop_delay,
        "result_json": args.result_json,
        "show_prediction": args.show_prediction,
        "preview_prediction": args.preview_prediction,
        "prediction_output": args.prediction_output,
    }

    try:
        result = run(**run_kwargs)
    except KeyboardInterrupt:
        _log(True, "[sim:stop] interrupted during initialization")
        return
    if result is not None:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
