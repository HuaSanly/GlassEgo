"""ROS-independent asynchronous HumanEgo policy server.

The server owns only session state, protocol handling, preprocessing hooks and
policy inference. Robot control and safety remain on the client side.
"""

from __future__ import annotations

from io import BytesIO
import os
import sys
import logging
import math
import threading
import time
import uuid
from concurrent import futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml
from google.protobuf.empty_pb2 import Empty
from PIL import Image


import cv2
import grpc

from .proto import inference_pb2 as pb
from .proto import inference_pb2_grpc as pb_grpc

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from .stereo_types import StereoFrame
from .perception import Perception
from .policy import ICTPolicy

LOGGER = logging.getLogger("policy_server")
PROTOCOL_MAJOR = 1
PROTOCOL_MINOR = 0
CONFIG_PATH = Path(__file__).resolve().parent / "config" / "params.yaml"



def _load_stereo_config(config: Any, params_path: str | Path) -> Dict[str, Any]:
    values = dict(config or {})
    calibration_path = values.get("calibration_path")
    if not calibration_path:
        return values
    calibration_file = Path(calibration_path)
    if not calibration_file.is_absolute():
        calibration_file = Path(params_path).resolve().parent / calibration_file
    with open(calibration_file, encoding="utf-8") as handle:
        calibration = yaml.safe_load(handle) or {}
    profiles = calibration.get("stereo_calibrations", [])
    if not profiles:
        raise ValueError(f"no stereo_calibrations found in {calibration_file}")
    profile_name = values.get("profile")
    selected = next((item for item in profiles if item.get("profile") == profile_name), None)
    if selected is None:
        selected = profiles[0]
    try:
        values["left_projection"] = np.asarray(selected["left"]["P"], dtype=np.float32).reshape(3, 4)
        values["right_projection"] = np.asarray(selected["right"]["P"], dtype=np.float32).reshape(3, 4)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid stereo calibration in {calibration_file}") from exc
    values["calibration"] = selected
    if "runtime_image_size" in selected:
        values.setdefault("runtime_image_size", list(selected["runtime_image_size"]))
    values["frame_id"] = str(values.get("frame_id", "camera_optical"))
    _validate_projection(values["left_projection"], "left_projection")
    _validate_projection(values["right_projection"], "right_projection")
    left_tx = values["left_projection"][0, 3] / values["left_projection"][0, 0]
    right_tx = values["right_projection"][0, 3] / values["right_projection"][0, 0]
    if abs(float(right_tx - left_tx)) <= 1e-9:
        raise ValueError("stereo calibration has zero baseline")
    return values


def _validate_projection(projection: np.ndarray, name: str) -> None:
    if projection.shape != (3, 4) or not np.all(np.isfinite(projection)):
        raise ValueError(f"{name} must be a finite 3x4 matrix")
    if projection[0, 0] <= 0 or projection[1, 1] <= 0:
        raise ValueError(f"{name} has invalid focal lengths")


@dataclass
class _DecodedObservation:
    message: pb.Observation
    frame: StereoFrame
    arm_poses: Dict[str, np.ndarray]
    grippers: Dict[str, float]


@dataclass
class _Session:
    session_id: str
    arm_ids: Tuple[str, ...]
    camera_ids: Tuple[str, ...]
    action_period_ns: int
    action_horizon: int
    control_period_ns: int
    condition: threading.Condition = field(default_factory=threading.Condition)
    inference_lock: threading.Lock = field(default_factory=threading.Lock)
    latest: Optional[_DecodedObservation] = None
    last_observation_id: int = 0
    last_control_tick: int = -1
    next_chunk_id: int = 1
    objects: Optional[Dict[str, Any]] = None
    closed: bool = False


def _finite(values: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")


def _pose_to_matrix(value: pb.Pose, expected_frame: str) -> np.ndarray:
    if not value.frame_id:
        raise ValueError("pose.frame_id is required")
    if value.frame_id != expected_frame:
        raise ValueError(f"pose frame {value.frame_id!r} does not match {expected_frame!r}")
    position = np.array([value.position_m.x, value.position_m.y, value.position_m.z], dtype=np.float32)
    quaternion = np.array(
        [value.orientation_xyzw.x, value.orientation_xyzw.y, value.orientation_xyzw.z, value.orientation_xyzw.w],
        dtype=np.float32,
    )
    _finite(position, "position_m")
    _finite(quaternion, "orientation_xyzw")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-6:
        raise ValueError("orientation quaternion must be non-zero")
    x, y, z, w = quaternion / norm
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    matrix[:3, 3] = position
    return matrix


def _matrix_to_pose(matrix: np.ndarray, frame_id: str) -> pb.Pose:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError("policy pose must have shape (4, 4)")
    _finite(matrix, "policy pose")
    rotation = matrix[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        w, x, y, z = 0.25 * scale, (rotation[2, 1] - rotation[1, 2]) / scale, (rotation[0, 2] - rotation[2, 0]) / scale, (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(max(1e-8, 1 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])) * 2
            x, y, z, w = 0.25 * scale, (rotation[0, 1] + rotation[1, 0]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale, (rotation[2, 1] - rotation[1, 2]) / scale
        elif index == 1:
            scale = math.sqrt(max(1e-8, 1 - rotation[0, 0] + rotation[1, 1] - rotation[2, 2])) * 2
            x, y, z, w = (rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale, (rotation[1, 2] + rotation[2, 1]) / scale, (rotation[0, 2] - rotation[2, 0]) / scale
        else:
            scale = math.sqrt(max(1e-8, 1 - rotation[0, 0] - rotation[1, 1] + rotation[2, 2])) * 2
            x, y, z, w = (rotation[0, 2] + rotation[2, 0]) / scale, (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale, (rotation[1, 0] - rotation[0, 1]) / scale
    return pb.Pose(
        frame_id=frame_id,
        position_m=pb.Vector3(x=float(matrix[0, 3]), y=float(matrix[1, 3]), z=float(matrix[2, 3])),
        orientation_xyzw=pb.Quaternion(x=float(x), y=float(y), z=float(z), w=float(w)),
    )


def _decode_camera(camera: pb.CameraFrame, cfg: Dict[str, Any]) -> StereoFrame:
    camera_cfg = cfg.get("camera", {})
    stereo_cfg = cfg.get("stereo", {})
    expected_left = str(camera_cfg.get("left_encoding", "jpeg")).lower()
    expected_right = str(camera_cfg.get("right_encoding", "jpeg")).lower()
    if camera.left_encoding.lower() != expected_left or not camera.left_data:
        raise ValueError("left image must be non-empty jpeg data")
    if camera.right_encoding.lower() != expected_right or not camera.right_data:
        raise ValueError("right image must be non-empty jpeg data")
    if not bool(stereo_cfg.get("rectified", True)) or not camera.rectified:
        raise ValueError("stereo images must be rectified")
    if camera.width == 0 or camera.height == 0:
        raise ValueError("camera width and height must be positive")
    expected_size = stereo_cfg.get("runtime_image_size")
    if expected_size is not None and tuple(expected_size) != (camera.width, camera.height):
        raise ValueError(
            f"camera resolution {(camera.width, camera.height)} does not match "
            f"stereo calibration {tuple(expected_size)}"
        )
    if cv2 is not None:
        left = cv2.imdecode(np.frombuffer(camera.left_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        right = cv2.imdecode(np.frombuffer(camera.right_data, dtype=np.uint8), cv2.IMREAD_COLOR)
    else:
        left = np.asarray(Image.open(BytesIO(camera.left_data)).convert("RGB"))[:, :, ::-1]
        right = np.asarray(Image.open(BytesIO(camera.right_data)).convert("RGB"))[:, :, ::-1]
    if left is None or right is None or left.shape[:2] != (camera.height, camera.width) or right.shape[:2] != (camera.height, camera.width):
        raise ValueError("stereo image data cannot be decoded at the declared resolution")
    left_projection = np.asarray(stereo_cfg.get("left_projection"), dtype=np.float32)
    right_projection = np.asarray(stereo_cfg.get("right_projection"), dtype=np.float32)
    _validate_projection(left_projection, "left_projection")
    _validate_projection(right_projection, "right_projection")
    return StereoFrame(
        left=left,
        right=right,
        left_projection=left_projection,
        right_projection=right_projection,
        capture_time_ns=int(camera.capture_time_ns),
    )


class AsyncInferenceServicer(pb_grpc.AsyncInferenceServicer if pb_grpc else object):
    def __init__(self, logger: logging.Logger | None = None):
        # Keep a single logger for the complete inference stack.  Accepting an
        # optional logger preserves the existing no-argument construction used
        # by callers while allowing applications to control handlers/levels at
        # the policy-server boundary.
        self.logger = logger if logger is not None else LOGGER
        with CONFIG_PATH.open(encoding="utf-8") as handle:
            self.cfg = yaml.safe_load(handle) or {}
        
        self.cfg["stereo"] = _load_stereo_config(self.cfg["stereo"], CONFIG_PATH)
        self.perception = Perception(self.cfg["perception"], logger=self.logger)
        self.policy = ICTPolicy(self.cfg["policy"], logger=self.logger)
        self.sessions: Dict[str, _Session] = {}
        self.sessions_lock = threading.Lock()

    def close(self) -> None:
        close = getattr(self.perception, "close", None)
        if close is not None:
            close()

    def _get_session(self, session_id: str) -> _Session:
        with self.sessions_lock:
            session = self.sessions.get(session_id)
        if session is None:
            raise ValueError("unknown session_id")
        return session

    @staticmethod
    def _abort(context, status, message):
        context.abort(status, message)

    def OpenSession(self, request, context):
        try:
            if request.protocol_major != PROTOCOL_MAJOR:
                raise ValueError(f"unsupported protocol major {request.protocol_major}")
            if request.action_period_ns <= 0 or request.action_horizon <= 0 or request.control_period_ns <= 0:
                raise ValueError("action and control periods and horizon must be positive")
            if not request.arm_ids or not request.camera_ids:
                raise ValueError("at least one arm and camera are required")
            policy_id = self.cfg["policy"].get("policy_id", self.cfg["policy"].get("id", "ict"))
            if request.policy_id and request.policy_id != policy_id:
                raise ValueError(f"unsupported policy_id {request.policy_id!r}")
            configured_arms = tuple(self.cfg["robot"].get("sides", ()))
            if configured_arms and any(arm_id not in configured_arms for arm_id in request.arm_ids):
                raise ValueError("request contains an arm not enabled by robot.sides")
            session_id = uuid.uuid4().hex
            session = _Session(
                session_id=session_id,
                arm_ids=tuple(request.arm_ids),
                camera_ids=tuple(request.camera_ids),
                action_period_ns=request.action_period_ns,
                action_horizon=request.action_horizon,
                control_period_ns=request.control_period_ns,
            )
            with self.sessions_lock:
                self.sessions[session_id] = session
            return pb.OpenSessionResponse(
                session_id=session_id,
                reference_frame_id=self.cfg["stereo"].get("frame_id", "camera_optical"),
                action_period_ns=session.action_period_ns,
                action_horizon=session.action_horizon,
                control_period_ns=session.control_period_ns,
            )
        except ValueError as exc:
            self._abort(context, grpc.StatusCode.INVALID_ARGUMENT, str(exc))

    def SendObservation(self, request, context):
        try:
            session = self._get_session(request.session_id)
            with session.condition:
                if session.closed:
                    self._abort(context, grpc.StatusCode.FAILED_PRECONDITION, "session is closed")
                if request.observation_id <= session.last_observation_id or request.control_tick <= session.last_control_tick:
                    return pb.ObservationAck(observation_id=request.observation_id, accepted=False, reason="stale observation")
                if not request.cameras or not request.arms:
                    raise ValueError("observation must contain arms and cameras")
                cameras = {camera.camera_id: camera for camera in request.cameras}
                camera_id = session.camera_ids[0]
                configured_camera_id = self.cfg["stereo"].get("camera_id", self.cfg["camera"].get("id"))
                if configured_camera_id and camera_id != configured_camera_id:
                    raise ValueError(f"camera {camera_id!r} does not match configured camera {configured_camera_id!r}")
                if camera_id not in cameras:
                    raise ValueError(f"missing configured camera {camera_id!r}")
                frame = _decode_camera(cameras[camera_id], self.cfg)
                arm_poses = {}
                grippers = {}
                for arm in request.arms:
                    if arm.arm_id not in session.arm_ids or arm.arm_id in arm_poses:
                        raise ValueError(f"invalid or duplicate arm {arm.arm_id!r}")
                    if not 0.0 <= arm.gripper <= 1.0 or not math.isfinite(arm.gripper):
                        raise ValueError(f"invalid gripper for arm {arm.arm_id!r}")
                    arm_poses[arm.arm_id] = _pose_to_matrix(
                        arm.ee_pose,
                        self.cfg["stereo"].get("frame_id", "camera_optical"),
                    )
                    grippers[arm.arm_id] = float(arm.gripper)
                session.latest = _DecodedObservation(request, frame, arm_poses, grippers)
                session.last_observation_id = request.observation_id
                session.last_control_tick = request.control_tick
                session.condition.notify_all()
            return pb.ObservationAck(observation_id=request.observation_id, accepted=True, server_queue_depth=1)
        except ValueError as exc:
            self._abort(context, grpc.StatusCode.INVALID_ARGUMENT, str(exc))

    def GetActionChunk(self, request, context):
        try:
            session = self._get_session(request.session_id)
            with session.condition:
                if session.closed:
                    self._abort(context, grpc.StatusCode.FAILED_PRECONDITION, "session is closed")
            observation = self._wait_for_observation(session, request.newest_observation_id, request.max_wait_ms)
            if observation is None:
                with session.condition:
                    if session.closed:
                        self._abort(context, grpc.StatusCode.FAILED_PRECONDITION, "session is closed")
                return self._no_action(session.session_id)
            with session.inference_lock:
                for _ in range(2):
                    chunk = self._infer(session, observation, request.last_executed_tick)
                    with session.condition:
                        latest_id = session.latest.message.observation_id if session.latest else 0
                        if latest_id == observation.message.observation_id:
                            return chunk
                        observation = session.latest
                        if observation is None:
                            return self._no_action(session.session_id)
                return self._no_action(session.session_id)
        except NotImplementedError as exc:
            self._abort(context, grpc.StatusCode.UNIMPLEMENTED, str(exc))
        except ValueError as exc:
            self._abort(context, grpc.StatusCode.INVALID_ARGUMENT, str(exc))

    def CloseSession(self, request, context):
        try:
            session = self._get_session(request.session_id)
        except ValueError as exc:
            self._abort(context, grpc.StatusCode.NOT_FOUND, str(exc))
        with session.condition:
            session.closed = True
            session.condition.notify_all()
        return Empty()

    def _wait_for_observation(self, session, newest_id: int, max_wait_ms: int):
        timeout = (max_wait_ms or self.cfg["server"].get("max_wait_ms", 100)) / 1000.0
        deadline = time.monotonic() + timeout
        with session.condition:
            while not session.closed:
                if session.latest and session.latest.message.observation_id > newest_id:
                    return session.latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                session.condition.wait(remaining)
        return None

    def _infer(self, session: _Session, observation: _DecodedObservation, last_executed_tick: int):
        if self.policy is None:
            raise NotImplementedError("no policy checkpoint is configured")
        if session.objects is None:
            session.objects = self.perception.estimate_objects(observation.frame)
        clean = self.perception.make_clean_image(
            observation.frame,
            observation.arm_poses,
            observation.grippers,
        )
        objects = session.objects
        anchor_key = self.cfg["perception"].get("anchor_key", "obj1")
        anchor = objects.get(anchor_key)
        x_rgb = self.policy.prepare_image(clean)
        left_projection = observation.frame.left_projection
        K = left_projection[:, :3]
        x_ict, ict_mask = self.policy.build_ict(
            observation.arm_poses,
            observation.grippers,
            objects,
            anchor_key,
        )
        anchor_uv = self.policy.compute_anchor_uv(
            anchor,
            K,
            observation.frame.left.shape[1],
            observation.frame.left.shape[0],
        )
        trajectory, done_probability = self.policy.infer(x_rgb, x_ict, ict_mask, anchor_uv)
        steps = []
        step_tick = max(1, round(session.action_period_ns / session.control_period_ns))
        start_tick = max(last_executed_tick + 1, observation.message.control_tick)
        for index in range(session.action_horizon):
            action_step = pb.ActionStep(target_tick=start_tick + index * step_tick)
            for arm_id in session.arm_ids:
                if arm_id not in trajectory or index >= len(trajectory[arm_id][0]):
                    continue
                positions, orientations, grasps = trajectory[arm_id]
                target = self.policy.decode_ee_in_cam(
                    positions[index],
                    orientations[index],
                    anchor,
                    np.asarray(self.cfg["robot"].get("T_align", np.eye(4)), dtype=np.float32),
                )
                action_step.arm_actions.add(
                    arm_id=arm_id,
                    target_ee_pose=_matrix_to_pose(
                        target,
                        self.cfg["stereo"].get("frame_id", "camera_optical"),
                    ),
                    gripper=float(np.asarray(grasps[index]).reshape(-1)[0]),
                )
            steps.append(action_step)
        chunk_id = session.next_chunk_id
        session.next_chunk_id += 1
        expires = steps[-1].target_tick if steps else start_tick
        return pb.ActionChunk(
            status=pb.ActionChunk.READY,
            session_id=session.session_id,
            chunk_id=chunk_id,
            source_observation_id=observation.message.observation_id,
            start_tick=start_tick,
            action_period_ns=session.action_period_ns,
            expires_at_tick=expires,
            steps=steps,
            done_probability=float(done_probability),
            terminal=float(done_probability) >= self.cfg["control"].get("done_threshold", 0.8),
        )

    @staticmethod
    def _no_action(session_id: str):
        return pb.ActionChunk(status=pb.ActionChunk.NO_ACTION, session_id=session_id)



def main() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    LOGGER.setLevel(logging.INFO)
    LOGGER.info("Run the ego policy server")

    servicer = AsyncInferenceServicer(logger=LOGGER)
    server_cfg = servicer.cfg["server"]
    max_message_mb = int(server_cfg.get("max_message_mb", 64))
    options = [
        ("grpc.max_send_message_length", max_message_mb * 1024 * 1024),
        ("grpc.max_receive_message_length", max_message_mb * 1024 * 1024),
    ]
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=int(server_cfg.get("max_workers", 8))),
        options=options,
    )
    pb_grpc.add_AsyncInferenceServicer_to_server(servicer, server)
    server._policy_servicer = servicer
    server.add_insecure_port(str(server_cfg.get("listen", "0.0.0.0:50051")))

    server.start()
    LOGGER.info("policy server listening on %s", server._policy_servicer.cfg["server"].get("listen"))
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=2)
    finally:
        server._policy_servicer.close()


if __name__ == "__main__":
    main()
