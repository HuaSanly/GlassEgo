import io
import yaml
import unittest

import numpy as np
from PIL import Image

from inference import policy_server
from inference.policy_server import CONFIG_PATH, _load_stereo_config, create_server

try:
    from inference.proto import inference_pb2 as pb
except ImportError:  # pragma: no cover - skipped when protobuf is unavailable
    pb = None

try:
    from inference.proto import inference_pb2_grpc as pb_grpc
except ImportError:  # pragma: no cover - the test class is skipped below
    pb_grpc = None


class PolicyServerParamsTest(unittest.TestCase):
    def test_params_loads_nested_module_configs_and_calibration(self):
        with CONFIG_PATH.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        stereo = _load_stereo_config(config["stereo"], CONFIG_PATH)
        self.assertEqual(config["control"]["control_hz"], 50.0)
        self.assertEqual(config["policy"]["action_period_ns"], 100_000_000)
        self.assertEqual(stereo["left_projection"].shape, (3, 4))
        self.assertEqual(stereo["right_projection"].shape, (3, 4))
        self.assertIn("dino", config["perception"])
        self.assertIn("lama", config["perception"])


class FakePolicy:
    def prepare_image(self, image):
        return image

    def build_ict(self, arm_poses, grippers, objects, anchor_key):
        return None, None

    def compute_anchor_uv(self, anchor, K, width, height):
        return None

    def infer(self, x_rgb, x_ict, ict_mask, anchor_uv):
        positions = np.zeros((3, 3), dtype=np.float32)
        orientations = np.zeros((3, 6), dtype=np.float32)
        orientations[:, 0] = 1.0
        grippers = np.full((3, 1), 0.25, dtype=np.float32)
        return {
            "left": (positions, orientations, grippers),
            "right": (positions, orientations, grippers),
        }, 0.25

    def decode_ee_in_cam(self, position, orientation, anchor, T_align):
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = position
        return pose


class FakePerception:
    def initialize(self, frames):
        return {"obj1": type("Object", (), {"T_in_cam": np.eye(4, dtype=np.float32)})()}

    def estimate_objects(self, frame):
        return self.initialize([frame])

    def make_clean_image(self, frame, arm_poses, grippers):
        return frame.left


@unittest.skipUnless(
    policy_server.grpc is not None and pb is not None and pb_grpc is not None,
    "grpcio/protobuf are not installed",
)
class PolicyServerRoundTripTest(unittest.TestCase):
    def setUp(self):
        cfg = {
            "server": {"max_wait_ms": 100, "max_workers": 2, "max_message_mb": 8},
            "policy": {"policy_id": "ict", "device": "cpu"},
            "perception": {"anchor_key": "obj1"},
            "stereo": {
                "frame_id": "camera_optical",
                "camera_id": "head",
                "left_projection": np.array(
                    [[100.0, 0.0, 2.0, 0.0], [0.0, 100.0, 1.5, 0.0], [0.0, 0.0, 1.0, 0.0]]
                ),
                "right_projection": np.array(
                    [[100.0, 0.0, 2.0, -10.0], [0.0, 100.0, 1.5, 0.0], [0.0, 0.0, 1.0, 0.0]]
                ),
            },
            "camera": {"id": "head", "left_encoding": "jpeg", "right_encoding": "jpeg"},
            "robot": {"sides": ["left", "right"], "T_align": np.eye(4).tolist()},
            "control": {"done_threshold": 0.8},
        }
        self.server = create_server(
            cfg,
            policy=FakePolicy(),
            perception=FakePerception(),
        )
        self.port = self.server.add_insecure_port("127.0.0.1:0")
        self.server.start()
        self.channel = policy_server.grpc.insecure_channel(f"127.0.0.1:{self.port}")
        self.stub = pb_grpc.AsyncInferenceStub(self.channel)

    def tearDown(self):
        self.channel.close()
        self.server.stop(0).wait()

    @staticmethod
    def _jpeg():
        output = io.BytesIO()
        Image.new("RGB", (4, 3), (20, 40, 60)).save(output, format="JPEG")
        return output.getvalue()

    @staticmethod
    def _pose():
        return pb.Pose(
            frame_id="camera_optical",
            position_m=pb.Vector3(),
            orientation_xyzw=pb.Quaternion(w=1),
        )

    def _open(self):
        return self.stub.OpenSession(
            pb.OpenSessionRequest(
                protocol_major=1,
                policy_id="ict",
                action_period_ns=100_000_000,
                action_horizon=3,
                control_period_ns=20_000_000,
                arm_ids=["left", "right"],
                camera_ids=["head"],
            )
        )

    def _observation(self, session_id, observation_id=1, control_tick=10):
        return pb.Observation(
            session_id=session_id,
            observation_id=observation_id,
            control_tick=control_tick,
            arms=[
                pb.ArmState(arm_id="left", ee_pose=self._pose(), gripper=0),
                pb.ArmState(arm_id="right", ee_pose=self._pose(), gripper=0),
            ],
            cameras=[
                pb.CameraFrame(
                    camera_id="head",
                    width=4,
                    height=3,
                    left_encoding="jpeg",
                    left_data=self._jpeg(),
                    right_encoding="jpeg",
                    right_data=self._jpeg(),
                    rectified=True,
                )
            ],
        )

    def test_round_trip_and_tick_mapping(self):
        opened = self._open()
        self.assertEqual(opened.control_period_ns, 20_000_000)
        accepted = self.stub.SendObservation(self._observation(opened.session_id))
        self.assertTrue(accepted.accepted)

        chunk = self.stub.GetActionChunk(
            pb.ActionRequest(session_id=opened.session_id, max_wait_ms=50, last_executed_tick=10)
        )
        self.assertEqual(chunk.status, pb.ActionChunk.READY)
        self.assertEqual([step.target_tick for step in chunk.steps], [11, 16, 21])
        self.assertEqual(len(chunk.steps[0].arm_actions), 2)
        self.assertEqual(chunk.source_observation_id, 1)

        closed = self.stub.CloseSession(pb.CloseSessionRequest(session_id=opened.session_id))
        self.assertIsNotNone(closed)

    def test_stale_observation_is_rejected(self):
        opened = self._open()
        self.assertTrue(self.stub.SendObservation(self._observation(opened.session_id, 2, 2)).accepted)
        stale = self.stub.SendObservation(self._observation(opened.session_id, 1, 1))
        self.assertFalse(stale.accepted)

    def test_no_action_when_no_observation_arrives(self):
        opened = self._open()
        chunk = self.stub.GetActionChunk(
            pb.ActionRequest(session_id=opened.session_id, max_wait_ms=1)
        )
        self.assertEqual(chunk.status, pb.ActionChunk.NO_ACTION)

    def test_closed_session_rejects_observations(self):
        opened = self._open()
        self.stub.CloseSession(pb.CloseSessionRequest(session_id=opened.session_id))
        with self.assertRaises(policy_server.grpc.RpcError) as raised:
            self.stub.SendObservation(self._observation(opened.session_id))
        self.assertEqual(raised.exception.code(), policy_server.grpc.StatusCode.FAILED_PRECONDITION)

    def test_closed_session_rejects_action_requests(self):
        opened = self._open()
        self.stub.CloseSession(pb.CloseSessionRequest(session_id=opened.session_id))
        with self.assertRaises(policy_server.grpc.RpcError) as raised:
            self.stub.GetActionChunk(
                pb.ActionRequest(session_id=opened.session_id, max_wait_ms=1)
            )
        self.assertEqual(raised.exception.code(), policy_server.grpc.StatusCode.FAILED_PRECONDITION)


if __name__ == "__main__":
    unittest.main()
