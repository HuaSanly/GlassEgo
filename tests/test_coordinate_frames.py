import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from preprocess.data_types.VIOTypes import (
    ARIA_MPS_WORLD_FRAME,
    BasaltTrajectory,
    CameraSample,
    VIOCalibration,
    VIOFrame,
    VIOTrajectory,
)
from preprocess.phase_segmentation.PhaseSegmentationGenerator import (
    PhaseSegmentationGenerator,
)
from preprocess.vio.BasaltAdapter import BasaltAdapter
from preprocess.vio.BasaltVIOGenerator import BasaltVIOGenerator
from preprocess.vio.VIOPoseProcessor import VIOPoseProcessor


def _transform(rotation=None, translation=None):
    result = np.eye(4, dtype=np.float64)
    if rotation is not None:
        result[:3, :3] = rotation
    if translation is not None:
        result[:3, 3] = translation
    return result


def _heading_rotation(yaw_deg):
    yaw = np.radians(yaw_deg)
    right = np.array([np.cos(yaw), 0.0, np.sin(yaw)])
    down = np.array([0.0, -1.0, 0.0])
    forward = np.array([np.sin(yaw), 0.0, -np.cos(yaw)])
    return np.column_stack([right, down, forward])


class AriaMPSCoordinateTests(unittest.TestCase):
    def setUp(self):
        self.processor = VIOPoseProcessor(
            min_pose_coverage=1.0,
            max_interpolation_gap_ms=100.0,
        )
        self.timestamps = np.array([100_000_000, 200_000_000], dtype=np.int64)
        self.camera = tuple(
            CameraSample(
                frame_idx=index,
                frame_id=index,
                rokid_timestamp_ns=int(timestamp),
                device_monotonic_ns=int(timestamp),
                timestamp_ns=int(timestamp),
            )
            for index, timestamp in enumerate(self.timestamps)
        )
        self.calibration = VIOCalibration(
            resolution=(640, 480),
            intrinsics=np.array([400.0, 400.0, 320.0, 240.0]),
            distortion=np.zeros(4),
            T_cam_imu=np.eye(4),
            T_imu_camera=np.eye(4),
            timeshift_cam_imu_s=0.0,
            noise={},
        )

    def _trajectory(self):
        # Camera +Z faces Basalt +X, camera +X faces Basalt -Y, and +Z is up.
        R_basalt_camera = np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )
        transforms = np.stack(
            [
                _transform(R_basalt_camera, [1.0, 2.0, 3.0]),
                _transform(R_basalt_camera, [2.0, 2.0, 3.0]),
            ]
        )
        return BasaltTrajectory(self.timestamps, transforms)

    def test_alignment_defines_right_up_backward_world(self):
        basalt = self._trajectory()
        result = self.processor.process(self.camera, basalt, self.calibration)
        first = result.frames[0].c2w

        np.testing.assert_allclose(first[:3, 3], np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(first[:3, 0], [1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(first[:3, 1], [0.0, -1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(first[:3, 2], [0.0, 0.0, -1.0], atol=1e-12)
        np.testing.assert_allclose(
            result.T_world_basalt[:3, :3] @ [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.T_world_basalt[:3, 1],
            [-1.0, 0.0, 0.0],
            atol=1e-12,
        )
        self.assertAlmostEqual(np.linalg.det(result.T_world_basalt[:3, :3]), 1.0)
        self.assertEqual(result.world_frame, ARIA_MPS_WORLD_FRAME)

    def test_camera_and_imu_use_the_same_world_alignment(self):
        basalt = self._trajectory()
        result = self.processor.process(self.camera, basalt, self.calibration)
        T_world_imu = self.processor.transform_basalt_trajectory(
            basalt,
            result.T_world_basalt,
        )
        for T_imu, frame in zip(T_world_imu, result.frames):
            np.testing.assert_allclose(
                T_imu @ self.calibration.T_imu_camera,
                frame.c2w,
                atol=1e-12,
            )

    def test_heading_parallel_to_gravity_is_rejected(self):
        basalt = BasaltTrajectory(
            self.timestamps,
            np.stack([np.eye(4), np.eye(4)]),
        )
        with self.assertRaisesRegex(ValueError, "parallel to the gravity axis"):
            self.processor.process(self.camera, basalt, self.calibration)

    def test_pose_schema_round_trip_and_old_schema_rejection(self):
        result = self.processor.process(
            self.camera,
            self._trajectory(),
            self.calibration,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "poses.json"
            result.save_json(path)
            loaded = VIOTrajectory.load_json(path)
            self.assertEqual(loaded.world_frame, ARIA_MPS_WORLD_FRAME)
            np.testing.assert_allclose(loaded.T_world_basalt, result.T_world_basalt)

            document = result.to_dict()
            document["schema_version"] = 2
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported VIO pose schema"):
                VIOTrajectory.load_json(path)
        self.assertGreaterEqual(BasaltVIOGenerator.CACHE_VERSION, 2)

    def test_world_csv_round_trip_and_metadata(self):
        basalt = self._trajectory()
        result = self.processor.process(self.camera, basalt, self.calibration)
        T_world_imu = self.processor.transform_basalt_trajectory(
            basalt,
            result.T_world_basalt,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "basalt_trajectory.csv"
            BasaltAdapter.write_world_trajectory(
                basalt.timestamps_ns,
                T_world_imu,
                path,
            )
            source = path.read_text(encoding="utf-8")
            self.assertIn(f"#world_frame={ARIA_MPS_WORLD_FRAME}", source)
            parsed = BasaltAdapter._parse_trajectory(path)
            timestamps, transforms = parsed.timestamps_ns, parsed.T_basalt_imu
            np.testing.assert_array_equal(timestamps, basalt.timestamps_ns)
            np.testing.assert_allclose(transforms, T_world_imu, atol=1e-12)

    def test_phase_yaw_is_zero_initially_and_positive_to_the_right(self):
        frames = tuple(
            VIOFrame(
                frame_idx=index,
                timestamp_ns=index * 1_000_000_000,
                c2w=_transform(_heading_rotation(yaw)),
            )
            for index, yaw in enumerate((0.0, 90.0))
        )
        _, _, linear_speed, angular_speed, yaw = (
            PhaseSegmentationGenerator._kinematics(frames)
        )
        np.testing.assert_allclose(yaw, [0.0, 90.0], atol=1e-12)
        np.testing.assert_allclose(linear_speed, [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(angular_speed, [0.0, np.pi / 2.0], atol=1e-12)

    def test_phase_yaw_unwraps_across_180_degrees(self):
        frames = tuple(
            VIOFrame(
                frame_idx=index,
                timestamp_ns=index * 1_000_000_000,
                c2w=_transform(_heading_rotation(yaw)),
            )
            for index, yaw in enumerate((170.0, -170.0))
        )
        _, _, _, angular_speed, yaw = PhaseSegmentationGenerator._kinematics(frames)
        np.testing.assert_allclose(yaw, [170.0, 190.0], atol=1e-12)
        self.assertAlmostEqual(angular_speed[1], np.radians(20.0))

    def test_se3_evaluation_is_invariant_to_world_alignment(self):
        timestamps = np.array([100, 200, 300, 400], dtype=np.int64)
        poses = np.stack(
            [
                _transform(_heading_rotation(yaw), [index, 0.0, -0.5 * index])
                for index, yaw in enumerate((0.0, 5.0, 10.0, 15.0))
            ]
        )
        alignment = _transform(_heading_rotation(35.0), [2.0, -1.0, 4.0])
        aligned = np.einsum("ij,njk->nik", alignment, poses)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gt_path = root / "ground_truth.csv"
            estimate_path = root / "estimate.csv"
            BasaltAdapter.write_world_trajectory(timestamps, poses, gt_path)
            BasaltAdapter.write_world_trajectory(timestamps, aligned, estimate_path)
            estimate = BasaltAdapter._parse_trajectory(estimate_path)
            ground_truth = BasaltAdapter._parse_trajectory(gt_path)
        self.assertEqual(estimate.timestamps_ns.tolist(), ground_truth.timestamps_ns.tolist())
        gt_relative = np.linalg.inv(ground_truth.T_basalt_imu[0]) @ ground_truth.T_basalt_imu
        estimate_relative = np.linalg.inv(estimate.T_basalt_imu[0]) @ estimate.T_basalt_imu
        np.testing.assert_allclose(estimate_relative, gt_relative, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
