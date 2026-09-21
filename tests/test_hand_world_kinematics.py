import unittest
from types import SimpleNamespace

import numpy as np
import cv2
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from preprocess.data_types.CamTypes import CamData
from preprocess.data_types.HandsTypes import HandData, Hands, HandsData, MidpointFrameBuilder
from preprocess.hand_tracking.HaMeRHandsGenerator import HaMeRHandsGenerator
from preprocess.hand_tracking.HandsTrajectoryOptimizer import HandsTrajectoryOptimizer


def _pose(angle_deg: float, translation) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_euler("z", angle_deg, degrees=True).as_matrix()
    pose[:3, 3] = translation
    return pose


def _hand(raw_pose=None) -> HandData:
    return HandData(wrist_pose_raw_world=raw_pose)


def _keypoints(offset=0.0) -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float64)
    points[:, 2] = 1.0
    points[0] = [0.0 + offset, 0.0, 1.0]
    points[2] = [-0.05 + offset, 0.1, 1.0]
    points[4] = [-0.1 + offset, 0.15, 1.0]
    points[5] = [0.05 + offset, 0.1, 1.0]
    points[8] = [0.15 + offset, 0.15, 1.0]
    points[9] = [0.1 + offset, 0.1, 1.0]
    points[13] = [0.15 + offset, 0.08, 1.0]
    points[17] = [0.2 + offset, 0.06, 1.0]
    return points


def _camera_pose(translation) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = translation
    return pose


class HandWorldKinematicsTests(unittest.TestCase):
    def test_so3_ema_preserves_valid_rotation_and_constant_pose(self):
        identity = np.eye(3)
        result = HandsTrajectoryOptimizer._ema_rotation(identity, None, 0.2)
        np.testing.assert_allclose(result, identity)
        np.testing.assert_allclose(result.T @ result, identity, atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(result), 1.0, places=12)

    def test_so3_ema_interpolates_rotation_without_axiswise_artifacts(self):
        previous = Rotation.from_euler("xyz", [20.0, -10.0, 15.0], degrees=True).as_matrix()
        current = Rotation.from_euler("xyz", [80.0, 40.0, -25.0], degrees=True).as_matrix()
        result = HandsTrajectoryOptimizer._ema_rotation(current, previous, 0.2)
        relative = Rotation.from_matrix(previous.T @ result)
        target = Rotation.from_matrix(previous.T @ current)
        self.assertAlmostEqual(relative.magnitude(), target.magnitude() * 0.2, places=10)
        np.testing.assert_allclose(result.T @ result, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(result), 1.0, places=12)

    def test_speed_gate_removes_tail_outliers_and_keeps_prior_pose(self):
        hands = Hands(
            hands=[
                HandsData(ts=0, hand_r=_hand(_pose(0.0, [0.0, 0.0, 0.0]))),
                HandsData(ts=100_000_000, hand_r=_hand(_pose(0.0, [0.1, 0.0, 0.0]))),
                HandsData(ts=200_000_000, hand_r=_hand(_pose(0.0, [2.0, 0.0, 0.0]))),
                HandsData(ts=300_000_000, hand_r=_hand(_pose(0.0, [2.1, 0.0, 0.0]))),
            ]
        )
        removed = HandsTrajectoryOptimizer.remove_excessive_speed_hands(
            hands, 2.5, "wrist_pose_raw_world"
        )
        self.assertEqual(removed, 2)
        self.assertIsNotNone(hands.hands[1].hand_r)
        self.assertIsNone(hands.hands[2].hand_r)
        self.assertIsNone(hands.hands[3].hand_r)

    def test_velocity_recomputation_uses_frame_timestamps_and_resets_on_gap(self):
        cfg = OmegaConf.create({
            "sg_window": 9,
            "sg_polyorder": 2,
            "min_valid_frames": 15,
            "ema_alpha_rotation": 0.2,
        })
        hands = Hands(
            hands=[
                HandsData(ts=0, hand_r=None),
                HandsData(ts=100_000_000, hand_r=_hand(_pose(0.0, [1.0, 0.0, 0.0]))),
                HandsData(ts=350_000_000, hand_r=_hand(_pose(0.0, [1.5, 0.0, 0.0]))),
            ]
        )
        for frame in hands.hands[1:]:
            hand = frame.hand_r
            hand.midpoint_pose_raw_world = _pose(0.0, [1.0, 0.0, 1.0])
            hand.midpoint_translation_raw_world = np.array([1.0, 0.0, 1.0])
            hand.wrist_pose_opt_world = hand.wrist_pose_raw_world.copy()
            hand.midpoint_pose_opt_world = hand.midpoint_pose_raw_world.copy()
            hand.midpoint_translation_opt_world = hand.midpoint_translation_raw_world.copy()
        HandsTrajectoryOptimizer(cfg).assign_velocities(hands)
        np.testing.assert_allclose(hands.hands[1].hand_r.wrist_lin_vel_raw_world, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(hands.hands[2].hand_r.wrist_lin_vel_raw_world, [2.0, 0.0, 0.0])

    def test_short_gap_interpolation_uses_world_points_and_current_c2w(self):
        generator = HaMeRHandsGenerator.__new__(HaMeRHandsGenerator)
        generator.mid_frame_builder = MidpointFrameBuilder()
        camera_matrix = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
        camera_frames = []
        for translation in ([0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.1, 0.0, 0.0]):
            camera_frames.append(
                CamData(
                    idx=len(camera_frames),
                    ts=len(camera_frames) * 100,
                    k=camera_matrix,
                    d=np.zeros(5),
                    c2w=_camera_pose(translation),
                )
            )
        generator.cam = SimpleNamespace(cam=camera_frames)

        start_hand = HandData(
            c2w=camera_frames[0].c2w,
            is_right=True,
            confidence=1.0,
            wrist_pose=_camera_pose([0.0, 0.0, 1.0]),
            hand_keypoints_3d=_keypoints(0.0),
            hand_keypoints_2d=np.zeros((21, 2)),
        )
        end_hand = HandData(
            c2w=camera_frames[2].c2w,
            is_right=True,
            confidence=1.0,
            wrist_pose=_camera_pose([0.0, 0.0, 1.0]),
            hand_keypoints_3d=_keypoints(0.0),
            hand_keypoints_2d=np.zeros((21, 2)),
        )
        generator._assign_world_kinematics(HandsData(hand_r=start_hand), camera_frames[0].c2w)
        generator._assign_world_kinematics(HandsData(hand_r=end_hand), camera_frames[2].c2w)
        hands = Hands(
            tss=[0, 100, 200],
            hands=[
                HandsData(idx=0, ts=0, hand_r=start_hand),
                HandsData(idx=1, ts=100),
                HandsData(idx=2, ts=200, hand_r=end_hand),
            ],
        )

        generator._interpolate_hand_trajectories(hands, max_gap=6)
        interpolated = hands.hands[1].hand_r
        self.assertEqual(interpolated.tracking_state, "interpolated")
        np.testing.assert_allclose(interpolated.c2w, camera_frames[1].c2w)
        np.testing.assert_allclose(
            interpolated.hand_keypoints_3d @ interpolated.c2w[:3, :3].T + interpolated.c2w[:3, 3],
            0.5 * (
                start_hand.hand_keypoints_3d @ start_hand.c2w[:3, :3].T + start_hand.c2w[:3, 3]
                + end_hand.hand_keypoints_3d @ end_hand.c2w[:3, :3].T + end_hand.c2w[:3, 3]
            ),
        )
        np.testing.assert_allclose(
            interpolated.hand_keypoints_2d,
            cv2.projectPoints(
                interpolated.hand_keypoints_3d,
                np.zeros(3),
                np.zeros(3),
                camera_matrix,
                np.zeros(5),
            )[0].reshape(21, 2),
        )

    def test_incomplete_geometry_is_not_interpolated(self):
        generator = HaMeRHandsGenerator.__new__(HaMeRHandsGenerator)
        generator.mid_frame_builder = MidpointFrameBuilder()
        generator.cam = SimpleNamespace(cam=[CamData(c2w=np.eye(4), k=np.eye(3), d=np.zeros(5)) for _ in range(3)])
        hands = Hands(
            tss=[0, 1, 2],
            hands=[HandsData(ts=0, hand_r=_hand(_pose(0, [0, 0, 0]))), HandsData(ts=1), HandsData(ts=2, hand_r=_hand(_pose(0, [0, 0, 0])))],
        )
        generator._interpolate_hand_trajectories(hands)
        self.assertIsNone(hands.hands[1].hand_r)


if __name__ == "__main__":
    unittest.main()
