import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from preprocess.data_types.VIOTypes import (
    CameraSample,
    RawTrajectory,
    VIOCalibration,
    VIOFrame,
    VIOTrajectory,
)


class VIOPoseProcessor:
    """将 Basalt IMU 轨迹转换为逐相机帧 c2w。"""

    def __init__(self, min_pose_coverage: float, max_interpolation_gap_ms: float):
        self.min_pose_coverage = float(min_pose_coverage)
        self.max_interpolation_gap_ns = int(
            round(float(max_interpolation_gap_ms) * 1_000_000.0)
        )
        if not 0.0 < self.min_pose_coverage <= 1.0:
            raise ValueError("vio.min_pose_coverage must be in (0, 1]")
        if self.max_interpolation_gap_ns <= 0:
            raise ValueError("vio.max_interpolation_gap_ms must be positive")

    def process(
        self,
        camera: tuple[CameraSample, ...],
        raw_trajectory: RawTrajectory,
        calibration: VIOCalibration,
    ) -> VIOTrajectory:
        pose_by_timestamp = {
            int(timestamp): transform
            for timestamp, transform in zip(
                raw_trajectory.timestamps_ns,
                raw_trajectory.T_world_imu,
            )
        }
        T_world_camera = []
        interpolated = []
        raw_matches = 0
        for sample in camera:
            if sample.timestamp_ns is None:
                raise ValueError("Camera timestamp was not synchronized")
            T_world_imu = pose_by_timestamp.get(sample.timestamp_ns)
            is_interpolated = False
            if T_world_imu is not None:
                raw_matches += 1
            else:
                T_world_imu = self._interpolate_pose(
                    sample.timestamp_ns,
                    raw_trajectory,
                )
                is_interpolated = T_world_imu is not None
            T_world_camera.append(
                None
                if T_world_imu is None
                else T_world_imu @ calibration.T_imu_camera
            )
            interpolated.append(is_interpolated)

        raw_coverage = raw_matches / len(camera)
        if raw_coverage < self.min_pose_coverage:
            raise RuntimeError(
                "Basalt pose coverage is below the configured threshold: "
                f"{raw_coverage:.2%} < {self.min_pose_coverage:.0%}"
            )
        missing_indices = [
            index for index, pose in enumerate(T_world_camera) if pose is None
        ]
        if missing_indices:
            raise RuntimeError(
                "VIO poses contain an edge or long gap that cannot be interpolated: "
                f"frames={missing_indices[:10]}"
            )

        normalization = self._world_normalization(T_world_camera)
        frames = []
        for sample, pose, is_interpolated in zip(
            camera,
            T_world_camera,
            interpolated,
        ):
            normalized_pose = normalization @ pose
            self._validate_transform(normalized_pose)
            frames.append(
                VIOFrame(
                    frame_idx=sample.frame_idx,
                    timestamp_ns=int(sample.timestamp_ns),
                    c2w=normalized_pose,
                    interpolated=is_interpolated,
                )
            )
        return VIOTrajectory(
            frames=tuple(frames),
            raw_pose_coverage=raw_coverage,
        )

    def _interpolate_pose(
        self,
        timestamp_ns: int,
        trajectory: RawTrajectory,
    ) -> np.ndarray | None:
        timestamps = trajectory.timestamps_ns
        right = int(np.searchsorted(timestamps, timestamp_ns, side="left"))
        if right == 0 or right >= len(timestamps):
            return None
        left = right - 1
        interval_ns = int(timestamps[right] - timestamps[left])
        if interval_ns <= 0 or interval_ns > self.max_interpolation_gap_ns:
            return None
        ratio = (timestamp_ns - int(timestamps[left])) / interval_ns

        result = np.eye(4, dtype=np.float64)
        result[:3, 3] = (
            (1.0 - ratio) * trajectory.T_world_imu[left, :3, 3]
            + ratio * trajectory.T_world_imu[right, :3, 3]
        )
        rotations = Rotation.from_matrix(
            np.stack(
                [
                    trajectory.T_world_imu[left, :3, :3],
                    trajectory.T_world_imu[right, :3, :3],
                ]
            )
        )
        result[:3, :3] = Slerp([0.0, 1.0], rotations)([ratio]).as_matrix()[0]
        return result

    @staticmethod
    def _world_normalization(poses: list[np.ndarray]) -> np.ndarray:
        first_pose = poses[0]
        forward = first_pose[:3, 2].copy()
        forward[2] = 0.0
        norm = np.linalg.norm(forward)
        if norm < 1e-8:
            raise ValueError("First camera heading is parallel to the gravity axis")
        forward /= norm
        yaw = np.arctan2(forward[1], forward[0])
        rotation = Rotation.from_euler("z", -yaw).as_matrix()

        normalization = np.eye(4, dtype=np.float64)
        normalization[:3, :3] = rotation
        normalization[:3, 3] = -rotation @ first_pose[:3, 3]
        return normalization

    @staticmethod
    def _validate_transform(transform: np.ndarray) -> None:
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("c2w must be a finite 4x4 matrix")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("c2w rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
            raise ValueError("c2w rotation determinant must be +1")
