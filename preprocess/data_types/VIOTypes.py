import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VIOUnitPaths:
    """单个 VIO 处理单元所需的输入路径。"""

    unit_dir: Path
    video_path: Path
    camera_csv_path: Path
    imu_csv_path: Path
    calibration_path: Path


@dataclass(frozen=True)
class CameraSample:
    """相机帧及其统一到 IMU 时钟域后的时间戳。"""

    frame_idx: int
    frame_id: int
    rokid_timestamp_ns: int
    device_monotonic_ns: int
    timestamp_ns: int | None = None


@dataclass(frozen=True)
class RawIMUSample:
    """单个原始 IMU 事件。"""

    sensor_type: str
    sequence: int
    timestamp_ns: int
    values: np.ndarray


@dataclass(frozen=True)
class IMUSample:
    """在陀螺仪时间戳上对齐的六轴 IMU 样本。"""

    timestamp_ns: int
    gyroscope: np.ndarray
    accelerometer: np.ndarray


@dataclass(frozen=True)
class RawSensorData:
    camera: tuple[CameraSample, ...]
    gyroscope: tuple[RawIMUSample, ...]
    accelerometer: tuple[RawIMUSample, ...]


@dataclass(frozen=True)
class SynchronizedSensorData:
    camera: tuple[CameraSample, ...]
    imu: tuple[IMUSample, ...]
    imu_update_rate_hz: float
    clock_sync: dict
    raw_gyroscope_samples: int
    raw_accelerometer_samples: int


@dataclass(frozen=True)
class VIOCalibration:
    """GlassEgo VIO 使用的相机与 IMU 标定。"""

    resolution: tuple[int, int]
    intrinsics: np.ndarray
    distortion: np.ndarray
    T_cam_imu: np.ndarray
    T_imu_camera: np.ndarray
    timeshift_cam_imu_s: float
    noise: dict


@dataclass(frozen=True)
class RawTrajectory:
    """Basalt 输出的 T_world_imu 轨迹。"""

    timestamps_ns: np.ndarray
    T_world_imu: np.ndarray


@dataclass(frozen=True)
class VIOFrame:
    frame_idx: int
    timestamp_ns: int
    c2w: np.ndarray
    interpolated: bool = False

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "timestamp_ns": self.timestamp_ns,
            "valid": True,
            "interpolated": self.interpolated,
            "c2w": self.c2w.tolist(),
        }


@dataclass(frozen=True)
class VIOTrajectory:
    frames: tuple[VIOFrame, ...]
    raw_pose_coverage: float
    backend: str = "basalt"
    camera_frame: str = "opencv_x_right_y_down_z_forward"
    world_frame: str = "first_camera_origin_x_forward_y_left_z_up"

    def to_dict(self) -> dict:
        return {
            "schema_version": 2,
            "backend": self.backend,
            "pose_type": "c2w",
            "timestamp_unit": "ns",
            "time_domain": "android_monotonic_imu_aligned",
            "camera_frame": self.camera_frame,
            "world_frame": self.world_frame,
            "raw_pose_coverage": self.raw_pose_coverage,
            "frames": [frame.to_dict() for frame in self.frames],
        }

    def save_json(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> "VIOTrajectory":
        with Path(path).open("r", encoding="utf-8") as stream:
            document = json.load(stream)
        if document.get("schema_version") != 2:
            raise ValueError("Unsupported VIO pose schema")

        frames = []
        for item in document.get("frames", []):
            if not item.get("valid", False) or item.get("c2w") is None:
                raise ValueError("Cached VIO trajectory contains an invalid pose")
            c2w = np.asarray(item["c2w"], dtype=np.float64)
            _validate_transform(c2w, "c2w")
            frames.append(
                VIOFrame(
                    frame_idx=int(item["frame_idx"]),
                    timestamp_ns=int(item["timestamp_ns"]),
                    c2w=c2w,
                    interpolated=bool(item.get("interpolated", False)),
                )
            )
        if not frames:
            raise ValueError("Cached VIO trajectory contains no frames")
        _validate_frame_sequence(frames)
        return cls(
            frames=tuple(frames),
            raw_pose_coverage=float(document["raw_pose_coverage"]),
            backend=str(document.get("backend", "basalt")),
            camera_frame=str(document["camera_frame"]),
            world_frame=str(document["world_frame"]),
        )


@dataclass(frozen=True)
class VIOResult:
    trajectory: VIOTrajectory
    calibration: VIOCalibration
    report: dict
    pose_path: Path
    trajectory_path: Path
    log_path: Path


@dataclass(frozen=True)
class BasaltRunResult:
    trajectory: RawTrajectory
    trajectory_path: Path
    log_path: Path
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _validate_transform(transform: np.ndarray, label: str) -> None:
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{label} must be a homogeneous transform")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{label} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{label} rotation determinant must be +1")


def _validate_frame_sequence(frames: list[VIOFrame]) -> None:
    for expected_idx, frame in enumerate(frames):
        if frame.frame_idx != expected_idx:
            raise ValueError("VIO frame_idx must be contiguous and start at zero")
        if expected_idx and frame.timestamp_ns <= frames[expected_idx - 1].timestamp_ns:
            raise ValueError("VIO timestamps must be strictly increasing")
