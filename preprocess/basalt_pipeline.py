import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation, Slerp

PREPROCESS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PREPROCESS_ROOT.parent
DEFAULT_CONFIG_ROOT = PREPROCESS_ROOT / "config"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PREPROCESS_ROOT) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_ROOT))

from utils.utils_math import time_it


@dataclass(frozen=True)
class BasaltProcessUnit:
    """表示一个独立的 Basalt 验证单元。"""

    unit_dir: Path
    video_path: Path
    camera_csv_path: Path
    imu_csv_path: Path
    calibration_path: Path
    ground_truth_path: Path | None = None


class BasaltPipeline:
    """Basalt 单目 VIO 适配与验证管线。"""

    CAMERA_COLUMNS = ("frame_idx", "timestamp_ns", "exposure_ns")
    IMU_COLUMNS = (
        "timestamp_ns",
        "wx_rad_s",
        "wy_rad_s",
        "wz_rad_s",
        "ax_m_s2",
        "ay_m_s2",
        "az_m_s2",
    )
    MIN_POSE_COVERAGE = 0.90

    def __init__(
        self,
        config_root: str | Path = DEFAULT_CONFIG_ROOT,
    ):
        self.cfg = self._load_config(config_root)

    @time_it
    def run(self, unit: BasaltProcessUnit) -> dict:
        """处理一个单目视频与 IMU 单元并保存验证结果。"""
        if not isinstance(unit, BasaltProcessUnit):
            raise TypeError("unit must be a BasaltProcessUnit")

        output_dir = unit.unit_dir / "preprocess" / "vio"
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "basalt.log"
        report_path = output_dir / "report.json"

        try:
            sensor_data = self._load_sensor_data(unit)
            calibration = self._load_calibration(unit.calibration_path)

            with tempfile.TemporaryDirectory(prefix="glassego_basalt_") as tmp:
                temp_dir = Path(tmp)
                bag_path = temp_dir / "input.bag"
                self._write_temporary_bag(
                    unit.video_path,
                    sensor_data,
                    calibration,
                    bag_path,
                )
                self._run_basalt(
                    bag_path,
                    unit.calibration_path,
                    temp_dir,
                    log_path,
                )
                trajectory_path = temp_dir / "trajectory.csv"
                timestamps_ns, t_w_i = self._parse_trajectory(trajectory_path)
                saved_trajectory_path = output_dir / "basalt_trajectory.csv"
                shutil.copy2(trajectory_path, saved_trajectory_path)
            frames, coverage = self._build_camera_poses(
                sensor_data["camera"],
                timestamps_ns,
                t_w_i,
                calibration["T_i_c"],
            )
            if coverage < self.MIN_POSE_COVERAGE:
                raise RuntimeError(
                    "Basalt pose coverage is below the validation threshold: "
                    f"{coverage:.2%} < {self.MIN_POSE_COVERAGE:.0%}"
                )

            poses_path = output_dir / "poses.json"
            self._save_poses(poses_path, frames)
            metrics = self._evaluate_trajectory(
                unit.ground_truth_path,
                timestamps_ns,
                t_w_i,
            )
            report = {
                "status": "completed",
                "unit_dir": str(unit.unit_dir),
                "video_path": str(unit.video_path),
                "camera_frames": len(sensor_data["camera"]),
                "imu_samples": len(sensor_data["imu"]),
                "trajectory_poses": len(timestamps_ns),
                "pose_coverage": coverage,
                "metrics": metrics,
                "outputs": {
                    "trajectory": str(saved_trajectory_path),
                    "poses": str(poses_path),
                    "log": str(log_path),
                },
            }
            self._save_report(report_path, report)
            return report
        except Exception as exc:
            self._save_report(
                report_path,
                {
                    "status": "failed",
                    "unit_dir": str(unit.unit_dir),
                    "error": f"{type(exc).__name__}: {exc}",
                    "log": str(log_path),
                },
            )
            raise

    def _load_sensor_data(self, unit: BasaltProcessUnit) -> dict:
        """读取并校验相机与 IMU CSV。"""
        if not unit.video_path.is_file():
            raise FileNotFoundError(f"Video not found: {unit.video_path}")
        if unit.video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video format: {unit.video_path}")

        camera = self._read_csv(unit.camera_csv_path, self.CAMERA_COLUMNS)
        imu = self._read_csv(unit.imu_csv_path, self.IMU_COLUMNS)
        if not camera:
            raise ValueError("camera.csv contains no frames")
        if not imu:
            raise ValueError("imu.csv contains no samples")

        camera_rows = []
        for row_index, row in enumerate(camera):
            frame_idx = self._parse_int(row["frame_idx"], "frame_idx", row_index)
            timestamp_ns = self._parse_int(
                row["timestamp_ns"], "timestamp_ns", row_index
            )
            exposure_ns = self._parse_int(
                row["exposure_ns"], "exposure_ns", row_index
            )
            if frame_idx != row_index:
                raise ValueError(
                    "camera.csv frame_idx must be contiguous and start at zero: "
                    f"row={row_index}, frame_idx={frame_idx}"
                )
            camera_rows.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp_ns": timestamp_ns,
                    "exposure_ns": exposure_ns,
                }
            )

        imu_rows = []
        for row_index, row in enumerate(imu):
            timestamp_ns = self._parse_int(
                row["timestamp_ns"], "timestamp_ns", row_index
            )
            values = np.array(
                [
                    self._parse_float(row[column], column, row_index)
                    for column in self.IMU_COLUMNS[1:]
                ],
                dtype=np.float64,
            )
            imu_rows.append({"timestamp_ns": timestamp_ns, "values": values})

        self._validate_timestamps(
            [row["timestamp_ns"] for row in camera_rows], "camera.csv"
        )
        self._validate_timestamps(
            [row["timestamp_ns"] for row in imu_rows], "imu.csv"
        )
        if imu_rows[0]["timestamp_ns"] > camera_rows[0]["timestamp_ns"]:
            raise ValueError("IMU data starts after the first camera frame")
        if imu_rows[-1]["timestamp_ns"] < camera_rows[-1]["timestamp_ns"]:
            raise ValueError("IMU data ends before the last camera frame")

        return {"camera": camera_rows, "imu": imu_rows}

    def _write_temporary_bag(
        self,
        video_path: Path,
        sensor_data: dict,
        calibration: dict,
        bag_path: Path,
    ) -> None:
        """流式解码视频并生成 Basalt 可读的单目 ROS1 bag。"""
        try:
            from rosbags.rosbag1 import Writer
            from rosbags.typesys import Stores, get_typestore
        except ImportError as exc:
            raise ImportError(
                "rosbags is required; install requirements.txt before running"
            ) from exc

        typestore = get_typestore(Stores.ROS1_NOETIC)
        Header = typestore.types["std_msgs/msg/Header"]
        Time = typestore.types["builtin_interfaces/msg/Time"]
        Image = typestore.types["sensor_msgs/msg/Image"]
        Imu = typestore.types["sensor_msgs/msg/Imu"]
        Quaternion = typestore.types["geometry_msgs/msg/Quaternion"]
        Vector3 = typestore.types["geometry_msgs/msg/Vector3"]

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open video: {video_path}")

        expected_width, expected_height = calibration["resolution"]
        camera_rows = sensor_data["camera"]
        imu_rows = sensor_data["imu"]
        frame_count = 0

        try:
            with Writer(bag_path) as writer:
                image_type = Image.__msgtype__
                imu_type = Imu.__msgtype__
                image_connection = writer.add_connection(
                    "/cam0/image_raw", image_type, typestore=typestore
                )
                imu_connection = writer.add_connection(
                    "/imu0", imu_type, typestore=typestore
                )

                events = []
                for row in imu_rows:
                    events.append((row["timestamp_ns"], 0, row))
                for row in camera_rows:
                    events.append((row["timestamp_ns"], 1, row))
                events.sort(key=lambda item: (item[0], item[1]))

                for timestamp_ns, event_type, row in events:
                    stamp = self._to_ros_time(Time, timestamp_ns)
                    if event_type == 0:
                        values = row["values"]
                        message = Imu(
                            Header(0, stamp, "imu"),
                            Quaternion(0.0, 0.0, 0.0, 0.0),
                            np.full(9, -1.0, dtype=np.float64),
                            Vector3(*values[:3]),
                            np.full(9, -1.0, dtype=np.float64),
                            Vector3(*values[3:]),
                            np.full(9, -1.0, dtype=np.float64),
                        )
                        writer.write(
                            imu_connection,
                            timestamp_ns,
                            typestore.serialize_ros1(message, imu_type),
                        )
                        continue

                    ok, frame_bgr = cap.read()
                    if not ok:
                        raise ValueError(
                            "Video contains fewer readable frames than camera.csv: "
                            f"expected={len(camera_rows)}, decoded={frame_count}"
                        )
                    height, width = frame_bgr.shape[:2]
                    if (width, height) != (expected_width, expected_height):
                        raise ValueError(
                            "Video resolution does not match calibration: "
                            f"video={width}x{height}, "
                            f"calibration={expected_width}x{expected_height}"
                        )
                    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                    exposure_ns = row["exposure_ns"]
                    frame_id = str(exposure_ns) if exposure_ns >= 0 else ""
                    message = Image(
                        Header(row["frame_idx"], stamp, frame_id),
                        height,
                        width,
                        "mono8",
                        0,
                        width,
                        frame_gray.reshape(-1),
                    )
                    writer.write(
                        image_connection,
                        timestamp_ns,
                        typestore.serialize_ros1(message, image_type),
                    )
                    frame_count += 1

                extra_frame, _ = cap.read()
                if extra_frame:
                    raise ValueError(
                        "Video contains more readable frames than camera.csv: "
                        f"camera_rows={len(camera_rows)}"
                    )
        finally:
            cap.release()

    def _run_basalt(
        self,
        bag_path: Path,
        calibration_path: Path,
        temp_dir: Path,
        log_path: Path,
    ) -> None:
        """运行 Basalt 并保存完整子进程日志。"""
        executable = str(self.cfg.vio.basalt.executable)
        resolved_executable = shutil.which(executable)
        if resolved_executable is None:
            raise FileNotFoundError(f"Basalt executable not found: {executable}")

        timeout_seconds = float(self.cfg.vio.basalt.timeout_seconds)
        if timeout_seconds <= 0:
            raise ValueError("sensors.vio.basalt.timeout_seconds must be positive")

        command = [
            resolved_executable,
            "--dataset-path",
            str(bag_path),
            "--dataset-type",
            "bag",
            "--cam-calib",
            str(calibration_path),
            "--show-gui",
            "0",
            "--use-imu",
            "1",
            "--save-trajectory",
            "euroc",
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=temp_dir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            log_path.write_text(output, encoding="utf-8")
            raise TimeoutError(
                f"Basalt exceeded timeout of {timeout_seconds:g} seconds"
            ) from exc

        log_path.write_text(completed.stdout or "", encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"Basalt failed with exit code {completed.returncode}; "
                f"see {log_path}"
            )

    def _parse_trajectory(self, trajectory_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """解析 Basalt EuroC 格式的 T_world_imu 轨迹。"""
        if not trajectory_path.is_file() or trajectory_path.stat().st_size == 0:
            raise FileNotFoundError(f"Basalt trajectory not found: {trajectory_path}")

        timestamps = []
        transforms = []
        with trajectory_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            for row_index, row in enumerate(reader):
                if not row or row[0].lstrip().startswith("#"):
                    continue
                if len(row) != 8:
                    raise ValueError(
                        f"Invalid Basalt trajectory row {row_index}: {row}"
                    )
                timestamp_ns = int(row[0])
                values = np.asarray([float(value) for value in row[1:]], dtype=np.float64)
                if not np.all(np.isfinite(values)):
                    raise ValueError(
                        f"Non-finite Basalt trajectory row {row_index}: {row}"
                    )
                position = values[:3]
                quaternion_wxyz = values[3:]
                rotation = self._rotation_from_wxyz(quaternion_wxyz)
                transform = np.eye(4, dtype=np.float64)
                transform[:3, :3] = rotation
                transform[:3, 3] = position
                timestamps.append(timestamp_ns)
                transforms.append(transform)

        if not timestamps:
            raise ValueError("Basalt trajectory contains no poses")
        self._validate_timestamps(timestamps, "Basalt trajectory")
        return np.asarray(timestamps, dtype=np.int64), np.stack(transforms)

    def _build_camera_poses(
        self,
        camera_rows: list[dict],
        pose_timestamps_ns: np.ndarray,
        t_w_i: np.ndarray,
        t_i_c: np.ndarray,
    ) -> tuple[list[dict], float]:
        """将 IMU 轨迹精确对齐到相机帧并转换为 c2w。"""
        pose_by_timestamp = {
            int(timestamp): transform
            for timestamp, transform in zip(pose_timestamps_ns, t_w_i)
        }
        valid_poses = []
        for row in camera_rows:
            transform = pose_by_timestamp.get(row["timestamp_ns"])
            valid_poses.append(None if transform is None else transform @ t_i_c)

        matched = sum(pose is not None for pose in valid_poses)
        coverage = matched / len(camera_rows)
        normalization = self._world_normalization(valid_poses)

        frames = []
        for row, pose in zip(camera_rows, valid_poses):
            normalized_pose = None if pose is None else normalization @ pose
            frames.append(
                {
                    "frame_idx": row["frame_idx"],
                    "timestamp_ns": row["timestamp_ns"],
                    "valid": normalized_pose is not None,
                    "c2w": (
                        None if normalized_pose is None else normalized_pose.tolist()
                    ),
                }
            )
        return frames, coverage

    def _evaluate_trajectory(
        self,
        ground_truth_path: Path | None,
        pose_timestamps_ns: np.ndarray,
        t_w_i: np.ndarray,
    ) -> dict | None:
        """使用可选真值生成 SE(3) 对齐后的 ATE 与 RPE。"""
        if ground_truth_path is None:
            return None
        gt_timestamps_ns, gt_t_w_i = self._load_ground_truth(ground_truth_path)
        estimate = []
        ground_truth = []
        for timestamp, transform in zip(pose_timestamps_ns, t_w_i):
            gt_transform = self._interpolate_pose(
                int(timestamp), gt_timestamps_ns, gt_t_w_i
            )
            if gt_transform is not None:
                estimate.append(transform)
                ground_truth.append(gt_transform)
        if len(estimate) < 3:
            raise ValueError(
                "Ground truth has fewer than three exact timestamp associations"
            )

        estimate = np.stack(estimate)
        ground_truth = np.stack(ground_truth)
        alignment = self._align_se3(
            estimate[:, :3, 3], ground_truth[:, :3, 3]
        )
        aligned_estimate = np.einsum("ij,njk->nik", alignment, estimate)

        translation_error = np.linalg.norm(
            aligned_estimate[:, :3, 3] - ground_truth[:, :3, 3], axis=1
        )
        rpe_translation = []
        rpe_rotation_deg = []
        for index in range(1, len(aligned_estimate)):
            estimate_delta = (
                np.linalg.inv(aligned_estimate[index - 1]) @ aligned_estimate[index]
            )
            ground_truth_delta = (
                np.linalg.inv(ground_truth[index - 1]) @ ground_truth[index]
            )
            delta_error = np.linalg.inv(ground_truth_delta) @ estimate_delta
            rpe_translation.append(np.linalg.norm(delta_error[:3, 3]))
            rpe_rotation_deg.append(
                np.degrees(Rotation.from_matrix(delta_error[:3, :3]).magnitude())
            )

        return {
            "associations": len(estimate),
            "ate_rmse_m": float(np.sqrt(np.mean(translation_error**2))),
            "rpe_translation_rmse_m": float(
                np.sqrt(np.mean(np.square(rpe_translation)))
            ),
            "rpe_rotation_rmse_deg": float(
                np.sqrt(np.mean(np.square(rpe_rotation_deg)))
            ),
        }

    @staticmethod
    def _interpolate_pose(
        timestamp_ns: int,
        timestamps_ns: np.ndarray,
        transforms: np.ndarray,
    ) -> np.ndarray | None:
        right = int(np.searchsorted(timestamps_ns, timestamp_ns, side="left"))
        if right < len(timestamps_ns) and timestamps_ns[right] == timestamp_ns:
            return transforms[right].copy()
        if right == 0 or right >= len(timestamps_ns):
            return None

        left = right - 1
        interval_ns = int(timestamps_ns[right] - timestamps_ns[left])
        if interval_ns <= 0 or interval_ns > 100_000_000:
            return None
        ratio = (timestamp_ns - int(timestamps_ns[left])) / interval_ns

        result = np.eye(4, dtype=np.float64)
        result[:3, 3] = (
            (1.0 - ratio) * transforms[left, :3, 3]
            + ratio * transforms[right, :3, 3]
        )
        rotations = Rotation.from_matrix(
            np.stack([transforms[left, :3, :3], transforms[right, :3, :3]])
        )
        result[:3, :3] = Slerp([0.0, 1.0], rotations)([ratio]).as_matrix()[0]
        return result

    @staticmethod
    def _load_calibration(path: Path) -> dict:
        if not path.is_file():
            raise FileNotFoundError(f"Calibration not found: {path}")
        with path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
        calibration = document.get("value0", document)

        intrinsics = calibration.get("intrinsics", [])
        extrinsics = calibration.get("T_imu_cam", [])
        resolutions = calibration.get("resolution", [])
        if len(intrinsics) != 1 or len(extrinsics) != 1 or len(resolutions) != 1:
            raise ValueError(
                "Basalt validation requires exactly one camera in calibration"
            )

        width, height = (int(value) for value in resolutions[0])
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid calibration resolution: {resolutions[0]}")

        pose = extrinsics[0]
        translation = np.array(
            [pose["px"], pose["py"], pose["pz"]], dtype=np.float64
        )
        quaternion_wxyz = np.array(
            [pose["qw"], pose["qx"], pose["qy"], pose["qz"]],
            dtype=np.float64,
        )
        t_i_c = np.eye(4, dtype=np.float64)
        t_i_c[:3, :3] = BasaltPipeline._rotation_from_wxyz(quaternion_wxyz)
        t_i_c[:3, 3] = translation
        return {"resolution": (width, height), "T_i_c": t_i_c}

    @classmethod
    def _load_ground_truth(cls, path: Path) -> tuple[np.ndarray, np.ndarray]:
        if not path.is_file():
            raise FileNotFoundError(f"Ground truth not found: {path}")
        timestamps = []
        transforms = []
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            for row_index, row in enumerate(reader):
                if not row or row[0].lstrip().startswith("#"):
                    continue
                if len(row) < 8:
                    raise ValueError(
                        f"Invalid ground-truth row {row_index}: {row}"
                    )
                timestamp_ns = int(row[0])
                position = np.asarray(row[1:4], dtype=np.float64)
                quaternion_wxyz = np.asarray(
                    [row[4], row[5], row[6], row[7]], dtype=np.float64
                )
                transform = np.eye(4, dtype=np.float64)
                transform[:3, :3] = cls._rotation_from_wxyz(quaternion_wxyz)
                transform[:3, 3] = position
                timestamps.append(timestamp_ns)
                transforms.append(transform)
        if not timestamps:
            raise ValueError("Ground truth contains no poses")
        cls._validate_timestamps(timestamps, "ground truth")
        return np.asarray(timestamps, dtype=np.int64), np.stack(transforms)

    @staticmethod
    def _world_normalization(poses: list[np.ndarray | None]) -> np.ndarray:
        first_pose = next((pose for pose in poses if pose is not None), None)
        if first_pose is None:
            raise ValueError("No valid camera pose is available for normalization")

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
    def _align_se3(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        source_mean = source.mean(axis=0)
        target_mean = target.mean(axis=0)
        covariance = (source - source_mean).T @ (target - target_mean)
        u, _, vt = np.linalg.svd(covariance)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        translation = target_mean - rotation @ source_mean
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        return transform

    @staticmethod
    def _rotation_from_wxyz(quaternion_wxyz: np.ndarray) -> np.ndarray:
        quaternion_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64)
        if not np.all(np.isfinite(quaternion_wxyz)):
            raise ValueError(f"Quaternion contains non-finite values: {quaternion_wxyz}")
        norm = np.linalg.norm(quaternion_wxyz)
        if norm < 1e-12:
            raise ValueError("Quaternion norm is zero")
        quaternion_wxyz /= norm
        return Rotation.from_quat(
            [
                quaternion_wxyz[1],
                quaternion_wxyz[2],
                quaternion_wxyz[3],
                quaternion_wxyz[0],
            ]
        ).as_matrix()

    @staticmethod
    def _to_ros_time(Time, timestamp_ns: int):
        return Time(sec=timestamp_ns // 1_000_000_000, nanosec=timestamp_ns % 1_000_000_000)

    @staticmethod
    def _read_csv(path: Path, required_columns: tuple[str, ...]) -> list[dict]:
        if not path.is_file():
            raise FileNotFoundError(f"CSV file not found: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = tuple(reader.fieldnames or ())
            missing = [column for column in required_columns if column not in fieldnames]
            if missing:
                raise ValueError(
                    f"Missing CSV column(s) in {path}: {', '.join(missing)}"
                )
            return list(reader)

    @staticmethod
    def _parse_int(value: str, column: str, row_index: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid integer in {column} at row {row_index}: {value!r}"
            ) from exc

    @staticmethod
    def _parse_float(value: str, column: str, row_index: int) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid float in {column} at row {row_index}: {value!r}"
            ) from exc
        if not np.isfinite(result):
            raise ValueError(
                f"Non-finite float in {column} at row {row_index}: {value!r}"
            )
        return result

    @staticmethod
    def _validate_timestamps(timestamps: list[int], label: str) -> None:
        if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
            raise ValueError(f"{label} timestamps must be strictly increasing")

    @staticmethod
    def _save_poses(path: Path, frames: list[dict]) -> None:
        document = {
            "schema_version": 1,
            "backend": "basalt",
            "pose_type": "c2w",
            "timestamp_unit": "ns",
            "camera_frame": "opencv_x_right_y_down_z_forward",
            "world_frame": "first_camera_origin_z_up_heading_x",
            "frames": frames,
        }
        with path.open("w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2)

    @staticmethod
    def _save_report(path: Path, report: dict) -> None:
        with path.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)

    @staticmethod
    def _load_config(config_root: str | Path) -> DictConfig:
        sensors_path = Path(config_root) / "sensors.yaml"
        if not sensors_path.is_file():
            raise FileNotFoundError(f"Missing preprocess config: {sensors_path}")
        cfg = OmegaConf.load(sensors_path)
        OmegaConf.resolve(cfg)
        return cfg


def _build_process_unit(args) -> BasaltProcessUnit:
    unit_dir = Path(args.unit).expanduser().resolve()
    if not unit_dir.is_dir():
        raise FileNotFoundError(f"Unit directory not found: {unit_dir}")
    videos = sorted(
        path
        for path in unit_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if len(videos) != 1:
        raise ValueError(
            f"Unit must contain exactly one video: {unit_dir} (found {len(videos)})"
        )
    return BasaltProcessUnit(
        unit_dir=unit_dir,
        video_path=videos[0],
        camera_csv_path=unit_dir / "camera.csv",
        imu_csv_path=unit_dir / "imu.csv",
        calibration_path=Path(args.calibration).expanduser().resolve(),
        ground_truth_path=(
            None
            if args.ground_truth is None
            else Path(args.ground_truth).expanduser().resolve()
        ),
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="Validate monocular Basalt VIO")
    parser.add_argument("--unit", required=True, help="Path to one data unit")
    parser.add_argument(
        "--calibration", required=True, help="Path to one-camera Basalt calibration"
    )
    parser.add_argument(
        "--ground_truth", default=None, help="Optional EuroC-format IMU ground truth"
    )
    return parser.parse_args()


if __name__ == "__main__":
    pipeline = BasaltPipeline()
    result = pipeline.run(_build_process_unit(_parse_args()))
    print(json.dumps(result, indent=2))
