import csv
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from preprocess.data_types.VIOTypes import (
    BasaltRunResult,
    RawTrajectory,
    SynchronizedSensorData,
    VIOCalibration,
)


class BasaltAdapter:
    """将 GlassEgo 单目数据适配到 Basalt 命令行程序。"""

    def __init__(self, cfg):
        self.cfg = cfg

    def run(
        self,
        video_path: Path,
        sensor_data: SynchronizedSensorData,
        calibration: VIOCalibration,
        work_dir: Path,
    ) -> BasaltRunResult:
        bag_path = work_dir / "input.bag"
        calibration_path = work_dir / "basalt_calibration.json"
        log_path = work_dir / "basalt.log"
        self._write_calibration(
            calibration,
            sensor_data.imu_update_rate_hz,
            calibration_path,
        )
        self._write_temporary_bag(
            video_path,
            sensor_data,
            calibration,
            bag_path,
        )
        warnings = self._run_basalt(
            bag_path,
            calibration_path,
            work_dir,
            log_path,
        )
        trajectory_path = work_dir / "trajectory.csv"
        trajectory = self._parse_trajectory(trajectory_path)
        return BasaltRunResult(
            trajectory=trajectory,
            trajectory_path=trajectory_path,
            log_path=log_path,
            warnings=tuple(warnings),
        )

    def executable_identity(self) -> dict:
        executable = str(self.cfg.executable)
        resolved = shutil.which(executable)
        if resolved is None:
            raise FileNotFoundError(f"Basalt executable not found: {executable}")
        path = Path(resolved).resolve()
        stat = path.stat()
        return {
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    @staticmethod
    def _write_calibration(
        calibration: VIOCalibration,
        imu_update_rate_hz: float,
        path: Path,
    ) -> None:
        fx, fy, cx, cy = (
            float(value) for value in calibration.intrinsics
        )
        distortion = calibration.distortion.tolist()
        if np.allclose(distortion, 0.0):
            camera = {
                "camera_type": "pinhole",
                "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
            }
        else:
            distortion.extend([0.0] * (8 - len(distortion)))
            camera = {
                "camera_type": "pinhole-radtan8",
                "intrinsics": {
                    "fx": fx,
                    "fy": fy,
                    "cx": cx,
                    "cy": cy,
                    "k1": distortion[0],
                    "k2": distortion[1],
                    "p1": distortion[2],
                    "p2": distortion[3],
                    "k3": distortion[4],
                    "k4": distortion[5],
                    "k5": distortion[6],
                    "k6": distortion[7],
                    "rpmax": 0.0,
                },
            }

        T_imu_camera = calibration.T_imu_camera
        quaternion_xyzw = Rotation.from_matrix(
            T_imu_camera[:3, :3]
        ).as_quat()
        translation = T_imu_camera[:3, 3]
        noise = calibration.noise
        document = {
            "value0": {
                "T_imu_cam": [
                    {
                        "px": float(translation[0]),
                        "py": float(translation[1]),
                        "pz": float(translation[2]),
                        "qx": float(quaternion_xyzw[0]),
                        "qy": float(quaternion_xyzw[1]),
                        "qz": float(quaternion_xyzw[2]),
                        "qw": float(quaternion_xyzw[3]),
                    }
                ],
                "intrinsics": [camera],
                "resolution": [list(calibration.resolution)],
                "calib_accel_bias": [0.0] * 9,
                "calib_gyro_bias": [0.0] * 12,
                "imu_update_rate": float(imu_update_rate_hz),
                "accel_noise_std": [noise["accel_noise_density"]] * 3,
                "gyro_noise_std": [noise["gyro_noise_density"]] * 3,
                "accel_bias_std": [noise["accel_random_walk"]] * 3,
                "gyro_bias_std": [noise["gyro_random_walk"]] * 3,
                "cam_time_offset_ns": 0,
                "vignette": [
                    {
                        "value0": 0,
                        "value1": 10_000_000_000,
                        "value2": [[1.0], [1.0], [1.0], [1.0]],
                    }
                ],
            }
        }
        with path.open("w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2)

    def _write_temporary_bag(
        self,
        video_path: Path,
        sensor_data: SynchronizedSensorData,
        calibration: VIOCalibration,
        bag_path: Path,
    ) -> None:
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

        expected_width, expected_height = calibration.resolution
        frame_count = 0
        try:
            with Writer(bag_path) as writer:
                image_type = Image.__msgtype__
                imu_type = Imu.__msgtype__
                image_connection = writer.add_connection(
                    "/cam0/image_raw",
                    image_type,
                    typestore=typestore,
                )
                imu_connection = writer.add_connection(
                    "/imu0",
                    imu_type,
                    typestore=typestore,
                )

                events = [
                    (sample.timestamp_ns, 0, sample)
                    for sample in sensor_data.imu
                ]
                events.extend(
                    (sample.timestamp_ns, 1, sample)
                    for sample in sensor_data.camera
                )
                events.sort(key=lambda item: (item[0], item[1]))

                for timestamp_ns, event_type, sample in events:
                    if timestamp_ns is None:
                        raise ValueError("Camera timestamp was not synchronized")
                    stamp = self._to_ros_time(Time, timestamp_ns)
                    if event_type == 0:
                        message = Imu(
                            Header(0, stamp, "imu"),
                            Quaternion(0.0, 0.0, 0.0, 0.0),
                            np.full(9, -1.0, dtype=np.float64),
                            Vector3(*sample.gyroscope),
                            np.full(9, -1.0, dtype=np.float64),
                            Vector3(*sample.accelerometer),
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
                            f"expected={len(sensor_data.camera)}, "
                            f"decoded={frame_count}"
                        )
                    height, width = frame_bgr.shape[:2]
                    if (width, height) != (expected_width, expected_height):
                        raise ValueError(
                            "Video resolution does not match calibration: "
                            f"video={width}x{height}, "
                            f"calibration={expected_width}x{expected_height}"
                        )
                    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                    message = Image(
                        Header(sample.frame_idx, stamp, ""),
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

                has_extra_frame, _ = cap.read()
                if has_extra_frame:
                    raise ValueError(
                        "Video contains more readable frames than camera.csv: "
                        f"camera_rows={len(sensor_data.camera)}"
                    )
        finally:
            cap.release()

    def _run_basalt(
        self,
        bag_path: Path,
        calibration_path: Path,
        work_dir: Path,
        log_path: Path,
    ) -> list[str]:
        identity = self.executable_identity()
        timeout_seconds = float(self.cfg.timeout_seconds)
        if timeout_seconds <= 0:
            raise ValueError("vio.basalt.timeout_seconds must be positive")

        command = [
            identity["path"],
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
                cwd=work_dir,
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
        numerical_failures = (completed.stdout or "").count(
            "Numerical failure in backsubstitution"
        )
        if numerical_failures:
            return [
                "Basalt reported numerical failure in backsubstitution "
                f"{numerical_failures} times"
            ]
        return []

    @staticmethod
    def _parse_trajectory(trajectory_path: Path) -> RawTrajectory:
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
                values = np.asarray(
                    [float(value) for value in row[1:]],
                    dtype=np.float64,
                )
                if not np.all(np.isfinite(values)):
                    raise ValueError(
                        f"Non-finite Basalt trajectory row {row_index}: {row}"
                    )
                quaternion_wxyz = values[3:]
                norm = np.linalg.norm(quaternion_wxyz)
                if norm < 1e-12:
                    raise ValueError("Basalt trajectory contains a zero quaternion")
                quaternion_wxyz /= norm
                transform = np.eye(4, dtype=np.float64)
                transform[:3, :3] = Rotation.from_quat(
                    [
                        quaternion_wxyz[1],
                        quaternion_wxyz[2],
                        quaternion_wxyz[3],
                        quaternion_wxyz[0],
                    ]
                ).as_matrix()
                transform[:3, 3] = values[:3]
                timestamps.append(timestamp_ns)
                transforms.append(transform)

        if not timestamps:
            raise ValueError("Basalt trajectory contains no poses")
        timestamps_array = np.asarray(timestamps, dtype=np.int64)
        if np.any(np.diff(timestamps_array) <= 0):
            raise ValueError("Basalt trajectory timestamps must be strictly increasing")
        return RawTrajectory(
            timestamps_ns=timestamps_array,
            T_world_imu=np.stack(transforms),
        )

    @staticmethod
    def _to_ros_time(Time, timestamp_ns: int):
        return Time(
            sec=timestamp_ns // 1_000_000_000,
            nanosec=timestamp_ns % 1_000_000_000,
        )
