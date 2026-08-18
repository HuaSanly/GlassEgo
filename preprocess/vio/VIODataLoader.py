import csv
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from preprocess.data_types.VIOTypes import (
    CameraSample,
    RawIMUSample,
    RawSensorData,
    VIOCalibration,
    VIOUnitPaths,
)


class VIODataLoader:
    """读取并校验一个 GlassEgo VIO 数据单元。"""

    VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
    CAMERA_COLUMNS = (
        "frame_idx",
        "frame_id",
        "rokid_timestamp_ns",
        "device_monotonic_ns",
    )
    IMU_COLUMNS = ("sensor_type", "sequence", "timestamp_ns", "x", "y", "z")
    IMU_TYPES = ("gyroscope", "accelerometer")

    def __init__(self, unit_dir: str | Path):
        self.paths = self._resolve_paths(Path(unit_dir).expanduser().resolve())

    def load_sensor_data(self) -> RawSensorData:
        camera_source = self._read_csv(
            self.paths.camera_csv_path,
            self.CAMERA_COLUMNS,
        )
        imu_source = self._read_csv(self.paths.imu_csv_path, self.IMU_COLUMNS)
        if not camera_source:
            raise ValueError("camera.csv contains no frames")
        if not imu_source:
            raise ValueError("imu.csv contains no samples")

        camera = []
        for row_index, row in enumerate(camera_source):
            sample = CameraSample(
                frame_idx=self._parse_int(row["frame_idx"], "frame_idx", row_index),
                frame_id=self._parse_int(row["frame_id"], "frame_id", row_index),
                rokid_timestamp_ns=self._parse_int(
                    row["rokid_timestamp_ns"],
                    "rokid_timestamp_ns",
                    row_index,
                ),
                device_monotonic_ns=self._parse_int(
                    row["device_monotonic_ns"],
                    "device_monotonic_ns",
                    row_index,
                ),
            )
            if sample.frame_idx != row_index:
                raise ValueError(
                    "camera.csv frame_idx must be contiguous and start at zero: "
                    f"row={row_index}, frame_idx={sample.frame_idx}"
                )
            camera.append(sample)

        if len(camera) < 2:
            raise ValueError("camera.csv must contain at least two frames")
        self._validate_increasing(
            [sample.frame_id for sample in camera],
            "camera.csv frame_id",
        )
        self._validate_increasing(
            [sample.rokid_timestamp_ns for sample in camera],
            "camera.csv Rokid timestamps",
        )
        self._validate_increasing(
            [sample.device_monotonic_ns for sample in camera],
            "camera.csv device timestamps",
        )

        imu_by_type = {sensor_type: [] for sensor_type in self.IMU_TYPES}
        for row_index, row in enumerate(imu_source):
            sensor_type = row["sensor_type"].strip().lower()
            if sensor_type not in imu_by_type:
                raise ValueError(
                    f"Invalid sensor_type at row {row_index}: {sensor_type!r}"
                )
            values = np.asarray(
                [
                    self._parse_float(row[column], column, row_index)
                    for column in ("x", "y", "z")
                ],
                dtype=np.float64,
            )
            imu_by_type[sensor_type].append(
                RawIMUSample(
                    sensor_type=sensor_type,
                    sequence=self._parse_int(
                        row["sequence"], "sequence", row_index
                    ),
                    timestamp_ns=self._parse_int(
                        row["timestamp_ns"], "timestamp_ns", row_index
                    ),
                    values=values,
                )
            )

        for sensor_type, samples in imu_by_type.items():
            if len(samples) < 2:
                raise ValueError(
                    f"imu.csv contains fewer than two {sensor_type} rows"
                )
            self._validate_increasing(
                [sample.sequence for sample in samples],
                f"imu.csv {sensor_type} sequence",
            )
            self._validate_increasing(
                [sample.timestamp_ns for sample in samples],
                f"imu.csv {sensor_type} timestamps",
            )

        return RawSensorData(
            camera=tuple(camera),
            gyroscope=tuple(imu_by_type["gyroscope"]),
            accelerometer=tuple(imu_by_type["accelerometer"]),
        )

    def load_calibration(self) -> VIOCalibration:
        path = self.paths.calibration_path
        document = OmegaConf.load(path)
        camera_model = str(OmegaConf.select(document, "camera.model", default=""))
        if camera_model != "pinhole":
            raise ValueError(f"Unsupported camera model: {camera_model!r}")

        resolution = OmegaConf.select(document, "camera.resolution", default=[])
        if len(resolution) != 2:
            raise ValueError("camera.resolution must contain width and height")
        width, height = (int(value) for value in resolution)
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid calibration resolution: {resolution}")

        intrinsics = np.asarray(
            OmegaConf.select(document, "camera.intrinsics", default=[]),
            dtype=np.float64,
        )
        if intrinsics.shape != (4,) or not np.all(np.isfinite(intrinsics)):
            raise ValueError("camera.intrinsics must contain finite fx, fy, cx, cy")
        if intrinsics[0] <= 0 or intrinsics[1] <= 0:
            raise ValueError("camera focal lengths must be positive")

        distortion_model = str(
            OmegaConf.select(document, "camera.distortion_model", default="")
        )
        distortion = np.asarray(
            OmegaConf.select(document, "camera.distortion_coeffs", default=[]),
            dtype=np.float64,
        )
        if distortion_model != "radtan" or distortion.shape not in ((4,), (8,)):
            raise ValueError(
                "camera distortion must use radtan with four or eight coefficients"
            )
        if not np.all(np.isfinite(distortion)):
            raise ValueError("camera.distortion_coeffs contains non-finite values")

        T_cam_imu = np.asarray(
            OmegaConf.select(document, "T_cam_imu", default=[]),
            dtype=np.float64,
        )
        self._validate_transform(T_cam_imu, "T_cam_imu")
        T_imu_camera = np.linalg.inv(T_cam_imu)

        timeshift = float(
            OmegaConf.select(document, "timeshift_cam_imu", default=0.0)
        )
        if not np.isfinite(timeshift):
            raise ValueError("timeshift_cam_imu must be finite")

        noise_defaults = {
            "gyro_noise_density": 0.000282,
            "accel_noise_density": 0.016,
            "gyro_random_walk": 0.0001,
            "accel_random_walk": 0.001,
        }
        noise = {}
        for key, default in noise_defaults.items():
            value = OmegaConf.select(document, f"imu.{key}", default=None)
            noise[key] = default if value is None else float(value)
            if not np.isfinite(noise[key]) or noise[key] <= 0:
                raise ValueError(f"imu.{key} must be positive")

        return VIOCalibration(
            resolution=(width, height),
            intrinsics=intrinsics,
            distortion=distortion,
            T_cam_imu=T_cam_imu,
            T_imu_camera=T_imu_camera,
            timeshift_cam_imu_s=timeshift,
            noise=noise,
        )

    @classmethod
    def _resolve_paths(cls, unit_dir: Path) -> VIOUnitPaths:
        if not unit_dir.is_dir():
            raise FileNotFoundError(f"Unit directory not found: {unit_dir}")
        videos = sorted(
            path
            for path in unit_dir.iterdir()
            if path.is_file() and path.suffix.lower() in cls.VIDEO_EXTENSIONS
        )
        if len(videos) != 1:
            raise ValueError(
                f"Unit must contain exactly one video: {unit_dir} "
                f"(found {len(videos)})"
            )
        paths = VIOUnitPaths(
            unit_dir=unit_dir,
            video_path=videos[0],
            camera_csv_path=unit_dir / "camera.csv",
            imu_csv_path=unit_dir / "imu.csv",
            calibration_path=unit_dir / "calibration.yaml",
        )
        for path in (
            paths.camera_csv_path,
            paths.imu_csv_path,
            paths.calibration_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(f"VIO input not found: {path}")
        return paths

    @staticmethod
    def _read_csv(path: Path, required_columns: tuple[str, ...]) -> list[dict]:
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
    def _validate_increasing(values: list[int], label: str) -> None:
        if any(current <= previous for previous, current in zip(values, values[1:])):
            raise ValueError(f"{label} must be strictly increasing")

    @staticmethod
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
