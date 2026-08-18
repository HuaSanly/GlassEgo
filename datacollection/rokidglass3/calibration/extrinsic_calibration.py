import csv
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
CAMERA_COLUMNS = (
    "frame_idx",
    "frame_id",
    "rokid_timestamp_ns",
    "device_monotonic_ns",
)
IMU_COLUMNS = ("sensor_type", "sequence", "timestamp_ns", "x", "y", "z")
IMU_TYPES = ("gyroscope", "accelerometer")
APRILGRID_COLUMNS = 6
APRILGRID_ROWS = 5
APRILGRID_TAG_SIZE_M = 0.030
APRILGRID_SPACING_RATIO = 0.3
MIN_DURATION_SECONDS = 60.0


def generate_extrinsic_board(
    board_path: str | Path,
    target_path: str | Path,
    executable: str | Path = "kalibr_create_target_pdf",
) -> dict:
    """生成 Kalibr 使用的 AprilGrid 标定板和 target YAML。"""
    board_path = Path(board_path)
    target_path = Path(target_path)
    board_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_executable = _resolve_executable(executable)
    with tempfile.TemporaryDirectory(
        prefix="glassego_aprilgrid_",
        dir=board_path.parent,
    ) as temporary:
        output_base = Path(temporary) / "aprilgrid_board"
        command = [
            resolved_executable,
            str(output_base),
            "--type",
            "apriltag",
            "--nx",
            str(APRILGRID_COLUMNS),
            "--ny",
            str(APRILGRID_ROWS),
            "--tsize",
            str(APRILGRID_TAG_SIZE_M),
            "--tspace",
            str(APRILGRID_SPACING_RATIO),
            "--tfam",
            "t36h11",
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(
                f"Kalibr board generator failed with exit code "
                f"{completed.returncode}: {detail}"
            )
        generated_path = output_base.with_suffix(".pdf")
        if not generated_path.is_file() or generated_path.stat().st_size == 0:
            raise RuntimeError("Kalibr board generator did not create a PDF")
        os.replace(generated_path, board_path)

    target = {
        "target_type": "aprilgrid",
        "tagCols": APRILGRID_COLUMNS,
        "tagRows": APRILGRID_ROWS,
        "tagSize": APRILGRID_TAG_SIZE_M,
        "tagSpacing": APRILGRID_SPACING_RATIO,
    }
    _write_yaml_atomic(target_path, target)
    return {
        "board_path": str(board_path),
        "target_path": str(target_path),
        "generator": resolved_executable,
        "tag_columns": APRILGRID_COLUMNS,
        "tag_rows": APRILGRID_ROWS,
        "tag_ids": list(range(APRILGRID_COLUMNS * APRILGRID_ROWS)),
        "tag_size_mm": APRILGRID_TAG_SIZE_M * 1000.0,
        "tag_spacing_ratio": APRILGRID_SPACING_RATIO,
        "dictionary": "tag36h11",
    }


def calibrate_extrinsic(
    unit_dir: str | Path,
    calibration: dict,
    report_path: str | Path,
    pdf_path: str | Path,
    results_path: str | Path,
    log_path: str | Path,
    executable: str | Path = "kalibr_calibrate_imu_camera",
) -> tuple[list[list[float]], float, dict]:
    """运行 Kalibr，并返回 IMU 到相机的外参与时间偏移。"""
    unit_dir = Path(unit_dir)
    report_path = Path(report_path)
    pdf_path = Path(pdf_path)
    results_path = Path(results_path)
    log_path = Path(log_path)
    try:
        video_path = _find_unit_video(unit_dir)
        camera_rows = _load_camera_rows(unit_dir / "camera.csv")
        imu_streams = _load_imu_streams(unit_dir / "imu.csv")
        clock_sync = _synchronize_camera_clock(camera_rows)
        imu_rows, imu_update_rate_hz = _merge_imu_streams(
            imu_streams["gyroscope"],
            imu_streams["accelerometer"],
        )
        duration_seconds = (
            camera_rows[-1]["timestamp_ns"] - camera_rows[0]["timestamp_ns"]
        ) * 1e-9
        if duration_seconds < MIN_DURATION_SECONDS:
            raise ValueError(
                "Camera and IMU calibration recording must be at least "
                f"{MIN_DURATION_SECONDS:.0f} seconds: {duration_seconds:.3f} s"
            )
        if imu_rows[0]["timestamp_ns"] > camera_rows[0]["timestamp_ns"]:
            raise ValueError("IMU data starts after the first camera frame")
        if imu_rows[-1]["timestamp_ns"] < camera_rows[-1]["timestamp_ns"]:
            raise ValueError("IMU data ends before the last camera frame")

        resolved_executable = _resolve_executable(executable)
        with tempfile.TemporaryDirectory(prefix="glassego_kalibr_") as temporary:
            temporary_dir = Path(temporary)
            bag_path = temporary_dir / "input.bag"
            camchain_path = temporary_dir / "camchain.yaml"
            imu_path = temporary_dir / "imu.yaml"
            target_path = temporary_dir / "aprilgrid.yaml"
            _write_camchain(camchain_path, calibration["camera"])
            _write_imu_config(imu_path, calibration["imu"], imu_update_rate_hz)
            _write_target_config(target_path)
            bag_report = _write_temporary_bag(
                video_path,
                camera_rows,
                imu_rows,
                calibration["camera"]["resolution"],
                bag_path,
            )
            command = [
                resolved_executable,
                "--bag",
                str(bag_path),
                "--cams",
                str(camchain_path),
                "--imu",
                str(imu_path),
                "--target",
                str(target_path),
                "--bag-freq",
                "20",
                "--dont-show-report",
            ]
            _run_kalibr(command, temporary_dir, log_path)
            output_yaml = _find_output(temporary_dir, "*-camchain-imucam.yaml")
            output_pdf = _find_output(temporary_dir, "*-report-imucam.pdf")
            output_results = _find_output(temporary_dir, "*-results-imucam.txt")
            t_cam_imu, timeshift_cam_imu = _parse_kalibr_output(output_yaml)
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_pdf, pdf_path)
            shutil.copy2(output_results, results_path)

        report = {
            "status": "completed",
            "unit_dir": str(unit_dir),
            "video_path": str(video_path),
            "duration_seconds": duration_seconds,
            "camera_frames": len(camera_rows),
            "raw_gyroscope_samples": len(imu_streams["gyroscope"]),
            "raw_accelerometer_samples": len(imu_streams["accelerometer"]),
            "merged_imu_samples": len(imu_rows),
            "imu_update_rate_hz": imu_update_rate_hz,
            "clock_sync": clock_sync,
            "bag": bag_report,
            "T_cam_imu": t_cam_imu,
            "timeshift_cam_imu": timeshift_cam_imu,
            "timeshift_definition": "t_imu = t_cam + timeshift_cam_imu",
            "outputs": {
                "kalibr_pdf": str(pdf_path),
                "kalibr_results": str(results_path),
                "kalibr_log": str(log_path),
            },
        }
        _write_json_atomic(report_path, report)
        return t_cam_imu, timeshift_cam_imu, report
    except Exception as exc:
        _write_json_atomic(
            report_path,
            {
                "status": "failed",
                "unit_dir": str(unit_dir),
                "error": f"{type(exc).__name__}: {exc}",
                "kalibr_log": str(log_path),
            },
        )
        raise


def _find_unit_video(unit_dir: Path) -> Path:
    if not unit_dir.is_dir():
        raise NotADirectoryError(f"Calibration unit not found: {unit_dir}")
    videos = sorted(
        path
        for path in unit_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if len(videos) != 1:
        raise ValueError(
            f"Calibration unit must contain exactly one video: {unit_dir}, "
            f"found={len(videos)}"
        )
    return videos[0]


def _load_camera_rows(path: Path) -> list[dict]:
    rows = _read_csv(path, CAMERA_COLUMNS)
    if len(rows) < 2:
        raise ValueError("camera.csv must contain at least two frames")
    parsed = []
    for row_index, row in enumerate(rows):
        try:
            values = {key: int(row[key]) for key in CAMERA_COLUMNS}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid camera.csv value at row {row_index}") from exc
        if values["frame_idx"] != row_index:
            raise ValueError(
                "camera.csv frame_idx must be contiguous and start at zero: "
                f"row={row_index}, frame_idx={values['frame_idx']}"
            )
        parsed.append(values)
    _validate_increasing(
        [row["frame_id"] for row in parsed],
        "camera.csv frame_id",
    )
    return parsed


def _load_imu_streams(path: Path) -> dict[str, list[dict]]:
    rows = _read_csv(path, IMU_COLUMNS)
    streams = {sensor_type: [] for sensor_type in IMU_TYPES}
    for row_index, row in enumerate(rows):
        sensor_type = row["sensor_type"].strip().lower()
        if sensor_type not in streams:
            raise ValueError(
                f"Unknown sensor_type at row {row_index}: {sensor_type!r}"
            )
        try:
            sequence = int(row["sequence"])
            timestamp_ns = int(row["timestamp_ns"])
            values = np.asarray(
                [float(row[axis]) for axis in ("x", "y", "z")],
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid imu.csv value at row {row_index}") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite imu.csv value at row {row_index}")
        streams[sensor_type].append(
            {
                "sequence": sequence,
                "timestamp_ns": timestamp_ns,
                "values": values,
            }
        )
    for sensor_type, samples in streams.items():
        if len(samples) < 2:
            raise ValueError(f"imu.csv contains fewer than two {sensor_type} samples")
        _validate_increasing(
            [sample["timestamp_ns"] for sample in samples],
            f"imu.csv {sensor_type} timestamp",
        )
        _validate_increasing(
            [sample["sequence"] for sample in samples],
            f"imu.csv {sensor_type} sequence",
        )
    return streams


def _read_csv(path: Path, required_columns: tuple[str, ...]) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        missing = [key for key in required_columns if key not in fieldnames]
        if missing:
            raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name} contains no rows")
    return rows


def _synchronize_camera_clock(camera_rows: list[dict]) -> dict:
    rokid_timestamps = np.asarray(
        [row["rokid_timestamp_ns"] for row in camera_rows],
        dtype=np.int64,
    )
    device_timestamps = np.asarray(
        [row["device_monotonic_ns"] for row in camera_rows],
        dtype=np.int64,
    )
    _validate_increasing(rokid_timestamps.tolist(), "camera Rokid timestamp")
    _validate_increasing(device_timestamps.tolist(), "camera device timestamp")
    source_delta = (rokid_timestamps - rokid_timestamps[0]).astype(np.float64)
    target_delta = (device_timestamps - device_timestamps[0]).astype(np.float64)
    centered_source = source_delta - source_delta.mean()
    denominator = float(centered_source @ centered_source)
    if denominator <= 0.0:
        raise ValueError("Unable to estimate camera clock mapping")
    scale = float(centered_source @ (target_delta - target_delta.mean())) / denominator
    relative_offset_ns = float(np.mean(target_delta - scale * source_delta))
    mapped = np.rint(
        device_timestamps[0] + relative_offset_ns + scale * source_delta
    ).astype(np.int64)
    _validate_increasing(mapped.tolist(), "synchronized camera timestamp")
    residuals = device_timestamps - mapped
    for row, timestamp_ns in zip(camera_rows, mapped):
        row["timestamp_ns"] = int(timestamp_ns)
    return {
        "model": "affine",
        "scale": scale,
        "drift_ppm": (scale - 1.0) * 1_000_000.0,
        "relative_offset_ns": relative_offset_ns,
        "residual_rms_ns": float(np.sqrt(np.mean(residuals.astype(float) ** 2))),
        "residual_max_abs_ns": int(np.max(np.abs(residuals))),
    }


def _merge_imu_streams(
    gyroscope: list[dict],
    accelerometer: list[dict],
) -> tuple[list[dict], float]:
    gyro_timestamps = np.asarray(
        [sample["timestamp_ns"] for sample in gyroscope], dtype=np.int64
    )
    accel_timestamps = np.asarray(
        [sample["timestamp_ns"] for sample in accelerometer], dtype=np.int64
    )
    gyro_values = np.stack([sample["values"] for sample in gyroscope])
    accel_values = np.stack([sample["values"] for sample in accelerometer])
    valid = (gyro_timestamps >= accel_timestamps[0]) & (
        gyro_timestamps <= accel_timestamps[-1]
    )
    gyro_timestamps = gyro_timestamps[valid]
    gyro_values = gyro_values[valid]
    if len(gyro_timestamps) < 2:
        raise ValueError("Gyroscope and accelerometer time ranges do not overlap")
    interpolated_accel = np.column_stack(
        [
            np.interp(gyro_timestamps, accel_timestamps, accel_values[:, axis])
            for axis in range(3)
        ]
    )
    merged = [
        {
            "timestamp_ns": int(timestamp_ns),
            "values": np.concatenate((gyro, accel)),
        }
        for timestamp_ns, gyro, accel in zip(
            gyro_timestamps,
            gyro_values,
            interpolated_accel,
        )
    ]
    median_interval_ns = float(np.median(np.diff(gyro_timestamps)))
    if median_interval_ns <= 0.0:
        raise ValueError("Unable to determine IMU update rate")
    return merged, 1_000_000_000.0 / median_interval_ns


def _write_temporary_bag(
    video_path: Path,
    camera_rows: list[dict],
    imu_rows: list[dict],
    resolution: list[int],
    bag_path: Path,
) -> dict:
    try:
        from rosbags.rosbag1 import Writer
        from rosbags.typesys import Stores, get_typestore
    except ImportError as exc:
        raise ImportError("rosbags is required to create the Kalibr input bag") from exc

    typestore = get_typestore(Stores.ROS1_NOETIC)
    Header = typestore.types["std_msgs/msg/Header"]
    Time = typestore.types["builtin_interfaces/msg/Time"]
    ImageMessage = typestore.types["sensor_msgs/msg/Image"]
    ImuMessage = typestore.types["sensor_msgs/msg/Imu"]
    Quaternion = typestore.types["geometry_msgs/msg/Quaternion"]
    Vector3 = typestore.types["geometry_msgs/msg/Vector3"]
    expected_width, expected_height = [int(value) for value in resolution]

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    decoded_frames = 0
    try:
        with Writer(bag_path) as writer:
            image_type = ImageMessage.__msgtype__
            imu_type = ImuMessage.__msgtype__
            image_connection = writer.add_connection(
                "/cam0/image_raw", image_type, typestore=typestore
            )
            imu_connection = writer.add_connection(
                "/imu0", imu_type, typestore=typestore
            )
            events = [
                (sample["timestamp_ns"], 0, sample) for sample in imu_rows
            ] + [
                (frame["timestamp_ns"], 1, frame) for frame in camera_rows
            ]
            events.sort(key=lambda event: (event[0], event[1]))
            for timestamp_ns, event_type, row in events:
                stamp = _to_ros_time(Time, timestamp_ns)
                if event_type == 0:
                    values = row["values"]
                    message = ImuMessage(
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

                ok, frame_bgr = capture.read()
                if not ok:
                    raise ValueError(
                        "Video contains fewer readable frames than camera.csv: "
                        f"expected={len(camera_rows)}, decoded={decoded_frames}"
                    )
                height, width = frame_bgr.shape[:2]
                if (width, height) != (expected_width, expected_height):
                    raise ValueError(
                        "Video resolution does not match calibration: "
                        f"video={width}x{height}, "
                        f"calibration={expected_width}x{expected_height}"
                    )
                frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                message = ImageMessage(
                    Header(row["frame_idx"], stamp, ""),
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
                decoded_frames += 1
            extra_frame, _ = capture.read()
            if extra_frame:
                raise ValueError(
                    "Video contains more readable frames than camera.csv: "
                    f"camera.csv={len(camera_rows)}"
                )
    finally:
        capture.release()
    return {
        "topics": ["/cam0/image_raw", "/imu0"],
        "image_messages": decoded_frames,
        "imu_messages": len(imu_rows),
        "start_timestamp_ns": min(
            camera_rows[0]["timestamp_ns"], imu_rows[0]["timestamp_ns"]
        ),
        "end_timestamp_ns": max(
            camera_rows[-1]["timestamp_ns"], imu_rows[-1]["timestamp_ns"]
        ),
    }


def _write_camchain(path: Path, camera: dict) -> None:
    _write_yaml_atomic(
        path,
        {
            "cam0": {
                "camera_model": "pinhole",
                "intrinsics": camera["intrinsics"],
                "distortion_model": "radtan",
                "distortion_coeffs": camera["distortion_coeffs"],
                "resolution": camera["resolution"],
                "rostopic": "/cam0/image_raw",
            }
        },
    )


def _write_imu_config(path: Path, imu: dict, update_rate_hz: float) -> None:
    _write_yaml_atomic(
        path,
        {
            "accelerometer_noise_density": float(imu["accel_noise_density"]),
            "accelerometer_random_walk": float(imu["accel_random_walk"]),
            "gyroscope_noise_density": float(imu["gyro_noise_density"]),
            "gyroscope_random_walk": float(imu["gyro_random_walk"]),
            "rostopic": "/imu0",
            "update_rate": float(update_rate_hz),
        },
    )


def _write_target_config(path: Path) -> None:
    _write_yaml_atomic(
        path,
        {
            "target_type": "aprilgrid",
            "tagCols": APRILGRID_COLUMNS,
            "tagRows": APRILGRID_ROWS,
            "tagSize": APRILGRID_TAG_SIZE_M,
            "tagSpacing": APRILGRID_SPACING_RATIO,
        },
    )


def _run_kalibr(command: list[str], working_dir: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            cwd=working_dir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Kalibr failed with exit code {completed.returncode}; see {log_path}"
        )


def _parse_kalibr_output(path: Path) -> tuple[list[list[float]], float]:
    with path.open("r", encoding="utf-8") as file:
        output = yaml.safe_load(file)
    if not isinstance(output, dict) or not isinstance(output.get("cam0"), dict):
        raise ValueError("Kalibr output is missing cam0")
    cam0 = output["cam0"]
    matrix = _validate_extrinsic(cam0.get("T_cam_imu"))
    timeshift = cam0.get("timeshift_cam_imu")
    if isinstance(timeshift, bool) or not isinstance(timeshift, (int, float)):
        raise ValueError("Kalibr timeshift_cam_imu must be a finite number")
    timeshift = float(timeshift)
    if not math.isfinite(timeshift):
        raise ValueError("Kalibr timeshift_cam_imu must be a finite number")
    return matrix.tolist(), timeshift


def _validate_extrinsic(value) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Kalibr T_cam_imu must be a numeric 4x4 matrix") from exc
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Kalibr T_cam_imu must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("Kalibr T_cam_imu has an invalid homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("Kalibr T_cam_imu rotation is not orthogonal")
    if not math.isclose(
        float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1e-5
    ):
        raise ValueError("Kalibr T_cam_imu rotation determinant is not +1")
    return matrix


def _resolve_executable(value: str | Path) -> str:
    text = str(value)
    if Path(text).is_file():
        return str(Path(text).resolve())
    resolved = shutil.which(text)
    if resolved is None:
        raise FileNotFoundError(f"Kalibr executable not found: {text}")
    return resolved


def _find_output(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1 or matches[0].stat().st_size == 0:
        raise FileNotFoundError(
            f"Expected one non-empty Kalibr output matching {pattern}, "
            f"found={len(matches)}"
        )
    return matches[0]


def _validate_increasing(values: list[int], name: str) -> None:
    if any(current <= previous for previous, current in zip(values, values[1:])):
        raise ValueError(f"{name} must be strictly increasing")


def _to_ros_time(Time, timestamp_ns: int):
    seconds, nanoseconds = divmod(int(timestamp_ns), 1_000_000_000)
    return Time(sec=seconds, nanosec=nanoseconds)


def _write_yaml_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}_",
        suffix=path.suffix,
        dir=path.parent,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            yaml.safe_dump(value, file, sort_keys=False)
        Path(temporary_name).replace(path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}_",
        suffix=path.suffix,
        dir=path.parent,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            json.dump(value, file, indent=2, ensure_ascii=False)
        Path(temporary_name).replace(path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
