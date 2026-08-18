import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import yaml

try:
    from .camera_calibration import calibrate_camera, generate_board
    from .extrinsic_calibration import (
        calibrate_extrinsic,
        generate_extrinsic_board,
    )
    from .imu_calibration import calibrate_imu
except ImportError:
    from camera_calibration import calibrate_camera, generate_board
    from extrinsic_calibration import calibrate_extrinsic, generate_extrinsic_board
    from imu_calibration import calibrate_imu


CALIBRATION_ROOT = Path(__file__).resolve().parent
CALIBRATION_PATH = CALIBRATION_ROOT / "calibration.yaml"
BOARD_ROOT = CALIBRATION_ROOT / "board"
REPORT_ROOT = CALIBRATION_ROOT / "report"
BOARD_PATH = BOARD_ROOT / "charuco_board.png"
CAMERA_REPORT_PATH = REPORT_ROOT / "camera_report.json"
CAMERA_REPROJECTION_PATH = REPORT_ROOT / "camera_reprojection.png"
IMU_REPORT_PATH = REPORT_ROOT / "imu_report.json"
IMU_ALLAN_PATH = REPORT_ROOT / "imu_allan.png"
EXTRINSIC_BOARD_PATH = BOARD_ROOT / "aprilgrid_board.pdf"
EXTRINSIC_TARGET_PATH = BOARD_ROOT / "aprilgrid_target.yaml"
EXTRINSIC_REPORT_PATH = REPORT_ROOT / "extrinsic_report.json"
EXTRINSIC_PDF_PATH = REPORT_ROOT / "extrinsic_kalibr_report.pdf"
EXTRINSIC_RESULTS_PATH = REPORT_ROOT / "extrinsic_results.txt"
EXTRINSIC_LOG_PATH = REPORT_ROOT / "extrinsic.log"


class CalibrationPipeline:
    """Rokid Glass3 相机与 IMU 标定入口。"""

    def board(self) -> dict:
        """生成固定规格的 ChArUco 标定板。"""
        return generate_board(BOARD_PATH)

    def camera(self, unit_dir: str | Path) -> dict:
        """标定相机，并仅更新 calibration.yaml 的 camera 段。"""
        camera_config, report = calibrate_camera(
            unit_dir,
            CAMERA_REPORT_PATH,
            CAMERA_REPROJECTION_PATH,
        )
        _validate_camera(camera_config)
        self._update_calibration_section("camera", camera_config)
        return report

    def imu(self, unit_dir: str | Path) -> dict:
        """标定 IMU 噪声，并仅更新 calibration.yaml 的 imu 段。"""
        imu_config, report = calibrate_imu(
            unit_dir,
            IMU_REPORT_PATH,
            IMU_ALLAN_PATH,
        )
        _validate_imu(imu_config)
        self._update_calibration_section("imu", imu_config)
        return report

    def extrinsic_board(self) -> dict:
        """生成固定规格的 Kalibr AprilGrid 标定板。"""
        return generate_extrinsic_board(
            EXTRINSIC_BOARD_PATH,
            EXTRINSIC_TARGET_PATH,
        )

    def extrinsic(self, unit_dir: str | Path) -> dict:
        """运行 Kalibr，并更新相机与 IMU 外参和时间偏移。"""
        calibration = self._load_calibration()
        _validate_camera(calibration.get("camera"))
        _validate_imu(calibration.get("imu"))
        matrix, timeshift, report = calibrate_extrinsic(
            unit_dir,
            calibration,
            EXTRINSIC_REPORT_PATH,
            EXTRINSIC_PDF_PATH,
            EXTRINSIC_RESULTS_PATH,
            EXTRINSIC_LOG_PATH,
        )
        _validate_extrinsic(matrix)
        _validate_timeshift(timeshift)
        calibration["T_cam_imu"] = matrix
        calibration["timeshift_cam_imu"] = timeshift
        _write_yaml_atomic(CALIBRATION_PATH, calibration)
        return report

    def validate(self) -> dict:
        """校验完整标定文件的结构与数值。"""
        calibration = self._load_calibration()
        _validate_camera(calibration.get("camera"))
        _validate_imu(calibration.get("imu"))
        _validate_extrinsic(calibration.get("T_cam_imu"))
        _validate_timeshift(calibration.get("timeshift_cam_imu"))
        return {
            "status": "valid",
            "path": str(CALIBRATION_PATH),
            "transform_convention": "p_camera = T_cam_imu @ p_imu",
        }

    def _update_calibration_section(self, key: str, value: dict) -> None:
        calibration = self._load_calibration()
        calibration[key] = value
        _write_yaml_atomic(CALIBRATION_PATH, calibration)

    @staticmethod
    def _load_calibration() -> dict:
        if not CALIBRATION_PATH.is_file():
            raise FileNotFoundError(f"Calibration file not found: {CALIBRATION_PATH}")
        with CALIBRATION_PATH.open("r", encoding="utf-8") as file:
            calibration = yaml.safe_load(file)
        if not isinstance(calibration, dict):
            raise ValueError("calibration.yaml must contain a mapping")
        for key in ("camera", "T_cam_imu", "timeshift_cam_imu", "imu"):
            if key not in calibration:
                raise ValueError(f"calibration.yaml is missing section: {key}")
        return calibration


def _validate_camera(camera) -> None:
    if not isinstance(camera, dict):
        raise ValueError("camera must be a mapping")
    if camera.get("model") != "pinhole":
        raise ValueError("camera.model must be pinhole")
    if camera.get("distortion_model") != "radtan":
        raise ValueError("camera.distortion_model must be radtan")

    resolution = _finite_vector(camera.get("resolution"), 2, "camera.resolution")
    if any(value <= 0 or not float(value).is_integer() for value in resolution):
        raise ValueError("camera.resolution must contain two positive integers")
    intrinsics = _finite_vector(camera.get("intrinsics"), 4, "camera.intrinsics")
    if intrinsics[0] <= 0.0 or intrinsics[1] <= 0.0:
        raise ValueError("camera fx and fy must be positive")
    _finite_vector(
        camera.get("distortion_coeffs"),
        4,
        "camera.distortion_coeffs",
    )


def _validate_imu(imu) -> None:
    if not isinstance(imu, dict):
        raise ValueError("imu must be a mapping")
    for key in (
        "gyro_noise_density",
        "gyro_random_walk",
        "accel_noise_density",
        "accel_random_walk",
    ):
        value = imu.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"imu.{key} must be a positive finite number")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"imu.{key} must be a positive finite number")


def _validate_extrinsic(value) -> None:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("T_cam_imu must be a numeric 4x4 matrix") from exc
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("T_cam_imu must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("T_cam_imu last row must be [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("T_cam_imu rotation must be orthogonal")
    determinant = float(np.linalg.det(rotation))
    if not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("T_cam_imu rotation determinant must be +1")


def _validate_timeshift(value) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("timeshift_cam_imu must be a finite number")
    if not math.isfinite(float(value)):
        raise ValueError("timeshift_cam_imu must be a finite number")


def _finite_vector(value, length: int, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    if any(isinstance(item, bool) for item in value):
        raise ValueError(f"{name} must contain finite numbers")
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite numbers") from exc
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite numbers")
    return result


def _write_yaml_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}_",
        suffix=path.suffix,
        dir=path.parent,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            yaml.safe_dump(
                value,
                file,
                sort_keys=False,
                allow_unicode=True,
            )
        Path(temporary_name).replace(path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rokid Glass3 calibration tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("board", help="Generate the printable ChArUco board")
    subparsers.add_parser(
        "extrinsic-board",
        help="Generate the printable Kalibr AprilGrid board",
    )

    camera_parser = subparsers.add_parser(
        "camera",
        help="Calibrate camera intrinsics from one data unit",
    )
    camera_parser.add_argument("--unit", required=True, type=Path)

    imu_parser = subparsers.add_parser(
        "imu",
        help="Estimate IMU noise from one stationary data unit",
    )
    imu_parser.add_argument("--unit", required=True, type=Path)
    extrinsic_parser = subparsers.add_parser(
        "extrinsic",
        help="Calibrate camera-IMU extrinsics with Kalibr",
    )
    extrinsic_parser.add_argument("--unit", required=True, type=Path)
    subparsers.add_parser("validate", help="Validate calibration.yaml")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    pipeline = CalibrationPipeline()
    if args.command == "board":
        result = pipeline.board()
    elif args.command == "extrinsic-board":
        result = pipeline.extrinsic_board()
    elif args.command == "camera":
        result = pipeline.camera(args.unit)
    elif args.command == "imu":
        result = pipeline.imu(args.unit)
    elif args.command == "extrinsic":
        result = pipeline.extrinsic(args.unit)
    else:
        result = pipeline.validate()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
