import csv
import json
import math
import os
import tempfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
CAMERA_COLUMNS = (
    "frame_idx",
    "frame_id",
    "rokid_timestamp_ns",
    "device_monotonic_ns",
)
BOARD_SQUARES = (8, 6)
SQUARE_LENGTH_M = 0.030
MARKER_LENGTH_M = 0.0225
SAMPLE_INTERVAL_NS = 250_000_000
MIN_CORNERS_PER_VIEW = 12
MIN_VALID_VIEWS = 10
A4_LANDSCAPE_MM = (297.0, 210.0)
BOARD_DPI = 300


def create_charuco_board():
    """创建本工具统一使用的 ChArUco 标定板。"""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return cv2.aruco.CharucoBoard(
        BOARD_SQUARES,
        SQUARE_LENGTH_M,
        MARKER_LENGTH_M,
        dictionary,
    )


def generate_board(output_path: str | Path) -> dict:
    """生成 A4 横向、300 DPI 的 ChArUco 标定板。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    page_size = tuple(
        round(length_mm / 25.4 * BOARD_DPI) for length_mm in A4_LANDSCAPE_MM
    )
    board_size_mm = (
        BOARD_SQUARES[0] * SQUARE_LENGTH_M * 1000.0,
        BOARD_SQUARES[1] * SQUARE_LENGTH_M * 1000.0,
    )
    board_size = tuple(
        round(length_mm / 25.4 * BOARD_DPI) for length_mm in board_size_mm
    )
    board_image = create_charuco_board().generateImage(
        board_size,
        marginSize=0,
        borderBits=1,
    )
    page = np.full((page_size[1], page_size[0]), 255, dtype=np.uint8)
    offset_x = (page_size[0] - board_size[0]) // 2
    offset_y = (page_size[1] - board_size[1]) // 2
    page[
        offset_y : offset_y + board_size[1],
        offset_x : offset_x + board_size[0],
    ] = board_image
    Image.fromarray(page).save(output_path, dpi=(BOARD_DPI, BOARD_DPI))

    return {
        "path": str(output_path),
        "page_size_px": list(page_size),
        "board_size_px": list(board_size),
        "dpi": BOARD_DPI,
        "squares": list(BOARD_SQUARES),
        "square_length_mm": SQUARE_LENGTH_M * 1000.0,
        "marker_length_mm": MARKER_LENGTH_M * 1000.0,
        "dictionary": "DICT_4X4_50",
    }


def calibrate_camera(
    unit_dir: str | Path,
    report_path: str | Path,
    reprojection_path: str | Path,
) -> tuple[dict, dict]:
    """从一个采集单元估计相机内参与 radtan4 畸变。"""
    unit_dir = Path(unit_dir)
    video_path = _find_unit_video(unit_dir)
    camera_rows = _load_camera_rows(unit_dir / "camera.csv")
    selected_indices = _select_frame_indices(camera_rows)

    board = create_charuco_board()
    detector = cv2.aruco.CharucoDetector(board)
    object_points = []
    image_points = []
    used_frame_indices = []
    corner_counts = []
    all_detected_points = []
    decoded_frames = 0
    detected_views = 0
    image_size = None

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    selected = set(selected_indices)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_idx = decoded_frames
            decoded_frames += 1
            current_size = (frame.shape[1], frame.shape[0])
            if image_size is None:
                image_size = current_size
            elif current_size != image_size:
                raise ValueError(
                    "Video frame resolution changed during capture: "
                    f"expected={image_size}, actual={current_size}, frame={frame_idx}"
                )
            if frame_idx not in selected:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
            if charuco_ids is None or len(charuco_ids) == 0:
                continue
            detected_views += 1
            points = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 2)
            if len(points) < MIN_CORNERS_PER_VIEW:
                continue

            matched_object, matched_image = board.matchImagePoints(
                charuco_corners,
                charuco_ids,
            )
            if matched_object is None or matched_image is None:
                continue
            matched_object = np.asarray(matched_object, dtype=np.float32).reshape(
                -1, 3
            )
            matched_image = np.asarray(matched_image, dtype=np.float32).reshape(-1, 2)
            if not _points_are_non_collinear(matched_object[:, :2]):
                continue
            object_points.append(
                matched_object.reshape(-1, 1, 3)
            )
            image_points.append(matched_image.reshape(-1, 1, 2))
            used_frame_indices.append(frame_idx)
            corner_counts.append(len(matched_image))
            all_detected_points.append(matched_image)
    finally:
        capture.release()

    if decoded_frames != len(camera_rows):
        raise ValueError(
            "Video frame count does not match camera.csv: "
            f"video={decoded_frames}, camera.csv={len(camera_rows)}"
        )
    if image_size is None:
        raise ValueError(f"Video contains no decodable frames: {video_path}")
    if len(object_points) < MIN_VALID_VIEWS:
        raise ValueError(
            "Insufficient valid ChArUco views: "
            f"required={MIN_VALID_VIEWS}, valid={len(object_points)}, "
            f"sampled={len(selected_indices)}"
        )

    flags = (
        cv2.CALIB_FIX_K3
        | cv2.CALIB_FIX_K4
        | cv2.CALIB_FIX_K5
        | cv2.CALIB_FIX_K6
    )
    (
        rms,
        camera_matrix,
        distortion,
        _,
        _,
        intrinsic_std,
        _,
        per_view_errors,
    ) = cv2.calibrateCameraExtended(
        object_points,
        image_points,
        image_size,
        None,
        None,
        flags=flags,
    )
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)
    if camera_matrix.shape != (3, 3) or len(distortion) < 4:
        raise RuntimeError("OpenCV returned an invalid camera calibration result")
    values = np.concatenate((camera_matrix.reshape(-1), distortion[:4]))
    if not math.isfinite(float(rms)) or not np.isfinite(values).all():
        raise RuntimeError("OpenCV returned non-finite camera parameters")

    camera_config = {
        "model": "pinhole",
        "resolution": [int(image_size[0]), int(image_size[1])],
        "intrinsics": [
            float(camera_matrix[0, 0]),
            float(camera_matrix[1, 1]),
            float(camera_matrix[0, 2]),
            float(camera_matrix[1, 2]),
        ],
        "distortion_model": "radtan",
        "distortion_coeffs": [float(value) for value in distortion[:4]],
    }
    report = {
        "video_path": str(video_path),
        "board": _board_report(),
        "opencv_version": cv2.__version__,
        "resolution": camera_config["resolution"],
        "decoded_frames": decoded_frames,
        "sampled_frames": len(selected_indices),
        "detected_views": detected_views,
        "used_views": len(object_points),
        "used_frame_indices": used_frame_indices,
        "corner_counts": corner_counts,
        "rms_reprojection_error_px": float(rms),
        "per_view_errors_px": [
            float(value) for value in np.asarray(per_view_errors).reshape(-1)
        ],
        "intrinsics": camera_config["intrinsics"],
        "distortion_coeffs": camera_config["distortion_coeffs"],
        "intrinsic_std_deviations": [
            float(value) for value in np.asarray(intrinsic_std).reshape(-1)
        ],
    }
    _save_reprojection_plot(
        reprojection_path,
        image_size,
        used_frame_indices,
        report["per_view_errors_px"],
        all_detected_points,
    )
    _write_json(report_path, report)
    return camera_config, report


def _find_unit_video(unit_dir: Path) -> Path:
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
    return videos[0]


def _load_camera_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"camera.csv not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        missing = [column for column in CAMERA_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(f"camera.csv is missing columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("camera.csv contains no frames")

    parsed = []
    for row_index, row in enumerate(rows):
        try:
            frame_idx = int(row["frame_idx"])
            frame_id = int(row["frame_id"])
            rokid_timestamp_ns = int(row["rokid_timestamp_ns"])
            device_monotonic_ns = int(row["device_monotonic_ns"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid camera.csv value at row {row_index}") from exc
        if frame_idx != row_index:
            raise ValueError(
                "camera.csv frame_idx must be contiguous and start at zero: "
                f"row={row_index}, frame_idx={frame_idx}"
            )
        parsed.append(
            {
                "frame_idx": frame_idx,
                "frame_id": frame_id,
                "rokid_timestamp_ns": rokid_timestamp_ns,
                "device_monotonic_ns": device_monotonic_ns,
            }
        )
    timestamps = [row["device_monotonic_ns"] for row in parsed]
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
        raise ValueError("camera.csv device_monotonic_ns must be strictly increasing")
    return parsed


def _select_frame_indices(rows: list[dict]) -> list[int]:
    selected = []
    last_timestamp = None
    for row in rows:
        timestamp = row["device_monotonic_ns"]
        if last_timestamp is None or timestamp - last_timestamp >= SAMPLE_INTERVAL_NS:
            selected.append(row["frame_idx"])
            last_timestamp = timestamp
    return selected


def _points_are_non_collinear(points: np.ndarray) -> bool:
    centered = points - np.mean(points, axis=0, keepdims=True)
    return np.linalg.matrix_rank(centered, tol=1e-3) == 2


def _board_report() -> dict:
    return {
        "squares": list(BOARD_SQUARES),
        "square_length_mm": SQUARE_LENGTH_M * 1000.0,
        "marker_length_mm": MARKER_LENGTH_M * 1000.0,
        "dictionary": "DICT_4X4_50",
    }


def _save_reprojection_plot(
    path: str | Path,
    image_size: tuple[int, int],
    frame_indices: list[int],
    errors: list[float],
    detected_points: list[np.ndarray],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(frame_indices, errors, marker="o", linewidth=1)
    axes[0].set_xlabel("Frame index")
    axes[0].set_ylabel("Reprojection RMSE (px)")
    axes[0].set_title("Per-view reprojection error")
    axes[0].grid(True, alpha=0.3)

    for points in detected_points:
        axes[1].scatter(points[:, 0], points[:, 1], s=5, alpha=0.35)
    axes[1].set_xlim(0, image_size[0])
    axes[1].set_ylim(image_size[1], 0)
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("x (px)")
    axes[1].set_ylabel("y (px)")
    axes[1].set_title("Detected corner coverage")
    axes[1].grid(True, alpha=0.3)
    figure.tight_layout()
    _save_figure_atomic(figure, path)
    plt.close(figure)


def _save_figure_atomic(figure, path: Path) -> None:
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}_",
        suffix=path.suffix,
        dir=path.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        figure.savefig(temporary_path, dpi=160)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}_",
        suffix=path.suffix,
        dir=path.parent,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
            file.write("\n")
        Path(temporary_name).replace(path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
