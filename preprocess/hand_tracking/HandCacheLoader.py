"""Load precomputed hand tracking data for downstream stages."""

import json
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from data_types.HandsTypes import HandData, Hands, HandsData, HandsJointAngles
except ModuleNotFoundError:
    from preprocess.data_types.HandsTypes import HandData, Hands, HandsData, HandsJointAngles
from preprocess.data_types.VIOTypes import (
    ARIA_MPS_INITIAL_HEADING,
    ARIA_MPS_WORLD_FRAME,
    ARIA_MPS_WORLD_ORIGIN,
    OPENCV_CAMERA_FRAME,
)


_HAND_ARRAY_FIELDS = (
    "d2c",
    "c2w",
    "wrist_pose",
    "palm_pose",
    "hand_keypoints_3d",
    "hand_keypoints_2d",
    "wrist_pose_raw_world",
    "wrist_pose_opt_world",
    "wrist_lin_vel_raw_world",
    "wrist_ang_vel_raw_world",
    "wrist_lin_vel_opt_world",
    "wrist_ang_vel_opt_world",
    "index_translation_raw_world",
    "index_translation_opt_world",
    "thumb_translation_raw_world",
    "thumb_translation_opt_world",
    "midpoint_translation_raw_world",
    "midpoint_orientation_raw_world",
    "midpoint_translation_opt_world",
    "midpoint_orientation_opt_world",
    "midpoint_pose_raw_world",
    "midpoint_pose_opt_world",
    "midpoint_lin_vel_raw_world",
    "midpoint_ang_vel_raw_world",
    "midpoint_lin_vel_opt_world",
    "midpoint_ang_vel_opt_world",
    "thumb_base_raw_world",
    "index_base_raw_world",
    "thumb_base_opt_world",
    "index_base_opt_world",
)


def load_cached_hands(
    unit_dir: str | Path,
    expected_timestamps: Sequence[int],
    filename: str = "hamer_hands.json",
) -> Hands:
    """Load and validate one per-frame hand cache sequence.

    The loader intentionally reconstructs only the shared ``Hands`` data
    contract. Model inference and cache discovery remain outside this module.
    """
    unit_dir = Path(unit_dir).expanduser().resolve()
    timestamps = [int(timestamp) for timestamp in expected_timestamps]
    if not timestamps:
        raise ValueError("Cannot load hand cache without expected timestamps")

    extra_frames = {
        int(frame_dir.name)
        for root in ("temp_data", "all_data")
        for frame_dir in (unit_dir / "preprocess" / root).glob("*")
        if frame_dir.name.isdigit()
        and (frame_dir / filename).is_file()
        and int(frame_dir.name) >= len(timestamps)
    }
    if extra_frames:
        raise ValueError(f"Hand cache has unexpected frames: {sorted(extra_frames)}")

    frames = []
    for frame_idx, timestamp_ns in enumerate(timestamps):
        candidates = (
            unit_dir / "preprocess" / "temp_data" / f"{frame_idx:05d}" / filename,
            unit_dir / "preprocess" / "all_data" / f"{frame_idx:05d}" / filename,
        )
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        if not path.is_file():
            raise FileNotFoundError(f"Cached hand frame is missing: {path}")
        with path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
        _validate_frame_metadata(document, frame_idx, timestamp_ns, path)
        frames.append(
            HandsData(
                idx=frame_idx,
                ts=timestamp_ns,
                hand_r=_parse_hand(document.get("hand_r"), True, path),
                hand_l=_parse_hand(document.get("hand_l"), False, path),
            )
        )

    return Hands(
        tss=timestamps,
        hands=frames,
        mps_path=str(unit_dir),
        camera_frame=OPENCV_CAMERA_FRAME,
        world_frame=ARIA_MPS_WORLD_FRAME,
        world_origin=ARIA_MPS_WORLD_ORIGIN,
        initial_heading=ARIA_MPS_INITIAL_HEADING,
    )


def _validate_frame_metadata(
    document: dict,
    expected_idx: int,
    expected_timestamp_ns: int,
    path: Path,
) -> None:
    if not isinstance(document, dict) or document.get("schema_version") != 2:
        raise ValueError(f"Unsupported hand cache schema: {path}")
    expected = {
        "camera_frame": OPENCV_CAMERA_FRAME,
        "world_frame": ARIA_MPS_WORLD_FRAME,
        "world_origin": ARIA_MPS_WORLD_ORIGIN,
        "initial_heading": ARIA_MPS_INITIAL_HEADING,
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise ValueError(f"Hand cache has invalid {key}: {path}")
    if document.get("idx") != expected_idx:
        raise ValueError(f"Hand cache frame index is not aligned: {path}")
    if document.get("ts") != expected_timestamp_ns:
        raise ValueError(f"Hand cache timestamp is not aligned: {path}")


def _parse_hand(document: dict | None, is_right: bool, path: Path) -> HandData | None:
    if document is None:
        return None
    if not isinstance(document, dict):
        raise ValueError(f"Hand cache side must be an object: {path}")

    tracking_state = document.get("tracking_state")
    if tracking_state not in ("observed", "interpolated"):
        raise ValueError(f"Hand cache has invalid tracking_state: {path}")

    confidence = document.get("confidence")
    if confidence is not None and not np.isfinite(float(confidence)):
        raise ValueError(f"Hand cache confidence is not finite: {path}")

    values = {
        field: _to_array(document.get(field))
        for field in _HAND_ARRAY_FIELDS
        if field in document
    }
    for field, serialized_name in {
        "hand_keypoints_3d": "kpts_3d",
        "hand_keypoints_2d": "kpts_2d",
    }.items():
        if field not in values and serialized_name in document:
            values[field] = _to_array(document[serialized_name])
    joint_angles = document.get("joint_angles")
    if isinstance(joint_angles, dict):
        values["joint_angles"] = HandsJointAngles(
            data={key: float(value) for key, value in joint_angles.items()}
        )
    grasp_value = document.get("grasp_score", document.get("grasp_state", 0.0))
    grasp_value = _optional_float(grasp_value) or 0.0
    return HandData(
        **values,
        is_right=is_right,
        tracking_state=tracking_state,
        confidence=None if confidence is None else float(confidence),
        grasp_state=float(np.clip(grasp_value, 0.0, 1.0)),
        grasp_tip_distance_m=_optional_float(document.get("grasp_tip_distance_m")),
        grasp_palm_size_m=_optional_float(document.get("grasp_palm_size_m")),
        grasp_ratio=_optional_float(document.get("grasp_ratio")),
    )


def _to_array(value):
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64)


def _optional_float(value):
    if value is None:
        return None
    result = float(value)
    if not np.isfinite(result):
        return None
    return result
