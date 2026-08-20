import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class HandCandidateDiagnostic:
    """单个检测候选在检测和 HaMeR 阶段的诊断数据。"""

    candidate_idx: int
    side: str
    bbox: list[float]
    bbox_width_px: float
    bbox_height_px: float
    bbox_area_px2: float
    bbox_area_ratio: float
    detector_confidence: float
    hamer_confidence: Optional[float] = None
    combined_confidence: Optional[float] = None
    final_confidence: Optional[float] = None
    geometry_confidence: Optional[float] = None
    iou_confidence: Optional[float] = None
    position_confidence: Optional[float] = None
    rotation_confidence: Optional[float] = None
    reprojection_error_px: Optional[float] = None
    positive_depth_ratio: Optional[float] = None
    position_speed_mps: Optional[float] = None
    angular_speed_rad_s: Optional[float] = None
    history_available: bool = False
    vitpose_valid_keypoints_count: Optional[int] = None
    whole_image_fallback: bool = False
    hamer_succeeded: bool = False
    reconstruction_valid: bool = False
    depth_recovery_attempted: bool = False
    depth_recovered: bool = False
    selected: bool = False
    rejection_reason: Optional[str] = None


@dataclass
class HandFrameDiagnostic:
    """一帧内的原始候选及各处理阶段手部存在状态。"""

    frame_idx: int
    timestamp_ns: int
    detector_backend: str
    whole_image_fallback: bool = False
    detector_candidate_count: int = 0
    candidates: list[HandCandidateDiagnostic] = field(default_factory=list)
    stages: dict[str, dict[str, bool]] = field(default_factory=dict)


@dataclass
class HandDiagnosticsResult:
    """完整视频的手部诊断序列。"""

    schema_version: int = 2
    frames: list[HandFrameDiagnostic] = field(default_factory=list)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f"{path.name}.tmp")
        try:
            with temporary_path.open("w", encoding="utf-8") as stream:
                json.dump(asdict(self), stream, indent=2)
            temporary_path.replace(path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
