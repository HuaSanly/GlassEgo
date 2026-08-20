import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ObjectMaskData:
    """单个 prompt 在单帧上的检测与分割结果。"""

    key: str
    prompt: str
    confidence: float
    boxes: np.ndarray
    confidences: np.ndarray
    mask_path: Path

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "prompt": self.prompt,
            "confidence": float(self.confidence),
            "boxes": self.boxes.tolist(),
            "confidences": self.confidences.tolist(),
            "mask_path": str(self.mask_path),
        }


@dataclass(frozen=True)
class ObjectFrameData:
    """单帧物体识别输出。"""

    frame_idx: int
    timestamp_ns: int
    objects: tuple[ObjectMaskData, ...]
    combined_mask_path: Path
    vis_path: Path | None

    def to_dict(self) -> dict:
        return {
            "frame_idx": int(self.frame_idx),
            "timestamp_ns": int(self.timestamp_ns),
            "objects": [item.to_dict() for item in self.objects],
            "combined_mask_path": str(self.combined_mask_path),
            "vis_path": str(self.vis_path) if self.vis_path is not None else None,
        }


@dataclass(frozen=True)
class ObjectTrackingResult:
    """单个数据单元的 DINO-SAM 物体识别结果。"""

    unit_dir: Path
    video_path: Path
    output_dir: Path
    frames: tuple[ObjectFrameData, ...]
    report: dict

    def to_dict(self) -> dict:
        def relative_path(path: Path) -> str:
            try:
                return str(Path(path).resolve().relative_to(self.unit_dir.resolve()))
            except ValueError:
                return str(path)

        return {
            "schema_version": 1,
            "unit_dir": ".",
            "video_path": relative_path(self.video_path),
            "output_dir": relative_path(self.output_dir),
            "frames": [
                {
                    **frame.to_dict(),
                    "combined_mask_path": relative_path(frame.combined_mask_path),
                    "vis_path": (
                        relative_path(frame.vis_path)
                        if frame.vis_path is not None
                        else None
                    ),
                    "objects": [
                        {
                            **item.to_dict(),
                            "mask_path": relative_path(item.mask_path),
                        }
                        for item in frame.objects
                    ],
                }
                for frame in self.frames
            ],
        }

    def save_json(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2)
