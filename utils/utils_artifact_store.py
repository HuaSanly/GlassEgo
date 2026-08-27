"""Shared output paths for frame-scoped preprocessing artifacts."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class FrameArtifactStore:
    """Keep training frames and context frames in separate flat trees."""

    def __init__(self, unit_dir: str | Path):
        self.unit_dir = Path(unit_dir).expanduser().resolve()
        self.preprocess_dir = self.unit_dir / "preprocess"
        self.all_data_dir = self.preprocess_dir / "all_data"
        self.temp_data_dir = self.preprocess_dir / "temp_data"
        self.vis_dir = self.preprocess_dir / "vis"

    def frame_dir(self, frame_idx: int, is_training: bool) -> Path:
        root = self.all_data_dir if is_training else self.temp_data_dir
        path = root / f"{int(frame_idx):05d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def frame_path(self, frame_idx: int, filename: str, is_training: bool) -> Path:
        return self.frame_dir(frame_idx, is_training) / filename

    def write_image(
        self,
        frame_idx: int,
        filename: str,
        image_bgr: np.ndarray,
        is_training: bool,
    ) -> Path:
        if not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8:
            raise ValueError("Frame images must be uint8 numpy arrays")
        path = self.frame_path(frame_idx, filename, is_training)
        if not cv2.imwrite(str(path), image_bgr):
            raise IOError(f"Failed to write frame artifact: {path}")
        return path

    def relative_path(self, path: str | Path) -> str:
        path = Path(path).expanduser().resolve()
        try:
            return str(path.relative_to(self.unit_dir))
        except ValueError:
            return str(path)

    def module_vis_dir(self, module: str) -> Path:
        path = self.vis_dir / module
        path.mkdir(parents=True, exist_ok=True)
        return path

    def temp_path(self, filename: str) -> Path:
        self.temp_data_dir.mkdir(parents=True, exist_ok=True)
        return self.temp_data_dir / filename

