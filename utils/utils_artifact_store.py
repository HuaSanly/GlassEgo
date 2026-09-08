"""Shared paths for preprocessing and training artifacts."""

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


class TrainingArtifactStore:
    """Own the output layout for one training run."""

    def __init__(
        self,
        runs_root: str | Path,
        task: str,
        job: str,
        experiment: str | None = None,
    ):
        parts = [self._validate_segment("task", task)]
        if experiment:
            parts.append(self._validate_segment("experiment", experiment))
        parts.append(self._validate_segment("job", job))

        self.run_dir = Path(runs_root).expanduser().resolve().joinpath(*parts)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_segment(name: str, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Training {name} must be a non-empty path segment")
        value = value.strip()
        if Path(value).name != value or value in {".", ".."}:
            raise ValueError(f"Training {name} must not contain path separators: {value}")
        return value

    @property
    def config_path(self) -> Path:
        return self.run_dir / "config.json"

    @property
    def dataset_stats_path(self) -> Path:
        return self.run_dir / "dataset_stats.json"

    @property
    def history_path(self) -> Path:
        return self.run_dir / "train_history.json"

    @property
    def latest_checkpoint_path(self) -> Path:
        return self.run_dir / "latest.pt"

    @property
    def train_curve_path(self) -> Path:
        return self.run_dir / "train_curve.png"

    @property
    def eval_curve_path(self) -> Path:
        return self.run_dir / "eval_curve.png"

    def eval_snapshot_path(self, epoch: int) -> Path:
        directory = self.run_dir / "eval_snapshots"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"eval_ep_{int(epoch):04d}.json"

    def eval_render_dir(self, epoch: int, unit_name: str) -> Path:
        unit_name = self._validate_segment("unit", unit_name)
        directory = (
            self.run_dir
            / "eval_render"
            / f"epoch_{int(epoch):04d}"
            / unit_name
            / "teacher_forced_vis"
        )
        directory.mkdir(parents=True, exist_ok=True)
        return directory
