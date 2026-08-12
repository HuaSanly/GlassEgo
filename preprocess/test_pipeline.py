import json
import sys
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

PREPROCESS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PREPROCESS_ROOT.parent
DEFAULT_CONFIG_ROOT = PREPROCESS_ROOT / "config"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
POSE_FILENAMES = ("poses.json", "pose.json", "camera_poses.json")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PREPROCESS_ROOT) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_ROOT))

from utils.utils_media import build_cam_from_disk
from utils.utils_math import time_it

@dataclass(frozen=True)
class ProcessUnit:
    """Metadata for one video processed independently by the pipeline."""

    unit_dir: Path
    video_path: Path
    pose_path: Path | None = None


class PreprocessPipeline:
    """预处理协调器"""

    def __init__(
        self,
        config_root: str | Path = DEFAULT_CONFIG_ROOT,
    ):
        self.cfg = self._load_preprocess_config(config_root)
        self.pending_units = self._load_pending_units()
    def run(self):
        for unit in self.pending_units:
            self.process_hands(unit)
        
    @time_it
    def process_hands(self, unit: ProcessUnit) -> dict:
        """Process exactly one video and return lightweight result metadata."""
        if not isinstance(unit, ProcessUnit):
            raise TypeError("unit must be a ProcessUnit")
        if not unit.video_path.is_file():
            raise FileNotFoundError(f"Video not found: {unit.video_path}")
        if unit.video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video format: {unit.video_path}")
        if not self.cfg.hand_tracking.enabled:
            return {"status": "skipped", "video_path": str(unit.video_path)}

        from hand_tracking.HaMeRHandsGenerator import HaMeRHandsGenerator

        focal_length_px = OmegaConf.select(
            self.cfg,
            "sensors.camera.fallback.focal_length_px",
            default=None,
        )
        cam = build_cam_from_disk(
            str(unit.video_path),
            focal_length_px=focal_length_px,
        )
        generator = HaMeRHandsGenerator(
            unit_dir=unit.unit_dir,
            cfg=self.cfg.hand_tracking,
            output_cfg=self.cfg.output,
            cam=cam,
        )
        hands = generator.get_hands_data()

        if len(hands.hands) != len(cam.cam) or len(hands.tss) != len(cam.tss):
            raise RuntimeError(
                "Hand output is not aligned with the input camera frames: "
                f"cam={len(cam.cam)}, hands={len(hands.hands)}, "
                f"timestamps={len(hands.tss)}"
            )
        return {
            "status": "completed",
            "video_path": str(unit.video_path),
            "unit_dir": str(unit.unit_dir),
            "frames": len(hands.hands),
        }

    def _load_pending_units(self) -> list[ProcessUnit]:
        """Discover video metadata without decoding any video frames."""
        def get_configured_path(key: str) -> Path:
            value = OmegaConf.select(self.cfg, key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Required path is missing from default.yaml: {key}")
            return resolve_project_path(value)

        def resolve_project_path(value: str | Path) -> Path:
            path = Path(value).expanduser()
            return path if path.is_absolute() else PROJECT_ROOT / path
        data_root = get_configured_path("paths.data_root")
        if not data_root.is_dir():
            raise FileNotFoundError(f"Data root not found: {data_root}")

        units = []
        for unit_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
            pose_path = next(
                (
                    unit_dir / filename
                    for filename in POSE_FILENAMES
                    if (unit_dir / filename).is_file()
                ),
                None,
            )
            videos = sorted(
                path
                for path in unit_dir.iterdir()
                if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
            )
            if not videos:
                continue
            if len(videos) > 1:
                raise ValueError(
                    f"Unit must contain exactly one video: {unit_dir} "
                    f"(found {len(videos)})"
                )
            units.append(
                ProcessUnit(
                    unit_dir=unit_dir,
                    video_path=videos[0],
                    pose_path=pose_path,
                )
            )
        return units



    @staticmethod
    def _json_safe(value):
        if is_dataclass(value):
            return {
                item.name: PreprocessPipeline._json_safe(getattr(value, item.name))
                for item in fields(value)
            }
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {
                str(key): PreprocessPipeline._json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [PreprocessPipeline._json_safe(item) for item in value]
        return value

    @staticmethod
    def _load_preprocess_config(
        config_root: str | Path = DEFAULT_CONFIG_ROOT,
    ) -> DictConfig:
        """Load the global, sensor, and hand-tracking preprocess configs."""
        config_root = Path(config_root)
        default_path = config_root / "default.yaml"
        config_paths = {
            "default": default_path,
            "sensors": config_root / "sensors.yaml",
            "hand_tracking": config_root / "hand_tracking.yaml",
        }
        missing = [str(path) for path in config_paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing preprocess config file(s): {', '.join(missing)}"
            )

        def load_yaml(path: Path) -> DictConfig:
            loaded = OmegaConf.load(path)
            return loaded if loaded is not None else OmegaConf.create()

        cfg = OmegaConf.merge(
            load_yaml(config_paths["default"]),
            {"sensors": load_yaml(config_paths["sensors"])},
            {"hand_tracking": load_yaml(config_paths["hand_tracking"])},
        )
        OmegaConf.resolve(cfg)
        return cfg

if __name__ == "__main__":
    processor = PreprocessPipeline()
    processor.run()
