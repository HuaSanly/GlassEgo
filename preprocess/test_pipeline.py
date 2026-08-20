import json
import gc
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
from data_types.HandsTypes import Hands
from preprocess.data_types.ObjectTypes import ObjectTrackingResult
from preprocess.data_types.PhaseTypes import PhaseSequence
from preprocess.data_types.VIOTypes import VIOResult

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
            vio_result = None
            hands = None
            phase_result = None
            try:
                vio_result = self.process_vio(unit)
                hands = self.process_hands(unit, vio_result)
                phase_result = self.process_phases(unit, vio_result, hands)
                hands = None
                self.process_objects(unit, vio_result, phase_result)
            finally:
                hands = None
                phase_result = None
                vio_result = None
                self._release_unit_resources()

    @time_it
    def process_vio(self, unit: ProcessUnit, force: bool = False) -> VIOResult:
        """Process exactly one VIO unit and return aligned camera poses."""
        if not isinstance(unit, ProcessUnit):
            raise TypeError("unit must be a ProcessUnit")
        if not self.cfg.vio.enabled:
            raise RuntimeError("VIO is required before hand preprocessing")

        from vio.BasaltVIOGenerator import BasaltVIOGenerator

        generator = BasaltVIOGenerator(
            unit_dir=unit.unit_dir,
            cfg=self.cfg.vio,
        )
        return generator.get_camera_poses(force=force)
        
    @time_it
    def process_hands(self, unit: ProcessUnit, vio_result: VIOResult) -> Hands | None:
        """Process exactly one video and return its aligned hand sequence."""
        if not isinstance(unit, ProcessUnit):
            raise TypeError("unit must be a ProcessUnit")
        if not isinstance(vio_result, VIOResult):
            raise TypeError("vio_result must be a VIOResult")
        if not unit.video_path.is_file():
            raise FileNotFoundError(f"Video not found: {unit.video_path}")
        if unit.video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video format: {unit.video_path}")
        if not self.cfg.hand_tracking.enabled:
            return None

        from hand_tracking.HaMeRHandsGenerator import HaMeRHandsGenerator

        cam = None
        generator = None
        try:
            cam = build_cam_from_disk(
                str(unit.video_path),
                vio_result=vio_result,
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
            return hands
        finally:
            if generator is not None:
                generator.cleanup()
            elif cam is not None:
                cam.cam.clear()
                cam.tss.clear()

    @staticmethod
    def _release_unit_resources() -> None:
        """释放单元级临时对象的 Python 与 CUDA 缓存。"""
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @time_it
    def process_phases(
        self,
        unit: ProcessUnit,
        vio_result: VIOResult,
        hands: Hands | None,
    ):
        """Segment one VIO-aligned video into candidate motion phases."""
        if not isinstance(unit, ProcessUnit):
            raise TypeError("unit must be a ProcessUnit")
        if not isinstance(vio_result, VIOResult):
            raise TypeError("vio_result must be a VIOResult")
        if hands is not None and not isinstance(hands, Hands):
            raise TypeError("hands must be a Hands or None")
        if not self.cfg.phase_segmentation.enabled:
            return None

        from phase_segmentation.PhaseSegmentationGenerator import (
            PhaseSegmentationGenerator,
        )

        generator = PhaseSegmentationGenerator(
            unit_dir=unit.unit_dir,
            video_path=unit.video_path,
            cfg=self.cfg.phase_segmentation,
        )
        return generator.get_phases(vio_result, hands)

    @time_it
    def process_objects(
        self,
        unit: ProcessUnit,
        vio_result: VIOResult,
        phase_result: PhaseSequence | None,
    ) -> ObjectTrackingResult | None:
        """按 HumanEgo 阶段窗口运行物体识别与三角化。"""
        if not isinstance(unit, ProcessUnit):
            raise TypeError("unit must be a ProcessUnit")
        if not isinstance(vio_result, VIOResult):
            raise TypeError("vio_result must be a VIOResult")
        if phase_result is not None and not isinstance(phase_result, PhaseSequence):
            raise TypeError("phase_result must be a PhaseSequence or None")
        if not bool(self.cfg.object_tracking.enabled):
            return None

        output_dir = unit.unit_dir / "preprocess" / "objects"
        output_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = unit.unit_dir / "object_prompts.yaml"
        if not prompt_path.is_file():
            self._write_object_skip_report(
                unit,
                "missing object_prompts.yaml",
                prompt_path,
            )
            return None
        if phase_result is None:
            self._write_object_skip_report(unit, "phase result is unavailable", None)
            return None
        if not any(frame.mode in (0, 3, 4) for frame in phase_result.frames):
            self._write_object_skip_report(
                unit,
                "phase result contains no STOP/TRANSITION/FINISHED frames",
                prompt_path,
            )
            return None

        prompts_cfg = OmegaConf.load(prompt_path)
        prompts = OmegaConf.select(prompts_cfg, "prompts", default={})
        prompts = dict(prompts or {})
        if not prompts:
            self._write_object_skip_report(unit, "object prompts are empty", prompt_path)
            return None
        object_cfg = OmegaConf.merge(self.cfg.object_tracking, {"prompts": prompts})

        from object_tracking.ObjectTrackingGenerator import ObjectTrackingGenerator

        generator = ObjectTrackingGenerator(
            unit_dir=unit.unit_dir,
            cfg=object_cfg,
            vio_result=vio_result,
            phase_result=phase_result,
        )
        return generator.get_object_data()

    @staticmethod
    def _write_object_skip_report(
        unit: ProcessUnit,
        reason: str,
        prompt_path: Path | None,
    ) -> None:
        output_dir = unit.unit_dir / "preprocess" / "objects"
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "status": "skipped",
            "unit_dir": str(unit.unit_dir),
            "reason": reason,
            "prompt_path": str(prompt_path) if prompt_path else None,
        }
        with (output_dir / "report.json").open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)

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
        """Load global and module-specific preprocess configs."""
        config_root = Path(config_root)
        default_path = config_root / "default.yaml"
        config_paths = {
            "default": default_path,
            "sensors": config_root / "sensors.yaml",
            "hand_tracking": config_root / "hand_tracking.yaml",
            "vio": config_root / "vio.yaml",
            "phase_segmentation": config_root / "phase_segmentation.yaml",
            "object_tracking": config_root / "object_tracking.yaml",
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
            {"vio": load_yaml(config_paths["vio"])},
            {"phase_segmentation": load_yaml(config_paths["phase_segmentation"])},
            {"object_tracking": load_yaml(config_paths["object_tracking"])},
        )
        OmegaConf.resolve(cfg)
        return cfg

if __name__ == "__main__":
    processor = PreprocessPipeline()
    processor.run()
