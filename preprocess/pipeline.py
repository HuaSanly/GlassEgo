import json
import gc
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path

import numpy as np
import cv2
from omegaconf import DictConfig, OmegaConf

PREPROCESS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PREPROCESS_ROOT.parent
DEFAULT_CONFIG_ROOT = PREPROCESS_ROOT / "config"
UNIT_REPORT_RELATIVE_PATH = Path("preprocess") / "vis" / "pipeline" / "report.json"
BATCH_REPORT_FILENAME = "preprocess_batch_report.json"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
POSE_FILENAMES = ("poses.json", "pose.json", "camera_poses.json")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PREPROCESS_ROOT) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_ROOT))

from utils.utils_media import build_cam_from_disk
from preprocess.DatasetGenerator import DatasetGenerator
from utils.utils_math import time_it
from utils.utils_artifact_store import FrameArtifactStore
from data_types.HandsTypes import Hands
from preprocess.data_types.ObjectTypes import ObjectTrackingResult
from preprocess.data_types.PhaseTypes import (
    FINISHED_TAIL_FRAMES,
    FORCED_NON_OPERATION_PREFIX_FRAMES,
    MIN_CLASSIFIED_MIDDLE_FRAMES,
    MIN_UNIT_FRAME_COUNT,
    OPERATION_MODE,
    PhaseSequence,
)
from preprocess.data_types.VIOTypes import (
    ARIA_MPS_INITIAL_HEADING,
    ARIA_MPS_WORLD_FRAME,
    ARIA_MPS_WORLD_ORIGIN,
    VIOResult,
)

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
        data_root: str | Path | None = None,
    ):
        self.config_root = Path(config_root).expanduser().resolve()
        self.cfg = self._load_preprocess_config(self.config_root)
        if data_root is not None:
            self.cfg.paths.data_root = str(Path(data_root).expanduser().resolve())
        self.pending_units = self._load_pending_units()
        self.selection_errors = []

    def select_units(self, keys: list[str]) -> list[ProcessUnit]:
        """Select units in the exact order requested by a CLI or UI caller."""
        units_by_key = {
            f"{unit.unit_dir.parent.name}/{unit.unit_dir.name}": unit
            for unit in self.pending_units
        }
        selected = []
        self.selection_errors = []
        for key in keys:
            unit = units_by_key.get(key)
            if unit is None:
                self.selection_errors.append(
                    {"key": key, "status": "failed", "reason": "unit not found"}
                )
                continue
            selected.append(unit)
        self.pending_units = selected
        return selected

    def run(self) -> dict:
        """Run every pending unit in a fresh subprocess, one at a time."""
        results = list(getattr(self, "selection_errors", []))
        for index, unit in enumerate(self.pending_units, start=1):
            print(
                f"║ [Batch] Unit {index}/{len(self.pending_units)}: {unit.unit_dir}",
                flush=True,
            )
            results.append(self._run_unit_subprocess(unit))

        report = {
            "status": "completed" if not any(
                item["status"] == "failed" for item in results
            ) else "failed",
            "unit_count": len(self.pending_units),
            "results": results,
        }
        self._atomic_write_json(
            self._batch_report_path(),
            report,
        )
        completed = sum(item["status"] == "completed" for item in results)
        skipped = sum(item["status"] == "skipped" for item in results)
        failed = sum(item["status"] == "failed" for item in results)
        print(
            f"║ [Batch] Finished: completed={completed}, skipped={skipped}, "
            f"failed={failed}",
            flush=True,
        )
        return report

    def run_unit(self, unit: ProcessUnit) -> dict:
        """Run exactly one unit inside the worker subprocess."""
        vio_result = None
        hands = None
        phase_result = None
        stage = "preflight"
        started_at = _utc_now()
        try:
            preflight = self.preflight_unit(unit)
            if preflight["status"] == "skipped":
                print(
                    "║ [Preflight] Skipping unit: "
                    f"{unit.unit_dir} ({preflight['reason']})",
                    flush=True,
                )
                return self._finish_unit_report(
                    unit,
                    status="skipped",
                    started_at=started_at,
                    reason=preflight["reason"],
                    stage=stage,
                )

            stage = "vio"
            vio_result = self.process_vio(unit)
            stage = "hands"
            hands = self.process_hands(unit, vio_result)
            stage = "phases"
            phase_result = self.process_phases(unit, vio_result, hands)
            stage = "objects"
            object_result = self.process_objects(unit, vio_result, phase_result, hands)
            if object_result is not None:
                training_frames = set(object_result.report["training_frames"])
                context_frames = set(object_result.report["object_centric_frames"])
                frame_indices = list(object_result.report["tracking_frames"])
                frame_images = self._load_frame_images(
                    unit,
                    frame_indices,
                    training_frames,
                )
                stage = "lama"
                self.process_lama(
                    unit,
                    frame_images,
                    training_frames,
                    context_frames,
                    float(object_result.report["fps"]),
                )
                stage = "visualkpts"
                self.process_visualkpts(
                    unit,
                    object_result,
                    frame_images,
                    frame_indices,
                    training_frames,
                    vio_result,
                    hands,
                )
                stage = "dataset"
                self.process_dataset(
                    unit,
                    object_result,
                    frame_images,
                    training_frames,
                    vio_result,
                )
            return self._finish_unit_report(
                unit,
                status="completed",
                started_at=started_at,
                stage=stage,
            )
        except Exception as error:
            error_traceback = traceback.format_exc()
            print(
                f"║ [Unit] Failed at {stage}: {unit.unit_dir}: {error}",
                file=sys.stderr,
                flush=True,
            )
            print(error_traceback, file=sys.stderr, end="", flush=True)
            return self._finish_unit_report(
                unit,
                status="failed",
                started_at=started_at,
                stage=stage,
                error=error,
                traceback_text=error_traceback,
            )
        finally:
            hands = None
            phase_result = None
            vio_result = None
            self._release_unit_resources()

    def _run_unit_subprocess(self, unit: ProcessUnit) -> dict:
        """Run one worker and convert crashes into a failed unit result."""
        report_path = unit.unit_dir / UNIT_REPORT_RELATIVE_PATH
        self._atomic_write_json(
            report_path,
            {
                "status": "running",
                "unit_dir": str(unit.unit_dir),
                "video_path": str(unit.video_path),
                "started_at": _utc_now(),
            },
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-unit",
            str(unit.unit_dir),
            "--config-root",
            str(self.config_root),
        ]
        configured_data_root = str(OmegaConf.select(self.cfg, "paths.data_root"))
        command.extend(["--data-root", configured_data_root])
        try:
            completed = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                check=False,
            )
            returncode = completed.returncode
        except Exception as error:
            returncode = None
            report = {
                "status": "failed",
                "unit_dir": str(unit.unit_dir),
                "error_type": type(error).__name__,
                "error": str(error),
                "worker_exit_code": None,
                "finished_at": _utc_now(),
            }
            print(
                f"║ [Batch] Could not start worker for {unit.unit_dir}: {error}",
                file=sys.stderr,
                flush=True,
            )
            self._atomic_write_json(report_path, report)
            return report

        report = self._read_json(report_path)
        if returncode != 0 or report.get("status") not in {"completed", "skipped"}:
            report.update(
                {
                    "status": "failed",
                    "worker_exit_code": returncode,
                    "worker_signal": -returncode if returncode is not None and returncode < 0 else None,
                    "finished_at": _utc_now(),
                }
            )
            if "error" not in report:
                report["error"] = "worker exited before producing a successful result"
            print(
                f"║ [Batch] Worker failed for {unit.unit_dir} "
                f"(exit={returncode})",
                file=sys.stderr,
                flush=True,
            )
            self._atomic_write_json(report_path, report)
        else:
            report["worker_exit_code"] = returncode
            self._atomic_write_json(report_path, report)
        return report

    def _finish_unit_report(
        self,
        unit: ProcessUnit,
        *,
        status: str,
        started_at: str,
        reason: str | None = None,
        stage: str | None = None,
        error: Exception | None = None,
        traceback_text: str | None = None,
    ) -> dict:
        report = {
            "status": status,
            "unit_dir": str(unit.unit_dir),
            "video_path": str(unit.video_path),
            "started_at": started_at,
            "finished_at": _utc_now(),
            "stage": stage,
            "reason": reason,
        }
        if error is not None:
            report.update(
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        if traceback_text is not None:
            report["traceback"] = traceback_text
        self._atomic_write_json(unit.unit_dir / UNIT_REPORT_RELATIVE_PATH, report)
        return report

    def _batch_report_path(self) -> Path:
        data_root = Path(str(self.cfg.paths.data_root)).expanduser()
        if not data_root.is_absolute():
            data_root = PROJECT_ROOT / data_root
        return data_root / BATCH_REPORT_FILENAME

    @staticmethod
    def _read_json(path: Path) -> dict:
        try:
            with path.open("r", encoding="utf-8") as stream:
                document = json.load(stream)
            return document if isinstance(document, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def preflight_unit(self, unit: ProcessUnit) -> dict:
        """Decode enough frames to reject undersized units before VIO starts."""
        if not isinstance(unit, ProcessUnit):
            raise TypeError("unit must be a ProcessUnit")

        capture = cv2.VideoCapture(str(unit.video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Cannot open input video: {unit.video_path}")

        checked_frames = 0
        try:
            while checked_frames < MIN_UNIT_FRAME_COUNT:
                ok, _ = capture.read()
                if not ok:
                    break
                checked_frames += 1
        finally:
            capture.release()

        accepted = checked_frames >= MIN_UNIT_FRAME_COUNT
        if accepted:
            reason_code = None
            reason = f"decoded at least {MIN_UNIT_FRAME_COUNT} frames"
        else:
            reason_code = "insufficient_frames"
            reason = (
                f"decoded {checked_frames} frames, requires {MIN_UNIT_FRAME_COUNT} "
                f"({FORCED_NON_OPERATION_PREFIX_FRAMES} non-operation + "
                f"{MIN_CLASSIFIED_MIDDLE_FRAMES} classified + "
                f"{FINISHED_TAIL_FRAMES} finished)"
            )
        report = {
            "status": "accepted" if accepted else "skipped",
            "reason_code": reason_code,
            "reason": reason,
            "threshold_frames": MIN_UNIT_FRAME_COUNT,
            "checked_frames": checked_frames,
            "phase_frame_requirements": {
                "non_operation_prefix": FORCED_NON_OPERATION_PREFIX_FRAMES,
                "classified_middle_minimum": MIN_CLASSIFIED_MIDDLE_FRAMES,
                "finished_tail": FINISHED_TAIL_FRAMES,
            },
            "unit_dir": str(unit.unit_dir),
            "video_path": str(unit.video_path),
        }
        self._atomic_write_json(
            unit.unit_dir / "preprocess" / "vis" / "preflight" / "report.json",
            report,
        )
        return report

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

        timestamps = [frame.timestamp_ns for frame in vio_result.trajectory.frames]
        cache_filename = str(self.cfg.output.json_filename)
        reuse_existing = bool(getattr(self.cfg.hand_tracking, "reuse_existing", False))
        if not self.cfg.hand_tracking.enabled:
            hand_input = OmegaConf.select(
                self.cfg,
                "phase_segmentation.hand_input",
                default={},
            ) or {}
            if not reuse_existing and not bool(hand_input.get("use_cached", False)):
                return None
            from hand_tracking.HandCacheLoader import load_cached_hands

            filename = (
                cache_filename
                if reuse_existing
                else str(hand_input.get("filename", "hamer_hands.json"))
            )
            try:
                return load_cached_hands(unit.unit_dir, timestamps, filename=filename)
            except (FileNotFoundError, ValueError) as error:
                raise ValueError(
                    f"Hand tracking is disabled and its cache is unavailable or stale: {error}"
                ) from error

        if reuse_existing:
            from hand_tracking.HandCacheLoader import load_cached_hands

            try:
                hands = load_cached_hands(
                    unit.unit_dir,
                    timestamps,
                    filename=cache_filename,
                )
                print(
                    f"[Hands] Reusing {len(hands.hands)} aligned cached frames "
                    f"from {unit.unit_dir}",
                    flush=True,
                )
                return hands
            except (FileNotFoundError, ValueError) as error:
                print(
                    f"[Hands] Cached frames are unavailable or stale ({error}); "
                    "running hand inference",
                    flush=True,
                )

        self._clear_hand_cache(unit.unit_dir, cache_filename)
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

    def _clear_hand_cache(self, unit_dir: Path, filename: str) -> None:
        """Invalidate only this hand method's artifacts before fresh inference."""
        video_filename = str(
            getattr(self.cfg.output, "video_filename", "hamer_hands_vis.mp4")
        )
        for name in (filename, video_filename):
            if Path(name).name != name or name in ("", ".", ".."):
                raise ValueError(f"Hand output filename must be a single path segment: {name}")
        preprocess_dir = unit_dir / "preprocess"
        for root in ("temp_data", "all_data"):
            for frame_dir in (preprocess_dir / root).glob("*"):
                path = frame_dir / filename
                if frame_dir.name.isdigit() and path.is_file():
                    path.unlink()
        analysis_dir = preprocess_dir / "vis" / "hands"
        if analysis_dir.is_dir():
            shutil.rmtree(analysis_dir)
        video_path = preprocess_dir / "vis" / video_filename
        video_path.unlink(missing_ok=True)
        video_path.with_suffix(".gif").unlink(missing_ok=True)
        (preprocess_dir / "vis" / "hamer_hands_diagnostics.mp4").unlink(
            missing_ok=True
        )

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
        hands: Hands | None = None,
    ) -> ObjectTrackingResult | None:
        """按 HumanEgo 阶段窗口运行物体识别与三角化。"""
        if not isinstance(unit, ProcessUnit):
            raise TypeError("unit must be a ProcessUnit")
        if not isinstance(vio_result, VIOResult):
            raise TypeError("vio_result must be a VIOResult")
        if phase_result is not None and not isinstance(phase_result, PhaseSequence):
            raise TypeError("phase_result must be a PhaseSequence or None")
        if hands is not None and not isinstance(hands, Hands):
            raise TypeError("hands must be a Hands or None")
        if not bool(self.cfg.object_tracking.enabled):
            return None

        output_dir = unit.unit_dir / "preprocess" / "vis" / "objects"
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
        if not any(frame.mode == OPERATION_MODE for frame in phase_result.frames):
            self._write_object_skip_report(
                unit,
                "phase result contains no operation frames",
                prompt_path,
            )
            return None

        prompts_cfg = OmegaConf.load(prompt_path)
        prompts = OmegaConf.select(prompts_cfg, "prompts", default={})
        prompts = dict(prompts or {})
        if not prompts:
            self._write_object_skip_report(unit, "object prompts are empty", prompt_path)
            return None
        object_cfg = OmegaConf.merge(
            self.cfg.object_tracking,
            {"prompts": prompts},
        )

        from object_tracking.ObjectTrackingGenerator import ObjectTrackingGenerator

        generator = ObjectTrackingGenerator(
            unit_dir=unit.unit_dir,
            cfg=object_cfg,
            vio_result=vio_result,
            phase_result=phase_result,
            hands=hands,
        )
        return generator.get_object_data()

    @time_it
    def process_lama(
        self,
        unit: ProcessUnit,
        frame_images: dict[int, np.ndarray],
        training_frames: set[int],
        context_frames: set[int],
        fps: float = 30.0,
    ) -> dict:
        from preprocess.object_tracking.LaMa import LaMaGenerator

        return LaMaGenerator(
            unit.unit_dir,
            self.cfg.object_tracking.lama,
            store=FrameArtifactStore(unit.unit_dir),
        ).run(frame_images, training_frames, context_frames, fps=fps)

    @time_it
    def process_visualkpts(
        self,
        unit: ProcessUnit,
        object_result: ObjectTrackingResult,
        frame_images: dict[int, np.ndarray],
        frame_indices: list[int],
        training_frames: set[int],
        vio_result: VIOResult,
        hands: Hands | None,
    ) -> dict:
        from preprocess.object_tracking.VisualKpts import VisualKptsGenerator

        tracks_path = unit.unit_dir / object_result.report["outputs"]["tracks"]
        with tracks_path.open("r", encoding="utf-8") as stream:
            tracks_document = json.load(stream)
        return VisualKptsGenerator(
            unit.unit_dir,
            self.cfg.object_tracking.visualkpts,
            store=FrameArtifactStore(unit.unit_dir),
        ).run(
            frame_images,
            frame_indices,
            training_frames,
            vio_result,
            hands,
            tracks_document,
            float(object_result.report["fps"]),
        )

    @time_it
    def process_dataset(
        self,
        unit: ProcessUnit,
        object_result: ObjectTrackingResult,
        frame_images: dict[int, np.ndarray],
        training_frames: set[int],
        vio_result: VIOResult,
    ) -> dict:
        outputs = object_result.report["outputs"]
        with (unit.unit_dir / outputs["object_poses"]).open("r", encoding="utf-8") as stream:
            pose_document = json.load(stream)
        with (unit.unit_dir / outputs["triangulation"]).open("r", encoding="utf-8") as stream:
            triangulation_document = json.load(stream)
        finished_frames = set(object_result.report.get("finished_frames", []))
        return DatasetGenerator(
            unit.unit_dir,
            cfg=self.cfg.dataset_generation,
            store=FrameArtifactStore(unit.unit_dir),
        ).run(
            pose_document,
            triangulation_document,
            object_result.frames,
            frame_images,
            vio_result,
            float(object_result.report["fps"]),
            training_frames,
            finished_frames,
            unit.video_path,
        )

    @staticmethod
    def _load_frame_images(
        unit: ProcessUnit,
        frame_indices: list[int],
        training_frames: set[int],
    ) -> dict[int, np.ndarray]:
        store = FrameArtifactStore(unit.unit_dir)
        images = {}
        for frame_idx in frame_indices:
            path = store.frame_dir(frame_idx, int(frame_idx) in training_frames) / "rgb.png"
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(f"Missing object-stage RGB frame: {path}")
            images[int(frame_idx)] = image
        return images

    @staticmethod
    def _write_object_skip_report(
        unit: ProcessUnit,
        reason: str,
        prompt_path: Path | None,
    ) -> None:
        output_dir = unit.unit_dir / "preprocess" / "vis" / "objects"
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "status": "skipped",
            "unit_dir": str(unit.unit_dir),
            "world_frame": ARIA_MPS_WORLD_FRAME,
            "world_origin": ARIA_MPS_WORLD_ORIGIN,
            "initial_heading": ARIA_MPS_INITIAL_HEADING,
            "reason": reason,
            "prompt_path": str(prompt_path) if prompt_path else None,
        }
        with (output_dir / "report.json").open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)

    def _load_pending_units(self) -> list[ProcessUnit]:
        """Discover data/<task>/<unit> metadata without decoding video frames."""
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
        task_dirs = sorted(path for path in data_root.iterdir() if path.is_dir())
        for task_dir in task_dirs:
            unit_dirs = sorted(path for path in task_dir.iterdir() if path.is_dir())
            for unit_dir in unit_dirs:
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
                        "Unit must contain exactly one video: "
                        f"task={task_dir.name}, unit={unit_dir.name}, "
                        f"path={unit_dir} (found {len(videos)})"
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
    def _atomic_write_json(path: Path, document: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        )
        temporary_path = Path(handle.name)
        try:
            with handle:
                json.dump(document, handle, indent=2, ensure_ascii=True)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)

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
            "dataset_generation": config_root / "dataset_generation.yaml",
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
            {"dataset_generation": load_yaml(config_paths["dataset_generation"])},
        )
        OmegaConf.resolve(cfg)
        return cfg


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Run GlassEgo preprocessing")
    parser.add_argument(
        "--units",
        nargs="*",
        default=[],
        help="task/unit keys; omit to process all discovered units",
    )
    parser.add_argument(
        "--run-unit",
        type=Path,
        help="internal worker mode: process exactly one unit directory",
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=DEFAULT_CONFIG_ROOT,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="External data root containing <task>/<unit> directories",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    pipeline = PreprocessPipeline(
        config_root=args.config_root,
        data_root=args.data_root,
    )
    if args.run_unit is not None:
        unit_dir = args.run_unit.expanduser().resolve()
        units = [unit for unit in pipeline.pending_units if unit.unit_dir == unit_dir]
        if len(units) != 1:
            print(f"Unable to find exactly one unit: {unit_dir}", file=sys.stderr)
            return 1
        result = pipeline.run_unit(units[0])
        return 0 if result["status"] in {"completed", "skipped"} else 1

    if args.units:
        pipeline.select_units(args.units)
        print(
            f"[preprocess] selected {len(pipeline.pending_units)}/"
            f"{len(args.units)} unit(s)",
            flush=True,
        )
    if not pipeline.pending_units and not pipeline.selection_errors:
        print("[preprocess] no units to process", flush=True)
        return 1
    report = pipeline.run()
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
