import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from omegaconf import OmegaConf

from preprocess.data_types.PhaseTypes import MIN_UNIT_FRAME_COUNT
from preprocess.data_types.VIOTypes import (
    ARIA_MPS_INITIAL_HEADING,
    ARIA_MPS_WORLD_FRAME,
    ARIA_MPS_WORLD_ORIGIN,
    OPENCV_CAMERA_FRAME,
    VIOTrajectory,
    VIOFrame,
    VIOResult,
)
from preprocess.pipeline import (
    UNIT_REPORT_RELATIVE_PATH,
    PreprocessPipeline,
    ProcessUnit,
)


class _FakeCapture:
    def __init__(self, frame_count=0, opened=True):
        self.frame_count = frame_count
        self.opened = opened
        self.index = 0

    def isOpened(self):
        return self.opened

    def read(self):
        if self.index >= self.frame_count:
            return False, None
        self.index += 1
        return True, np.zeros((1, 1, 3), dtype=np.uint8)

    def release(self):
        pass


class PreprocessPipelinePreflightTests(unittest.TestCase):
    def test_short_unit_is_skipped_without_running_later_stages(self):
        pipeline = PreprocessPipeline.__new__(PreprocessPipeline)
        first = ProcessUnit(Path("/tmp/task/short"), Path("short.mp4"))
        pipeline.preflight_unit = Mock(
            return_value={"status": "skipped", "reason": "insufficient frames"}
        )
        pipeline.process_vio = Mock(return_value=object())
        pipeline.process_hands = Mock(return_value=None)
        pipeline.process_phases = Mock(return_value=None)
        pipeline.process_objects = Mock(return_value=None)
        pipeline._release_unit_resources = Mock()

        with tempfile.TemporaryDirectory() as temporary_dir:
            first = ProcessUnit(Path(temporary_dir) / "short", first.video_path)
            result = pipeline.run_unit(first)

        self.assertEqual(result["status"], "skipped")
        pipeline.process_vio.assert_not_called()
        pipeline._release_unit_resources.assert_called_once_with()

    def test_preflight_reports_179_as_skipped_and_180_as_accepted(self):
        pipeline = PreprocessPipeline.__new__(PreprocessPipeline)
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            for frame_count, expected_status in (
                (MIN_UNIT_FRAME_COUNT - 1, "skipped"),
                (MIN_UNIT_FRAME_COUNT, "accepted"),
            ):
                with self.subTest(frame_count=frame_count):
                    unit_dir = root / str(frame_count)
                    unit = ProcessUnit(unit_dir, unit_dir / "video.mp4")
                    with patch(
                        "preprocess.pipeline.cv2.VideoCapture",
                        return_value=_FakeCapture(frame_count),
                    ):
                        report = pipeline.preflight_unit(unit)

                    report_path = (
                        unit_dir / "preprocess" / "vis" / "preflight" / "report.json"
                    )
                    with report_path.open("r", encoding="utf-8") as stream:
                        saved = json.load(stream)
                    self.assertEqual(report["status"], expected_status)
                    self.assertEqual(saved["status"], expected_status)
                    self.assertEqual(saved["threshold_frames"], MIN_UNIT_FRAME_COUNT)
                    self.assertEqual(saved["checked_frames"], frame_count)
                    self.assertEqual(saved["video_path"], str(unit.video_path))
                    if expected_status == "skipped":
                        self.assertEqual(saved["reason_code"], "insufficient_frames")
                    else:
                        self.assertIsNone(saved["reason_code"])

    def test_unopenable_video_raises_input_error(self):
        pipeline = PreprocessPipeline.__new__(PreprocessPipeline)
        unit = ProcessUnit(Path("/tmp/task/bad"), Path("bad.mp4"))

        with patch(
            "preprocess.pipeline.cv2.VideoCapture",
            return_value=_FakeCapture(opened=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "Cannot open input video"):
                pipeline.preflight_unit(unit)

    def test_process_hands_reuses_aligned_cache_before_loading_video(self):
        pipeline = PreprocessPipeline.__new__(PreprocessPipeline)
        pipeline.cfg = OmegaConf.create(
            {
                "hand_tracking": {"enabled": True, "reuse_existing": True},
                "output": {"json_filename": "hamer_hands.json"},
            }
        )
        identity = np.eye(4, dtype=np.float64)
        vio_result = VIOResult(
            trajectory=VIOTrajectory(
                frames=(VIOFrame(0, 100, identity),),
                raw_pose_coverage=1.0,
                T_world_basalt=identity,
            ),
            calibration=None,
            report={},
            pose_path=Path("poses.json"),
            trajectory_path=Path("trajectory.csv"),
            log_path=Path("vio.log"),
        )
        cached_hands = Mock()
        cached_hands.hands = [Mock()]

        with tempfile.TemporaryDirectory() as temporary_dir:
            unit_dir = Path(temporary_dir)
            video_path = unit_dir / "video.mp4"
            video_path.touch()
            unit = ProcessUnit(unit_dir, video_path)
            with (
                patch(
                    "hand_tracking.HandCacheLoader.load_cached_hands",
                    return_value=cached_hands,
                ) as load_cache,
                patch("preprocess.pipeline.build_cam_from_disk") as load_video,
            ):
                result = pipeline.process_hands(unit, vio_result)

        self.assertIs(result, cached_hands)
        load_cache.assert_called_once_with(
            unit.unit_dir,
            [100],
            filename="hamer_hands.json",
        )
        load_video.assert_not_called()

    def test_fresh_hand_inference_removes_only_configured_hand_artifacts(self):
        for invalid_field in (None, "schema_version", "idx", "ts"):
            with self.subTest(invalid_field=invalid_field), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                pipeline, unit, vio_result = self._hand_pipeline(root)
                pipeline.cfg.hand_tracking.reuse_existing = invalid_field is not None
                document = self._hand_cache_document()
                if invalid_field is not None:
                    document[invalid_field] = -1
                cache_paths = [
                    root / "preprocess" / tree / "00000" / "custom_hands.json"
                    for tree in ("temp_data", "all_data")
                ]
                removed_paths = cache_paths + [
                    root / "preprocess/vis/hands/report.json",
                    root / "preprocess/vis/custom_hands.mp4",
                    root / "preprocess/vis/custom_hands.gif",
                    root / "preprocess/vis/hamer_hands_diagnostics.mp4",
                ]
                retained_paths = [
                    root / "preprocess/all_data/00000/training_data.json",
                    root / "preprocess/all_data/00000/rgb.png",
                    root / "preprocess/all_data/00000/other_hands.json",
                    root / "preprocess/temp_data/object_poses.json",
                    root / "preprocess/vis/objects/report.json",
                    root / "preprocess/vis/other_hands.mp4",
                    root / "preprocess/all_data/context/custom_hands.json",
                ]
                for path in removed_paths + retained_paths:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(document) if path in cache_paths else "keep")
                cam = Mock(cam=[Mock()], tss=[100])
                hands = Mock(hands=[Mock()], tss=[100])
                generator = Mock()

                def infer():
                    self.assertTrue(all(not path.exists() for path in removed_paths))
                    self.assertTrue(all(path.read_text() == "keep" for path in retained_paths))
                    return hands

                generator.get_hands_data.side_effect = infer
                generator_class = Mock(return_value=generator)
                with (
                    patch.dict("sys.modules", {
                        "hand_tracking.HaMeRHandsGenerator": SimpleNamespace(
                            HaMeRHandsGenerator=generator_class
                        ),
                    }),
                    patch("preprocess.pipeline.build_cam_from_disk", return_value=cam),
                ):
                    self.assertIs(pipeline.process_hands(unit, vio_result), hands)
                generator.get_hands_data.assert_called_once_with()
                generator.cleanup.assert_called_once_with()

    def test_disabled_hand_tracking_rejects_stale_cache_without_deleting_it(self):
        for reuse_existing in (False, True):
            with self.subTest(reuse_existing=reuse_existing), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                pipeline, unit, vio_result = self._hand_pipeline(root)
                pipeline.cfg.hand_tracking.enabled = False
                pipeline.cfg.hand_tracking.reuse_existing = reuse_existing
                path = root / "preprocess/temp_data/00000/custom_hands.json"
                path.parent.mkdir(parents=True)
                document = self._hand_cache_document()
                document["schema_version"] = 1
                path.write_text(json.dumps(document))
                with patch("preprocess.pipeline.build_cam_from_disk") as load_video:
                    with self.assertRaisesRegex(ValueError, "Hand tracking is disabled"):
                        pipeline.process_hands(unit, vio_result)
                load_video.assert_not_called()
                self.assertTrue(path.is_file())

    @staticmethod
    def _hand_pipeline(root):
        pipeline = PreprocessPipeline.__new__(PreprocessPipeline)
        pipeline.cfg = OmegaConf.create({
            "hand_tracking": {"enabled": True, "reuse_existing": True},
            "output": {
                "json_filename": "custom_hands.json",
                "video_filename": "custom_hands.mp4",
            },
            "phase_segmentation": {
                "hand_input": {"use_cached": True, "filename": "custom_hands.json"},
            },
        })
        video_path = root / "video.mp4"
        video_path.touch()
        identity = np.eye(4)
        vio_result = VIOResult(
            trajectory=VIOTrajectory(
                frames=(VIOFrame(0, 100, identity),),
                raw_pose_coverage=1.0,
                T_world_basalt=identity,
            ),
            calibration=None,
            report={},
            pose_path=Path("poses.json"),
            trajectory_path=Path("trajectory.csv"),
            log_path=Path("vio.log"),
        )
        return pipeline, ProcessUnit(root, video_path), vio_result

    @staticmethod
    def _hand_cache_document():
        return {
            "schema_version": 2,
            "camera_frame": OPENCV_CAMERA_FRAME,
            "world_frame": ARIA_MPS_WORLD_FRAME,
            "world_origin": ARIA_MPS_WORLD_ORIGIN,
            "initial_heading": ARIA_MPS_INITIAL_HEADING,
            "idx": 0,
            "ts": 100,
            "hand_l": None,
            "hand_r": None,
        }

    def test_batch_continues_after_worker_failure_and_returns_nonzero_summary(self):
        pipeline = PreprocessPipeline.__new__(PreprocessPipeline)
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            pipeline.cfg = OmegaConf.create({"paths": {"data_root": root / "data"}})
            pipeline.config_root = root / "config"
            pipeline.selection_errors = []
            pipeline.pending_units = [
                ProcessUnit(root / "task" / name, root / "task" / name / "video.mp4")
                for name in ("first", "second", "third")
            ]

            def fake_worker(command, **_kwargs):
                unit_dir = Path(command[command.index("--run-unit") + 1])
                index = {unit.unit_dir.name: position for position, unit in enumerate(pipeline.pending_units)}[
                    unit_dir.name
                ]
                if index != 1:
                    pipeline._atomic_write_json(
                        unit_dir / UNIT_REPORT_RELATIVE_PATH,
                        {"status": "completed", "unit_dir": str(unit_dir)},
                    )
                return subprocess.CompletedProcess(command, -9 if index == 1 else 0)

            with patch("preprocess.pipeline.subprocess.run", side_effect=fake_worker) as run_worker:
                report = pipeline.run()

            self.assertEqual(run_worker.call_count, 3)
            self.assertEqual(
                [item["status"] for item in report["results"]],
                ["completed", "failed", "completed"],
            )
            self.assertEqual(report["results"][1]["worker_signal"], 9)
            self.assertEqual(report["status"], "failed")
            with (root / "data" / "preprocess_batch_report.json").open() as stream:
                saved = json.load(stream)
            self.assertEqual(saved["status"], "failed")

    def test_worker_exception_is_reported_and_cleanup_still_runs(self):
        pipeline = PreprocessPipeline.__new__(PreprocessPipeline)
        pipeline.preflight_unit = Mock(return_value={"status": "accepted"})
        pipeline.process_vio = Mock(side_effect=RuntimeError("VIO failed"))
        pipeline._release_unit_resources = Mock()

        with tempfile.TemporaryDirectory() as temporary_dir:
            unit_dir = Path(temporary_dir)
            unit = ProcessUnit(unit_dir, unit_dir / "video.mp4")
            result = pipeline.run_unit(unit)
            with (unit_dir / UNIT_REPORT_RELATIVE_PATH).open() as stream:
                saved = json.load(stream)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(saved["stage"], "vio")
        self.assertEqual(saved["error_type"], "RuntimeError")
        pipeline._release_unit_resources.assert_called_once_with()

    def test_select_units_preserves_request_order_and_reports_missing_keys(self):
        pipeline = PreprocessPipeline.__new__(PreprocessPipeline)
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            first = ProcessUnit(root / "task" / "first", root / "task" / "first" / "video.mp4")
            second = ProcessUnit(root / "task" / "second", root / "task" / "second" / "video.mp4")
            pipeline.pending_units = [first, second]

            selected = pipeline.select_units(["task/second", "task/missing", "task/first"])

        self.assertEqual(selected, [second, first])
        self.assertEqual(pipeline.selection_errors[0]["key"], "task/missing")


if __name__ == "__main__":
    unittest.main()
