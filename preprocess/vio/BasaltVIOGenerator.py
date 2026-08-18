import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from preprocess.data_types.VIOTypes import VIOResult, VIOTrajectory
from preprocess.vio.BasaltAdapter import BasaltAdapter
from preprocess.vio.SensorSynchronizer import SensorSynchronizer
from preprocess.vio.VIODataLoader import VIODataLoader
from preprocess.vio.VIOPoseProcessor import VIOPoseProcessor


class BasaltVIOGenerator:
    """协调单个数据单元的 Basalt VIO 处理。"""

    CACHE_VERSION = 1

    def __init__(self, unit_dir: str | Path, cfg):
        self.unit_dir = Path(unit_dir).expanduser().resolve()
        self.cfg = cfg
        self.loader = VIODataLoader(self.unit_dir)
        self.adapter = BasaltAdapter(cfg.basalt)
        self.pose_processor = VIOPoseProcessor(
            min_pose_coverage=cfg.min_pose_coverage,
            max_interpolation_gap_ms=cfg.max_interpolation_gap_ms,
        )
        self.output_dir = self.unit_dir / "preprocess" / "vio"
        self.trajectory_path = self.output_dir / "basalt_trajectory.csv"
        self.pose_path = self.output_dir / "poses.json"
        self.report_path = self.output_dir / "report.json"
        self.log_path = self.output_dir / "basalt.log"

    def get_camera_poses(self, force: bool = False) -> VIOResult:
        """运行或复用 VIO，并返回逐帧相机位姿。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        raw_sensor_data = self.loader.load_sensor_data()
        calibration = self.loader.load_calibration()
        executable_identity = self.adapter.executable_identity()
        fingerprint = self._build_fingerprint(executable_identity)

        if bool(self.cfg.reuse_existing) and not force:
            cached = self._load_cached_result(
                fingerprint,
                calibration,
                len(raw_sensor_data.camera),
            )
            if cached is not None:
                return cached

        try:
            sensor_data = SensorSynchronizer().synchronize(
                raw_sensor_data,
                calibration,
            )
            with tempfile.TemporaryDirectory(prefix="glassego_basalt_") as tmp:
                work_dir = Path(tmp)
                try:
                    basalt_result = self.adapter.run(
                        self.loader.paths.video_path,
                        sensor_data,
                        calibration,
                        work_dir,
                    )
                except Exception:
                    temporary_log = work_dir / "basalt.log"
                    if temporary_log.is_file():
                        self._atomic_copy(temporary_log, self.log_path)
                    raise

                trajectory = self.pose_processor.process(
                    sensor_data.camera,
                    basalt_result.trajectory,
                    calibration,
                )
                report = self._build_report(
                    fingerprint,
                    executable_identity,
                    sensor_data,
                    trajectory,
                    basalt_result.warnings,
                )
                self._atomic_copy(
                    basalt_result.trajectory_path,
                    self.trajectory_path,
                )
                self._atomic_copy(basalt_result.log_path, self.log_path)
                self._atomic_write_json(self.pose_path, trajectory.to_dict())
                self._atomic_write_json(self.report_path, report)
        except Exception as exc:
            self._atomic_write_json(
                self.report_path,
                {
                    "status": "failed",
                    "unit_dir": str(self.unit_dir),
                    "input_fingerprint": fingerprint,
                    "error": f"{type(exc).__name__}: {exc}",
                    "log": str(self.log_path),
                },
            )
            raise

        return VIOResult(
            trajectory=trajectory,
            calibration=calibration,
            report=report,
            pose_path=self.pose_path,
            trajectory_path=self.trajectory_path,
            log_path=self.log_path,
        )

    def _load_cached_result(
        self,
        fingerprint: str,
        calibration,
        expected_frames: int,
    ) -> VIOResult | None:
        if not all(
            path.is_file()
            for path in (
                self.report_path,
                self.pose_path,
                self.trajectory_path,
                self.log_path,
            )
        ):
            return None
        try:
            with self.report_path.open("r", encoding="utf-8") as stream:
                report = json.load(stream)
            if report.get("status") not in ("completed", "completed_with_warnings"):
                return None
            if report.get("input_fingerprint") != fingerprint:
                return None
            if self.trajectory_path.stat().st_size == 0:
                return None
            trajectory = VIOTrajectory.load_json(self.pose_path)
            if len(trajectory.frames) != expected_frames:
                return None
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

        cached_report = dict(report)
        cached_report["cache_reused"] = True
        return VIOResult(
            trajectory=trajectory,
            calibration=calibration,
            report=cached_report,
            pose_path=self.pose_path,
            trajectory_path=self.trajectory_path,
            log_path=self.log_path,
        )

    def _build_fingerprint(self, executable_identity: dict) -> str:
        digest = hashlib.sha256()
        digest.update(f"glassego-vio-cache-{self.CACHE_VERSION}".encode("utf-8"))
        for path in (
            self.loader.paths.video_path,
            self.loader.paths.camera_csv_path,
            self.loader.paths.imu_csv_path,
            self.loader.paths.calibration_path,
        ):
            digest.update(path.name.encode("utf-8"))
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)

        config_data = OmegaConf.to_container(self.cfg, resolve=True)
        digest.update(
            json.dumps(
                config_data,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(
            json.dumps(
                executable_identity,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return digest.hexdigest()

    def _build_report(
        self,
        fingerprint,
        executable_identity,
        sensor_data,
        trajectory,
        warnings,
    ) -> dict:
        return {
            "status": "completed_with_warnings" if warnings else "completed",
            "unit_dir": str(self.unit_dir),
            "video_path": str(self.loader.paths.video_path),
            "input_fingerprint": fingerprint,
            "cache_reused": False,
            "basalt_executable": executable_identity,
            "camera_frames": len(sensor_data.camera),
            "imu_samples": len(sensor_data.imu),
            "raw_gyroscope_samples": sensor_data.raw_gyroscope_samples,
            "raw_accelerometer_samples": sensor_data.raw_accelerometer_samples,
            "imu_update_rate_hz": sensor_data.imu_update_rate_hz,
            "clock_sync": sensor_data.clock_sync,
            "trajectory_poses": len(trajectory.frames),
            "raw_pose_coverage": trajectory.raw_pose_coverage,
            "final_pose_coverage": 1.0,
            "interpolated_frames": sum(
                frame.interpolated for frame in trajectory.frames
            ),
            "warnings": list(warnings),
            "outputs": {
                "trajectory": str(self.trajectory_path),
                "poses": str(self.pose_path),
                "log": str(self.log_path),
            },
        }

    @staticmethod
    def _atomic_copy(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            dir=destination.parent,
            delete=False,
        )
        temporary_path = Path(handle.name)
        handle.close()
        try:
            shutil.copyfile(source, temporary_path)
            os.replace(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _atomic_write_json(path: Path, document: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            dir=path.parent,
            delete=False,
        )
        temporary_path = Path(handle.name)
        try:
            with handle:
                json.dump(document, handle, indent=2)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
