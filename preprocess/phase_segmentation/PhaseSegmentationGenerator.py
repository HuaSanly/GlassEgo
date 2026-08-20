import json
import os
import tempfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from data_types.HandsTypes import Hands
from preprocess.data_types.PhaseTypes import (
    CandidateSegment,
    PHASE_NAMES,
    PhaseFrame,
    PhaseSequence,
)
from preprocess.data_types.VIOTypes import VIOResult
from preprocess.phase_segmentation.PhaseSegmentationOps import PhaseSegmentationOps


class PhaseSegmentationGenerator:
    """使用 VIO 与可选手部运动学切割单个数据单元的动作阶段。"""

    def __init__(self, unit_dir: str | Path, video_path: str | Path, cfg):
        self.unit_dir = Path(unit_dir).expanduser().resolve()
        self.video_path = Path(video_path).expanduser().resolve()
        self.cfg = cfg
        self.output_dir = self.unit_dir / "preprocess" / "phases"
        self.result_path = self.output_dir / "phases.json"
        self.report_path = self.output_dir / "report.json"
        self.analysis_path = self.output_dir / "phases_analysis.png"
        self.video_path_out = self.output_dir / cfg.output.video_filename

    def get_phases(
        self,
        vio_result: VIOResult,
        hands: Hands | None = None,
    ) -> PhaseSequence:
        """生成完整阶段序列，并按配置导出报告与可视化。"""
        if not isinstance(vio_result, VIOResult):
            raise TypeError("vio_result must be a VIOResult")
        if not self.video_path.is_file():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        trajectory = vio_result.trajectory
        if not trajectory.frames:
            raise ValueError("VIO trajectory contains no frames")
        if hands is not None and len(hands.hands) != len(trajectory.frames):
            raise ValueError(
                "Hands and VIO trajectories must have the same frame count: "
                f"hands={len(hands.hands)}, vio={len(trajectory.frames)}"
            )

        frame_indices, timestamps, linear_speed, angular_speed, yaw = self._kinematics(
            trajectory.frames
        )
        stop = PhaseSegmentationOps.compute_stop_mask(
            linear_speed,
            angular_speed,
            float(self.cfg.v_stop_thresh),
            float(self.cfg.w_stop_thresh),
            int(self.cfg.stop_hold_frames),
            int(self.cfg.stop_debounce_frames),
        )
        stop = PhaseSegmentationOps.apply_yaw_veto(
            stop,
            yaw,
            int(self.cfg.stop_min_on_frames),
            float(self.cfg.stop_yaw_veto_deg),
        )
        stop = PhaseSegmentationOps.delay_stop_start(
            stop,
            int(self.cfg.stop_offset_frames),
        )
        mode = PhaseSegmentationOps.compute_modes(
            stop,
            linear_speed,
            angular_speed,
            float(self.cfg.w_rot_thresh),
            float(self.cfg.v_rot_max),
            int(self.cfg.mode_median_window_frames),
            int(self.cfg.mode_min_run_frames),
            int(self.cfg.transition_offset_frames),
        )

        hand_refined = False
        if bool(self.cfg.hand_refinement.enabled) and hands is not None:
            hand_speed = self._hand_speeds(hands)
            if np.isfinite(hand_speed).any():
                mode = PhaseSegmentationOps.refine_stop_with_hand_speed(
                    mode,
                    hand_speed,
                    float(self.cfg.hand_refinement.velocity_threshold_mps),
                    int(self.cfg.hand_refinement.stable_frames),
                    int(self.cfg.hand_refinement.boundary_frames),
                    float(self.cfg.hand_refinement.min_valid_fraction),
                )
                hand_refined = True
        mode = PhaseSegmentationOps.inject_finished(
            mode,
            int(self.cfg.finished_frames),
        )

        frames = tuple(
            PhaseFrame(
                frame_idx=int(frame_idx),
                timestamp_ns=int(timestamp),
                mode=int(mode[index]),
                stop=bool(stop[index]),
                linear_speed_mps=float(linear_speed[index]),
                angular_speed_rad_s=float(angular_speed[index]),
                yaw_unwrapped_deg=float(yaw[index]),
            )
            for index, (frame_idx, timestamp) in enumerate(
                zip(frame_indices, timestamps)
            )
        )
        candidate_segments = self._candidate_segments(frames)
        duration_s = (
            (int(timestamps[-1]) - int(timestamps[0])) / 1_000_000_000.0
            if len(timestamps) > 1
            else 0.0
        )
        summary = {
            "status": "completed",
            "unit_dir": str(self.unit_dir),
            "video_path": str(self.video_path),
            "total_frames": len(frames),
            "duration_s": duration_s,
            "raw_vio_pose_coverage": float(trajectory.raw_pose_coverage),
            "hand_refinement_applied": hand_refined,
            "mode_counts": {
                name: int(np.sum(mode == phase))
                for phase, name in PHASE_NAMES.items()
            },
            "candidate_segment_count": len(candidate_segments),
            "config": OmegaConf.to_container(self.cfg, resolve=True),
        }
        sequence = PhaseSequence(
            frames=frames,
            candidate_segments=tuple(candidate_segments),
            summary=summary,
        )
        self._save_outputs(sequence)
        return sequence

    @staticmethod
    def _kinematics(vio_frames):
        frame_indices = np.asarray([frame.frame_idx for frame in vio_frames], dtype=np.int32)
        timestamps = np.asarray([frame.timestamp_ns for frame in vio_frames], dtype=np.int64)
        if not np.all(np.diff(frame_indices) == 1):
            raise ValueError("VIO frame indices must be contiguous")
        if len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0):
            raise ValueError("VIO timestamps must be strictly increasing")

        positions = np.asarray([frame.c2w[:3, 3] for frame in vio_frames], dtype=np.float64)
        yaw_rad = np.unwrap(
            np.asarray(
                [
                    np.arctan2(frame.c2w[1, 0], frame.c2w[0, 0])
                    for frame in vio_frames
                ],
                dtype=np.float64,
            )
        )
        linear_speed = np.zeros(len(vio_frames), dtype=np.float64)
        angular_speed = np.zeros(len(vio_frames), dtype=np.float64)
        if len(vio_frames) > 1:
            dt = np.diff(timestamps).astype(np.float64) / 1_000_000_000.0
            if np.any(dt <= 0.0):
                raise ValueError("VIO timestamps must have a positive interval")
            linear_speed[1:] = np.linalg.norm(np.diff(positions, axis=0), axis=1) / dt
            angular_speed[1:] = np.abs(np.diff(yaw_rad)) / dt
        return frame_indices, timestamps, linear_speed, angular_speed, np.degrees(yaw_rad)

    @staticmethod
    def _hand_speeds(hands: Hands) -> np.ndarray:
        values = np.full(len(hands.hands), np.nan, dtype=np.float64)
        for index, frame in enumerate(hands.hands):
            speeds = []
            for hand in (frame.hand_r, frame.hand_l):
                if (
                    hand is None
                    or hand.midpoint_translation_opt_world is None
                    or hand.midpoint_lin_vel_opt_world is None
                ):
                    continue
                velocity = np.asarray(hand.midpoint_lin_vel_opt_world, dtype=np.float64)
                if velocity.shape == (3,) and np.all(np.isfinite(velocity)):
                    speeds.append(float(np.linalg.norm(velocity)))
            if speeds:
                values[index] = max(speeds)
        return values

    @staticmethod
    def _candidate_segments(frames: tuple[PhaseFrame, ...]) -> list[CandidateSegment]:
        mode = np.asarray([frame.mode for frame in frames], dtype=np.int32)
        segments = []
        for start, end, value in PhaseSegmentationOps.find_segments(mode):
            if value != 0:
                continue
            first = frames[start]
            last = frames[end]
            segments.append(
                CandidateSegment(
                    start_frame_idx=first.frame_idx,
                    end_frame_idx=last.frame_idx,
                    start_timestamp_ns=first.timestamp_ns,
                    end_timestamp_ns=last.timestamp_ns,
                )
            )
        return segments

    def _save_outputs(self, sequence: PhaseSequence) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if bool(self.cfg.output.export_json):
            self._atomic_write_json(self.result_path, sequence.to_dict())
        self._atomic_write_json(self.report_path, sequence.summary)
        if bool(self.cfg.output.export_analysis):
            self._save_analysis(sequence)
        if bool(self.cfg.output.export_video):
            self._save_video(sequence)

    def _save_analysis(self, sequence: PhaseSequence) -> None:
        frames = sequence.frames
        time_s = np.asarray(
            [frame.timestamp_ns - frames[0].timestamp_ns for frame in frames],
            dtype=np.float64,
        ) / 1_000_000_000.0
        linear_speed = [frame.linear_speed_mps for frame in frames]
        angular_speed = [frame.angular_speed_rad_s for frame in frames]
        yaw = [frame.yaw_unwrapped_deg for frame in frames]
        modes = [frame.mode for frame in frames]

        figure, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
        axes[0].plot(time_s, linear_speed, color="#1976d2", linewidth=1.0)
        axes[0].axhline(self.cfg.v_stop_thresh, color="#d32f2f", linestyle="--")
        axes[0].set_ylabel("v (m/s)")
        axes[1].plot(time_s, angular_speed, color="#388e3c", linewidth=1.0)
        axes[1].axhline(self.cfg.w_stop_thresh, color="#d32f2f", linestyle="--")
        axes[1].axhline(self.cfg.w_rot_thresh, color="#f57c00", linestyle="--")
        axes[1].set_ylabel("w (rad/s)")
        axes[2].plot(time_s, yaw, color="#6a1b9a", linewidth=1.0)
        axes[2].set_ylabel("yaw (deg)")
        colors = ["#e0e0e0", "#b3e5fc", "#fff9c4", "#ffccbc", "#c8e6c9"]
        axes[3].scatter(time_s, modes, c=[colors[mode] for mode in modes], s=4)
        axes[3].set_yticks(sorted(PHASE_NAMES))
        axes[3].set_yticklabels([PHASE_NAMES[index] for index in sorted(PHASE_NAMES)])
        axes[3].set_xlabel("time (s)")
        axes[3].set_ylabel("phase")
        figure.tight_layout()
        figure.savefig(self.analysis_path, dpi=180)
        plt.close(figure)

    def _save_video(self, sequence: PhaseSequence) -> None:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video for phase visualization: {self.video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0.0 or width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError("Video metadata is invalid for phase visualization")
        writer = cv2.VideoWriter(
            str(self.video_path_out),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            writer.release()
            raise RuntimeError(f"Cannot open phase VideoWriter: {self.video_path_out}")
        index = 0
        try:
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                if index >= len(sequence.frames):
                    raise RuntimeError("Video contains more frames than phase sequence")
                writer.write(self._draw_phase_hud(image, sequence.frames[index]))
                index += 1
        finally:
            cap.release()
            writer.release()
        if index != len(sequence.frames):
            raise RuntimeError(
                "Video frame count does not match phase sequence: "
                f"video={index}, phases={len(sequence.frames)}"
            )
        if not self.video_path_out.is_file() or self.video_path_out.stat().st_size == 0:
            raise RuntimeError(f"Phase visualization was not created: {self.video_path_out}")

    @staticmethod
    def _draw_phase_hud(image_bgr: np.ndarray, frame: PhaseFrame) -> np.ndarray:
        image = image_bgr.copy()
        color = {
            0: (190, 190, 190),
            1: (255, 220, 80),
            2: (80, 220, 255),
            3: (90, 150, 255),
            4: (100, 210, 120),
        }[frame.mode]
        cv2.rectangle(image, (12, 12), (285, 118), (20, 20, 20), -1)
        cv2.rectangle(image, (12, 12), (285, 118), color, 2)
        cv2.putText(image, frame.mode_name, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(
            image,
            f"v: {frame.linear_speed_mps:.3f} m/s",
            (25, 73),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"w: {frame.angular_speed_rad_s:.3f} rad/s",
            (25, 97),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return image

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
