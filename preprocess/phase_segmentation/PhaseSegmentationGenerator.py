import json
import os
import tempfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from preprocess.data_types.HandsTypes import Hands
from preprocess.data_types.PhaseTypes import (
    CandidateSegment,
    FINISHED_MODE,
    FINISHED_TAIL_FRAMES,
    FORCED_NON_OPERATION_PREFIX_FRAMES,
    MIN_CLASSIFIED_MIDDLE_FRAMES,
    MIN_UNIT_FRAME_COUNT,
    NON_OPERATION_MODE,
    OPERATION_MODE,
    PHASE_NAMES,
    PHASE_SCHEMA_VERSION,
    PhaseFrame,
    PhaseSequence,
)
from preprocess.data_types.VIOTypes import (
    ARIA_MPS_INITIAL_HEADING,
    ARIA_MPS_WORLD_FRAME,
    ARIA_MPS_WORLD_ORIGIN,
    ARIA_MPS_YAW_CONVENTION,
    VIOResult,
)
from preprocess.phase_segmentation.PhaseSegmentationOps import PhaseSegmentationOps


class PhaseSegmentationGenerator:
    """使用 VIO 和预计算手部证据切割操作/非操作阶段。"""

    def __init__(self, unit_dir: str | Path, video_path: str | Path, cfg):
        self.unit_dir = Path(unit_dir).expanduser().resolve()
        self.video_path = Path(video_path).expanduser().resolve()
        self.cfg = cfg
        self.output_dir = self.unit_dir / "preprocess" / "temp_data"
        self.result_path = self.output_dir / "phases.json"
        self.vis_dir = self.unit_dir / "preprocess" / "vis" / "phases"
        self.report_path = self.vis_dir / "report.json"
        self.analysis_path = self.vis_dir / "phases_analysis.png"
        self.video_path_out = self.unit_dir / "preprocess" / "vis" / cfg.output.video_filename

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
        if trajectory.world_frame != ARIA_MPS_WORLD_FRAME:
            raise ValueError("Phase segmentation requires Aria MPS VIO poses")
        if not trajectory.frames:
            raise ValueError("VIO trajectory contains no frames")
        if hands is not None and hands.world_frame != ARIA_MPS_WORLD_FRAME:
            raise ValueError("Phase segmentation requires Aria MPS hand poses")
        if hands is not None and len(hands.hands) != len(trajectory.frames):
            raise ValueError(
                "Hands and VIO trajectories must have the same frame count: "
                f"hands={len(hands.hands)}, vio={len(trajectory.frames)}"
            )

        frame_indices, timestamps, linear_speed, angular_speed, yaw = self._kinematics(
            trajectory.frames
        )
        if len(frame_indices) < MIN_UNIT_FRAME_COUNT:
            raise ValueError(
                "Phase segmentation requires at least "
                f"{MIN_UNIT_FRAME_COUNT} frames "
                f"({FORCED_NON_OPERATION_PREFIX_FRAMES} fixed non-operation + "
                f"{MIN_CLASSIFIED_MIDDLE_FRAMES} classified + "
                f"{FINISHED_TAIL_FRAMES} finished); "
                f"got {len(frame_indices)}"
            )
        operation_cfg = self.cfg.operation
        feature_window = int(operation_cfg.feature_window_frames)
        if feature_window < 1:
            raise ValueError("operation.feature_window_frames must be positive")
        smoothed_linear = PhaseSegmentationOps.median_filter_values(
            linear_speed,
            feature_window,
        )
        smoothed_angular = PhaseSegmentationOps.median_filter_values(
            angular_speed,
            feature_window,
        )
        linear_score = PhaseSegmentationOps.sigmoid_score(
            smoothed_linear,
            float(operation_cfg.linear_speed_reference_mps),
            float(operation_cfg.linear_speed_scale_mps),
        )
        angular_score = PhaseSegmentationOps.sigmoid_score(
            smoothed_angular,
            float(operation_cfg.angular_speed_reference_rad_s),
            float(operation_cfg.angular_speed_scale_rad_s),
        )
        camera_weights = np.asarray(
            [operation_cfg.linear_weight, operation_cfg.angular_weight],
            dtype=np.float64,
        )
        if np.any(camera_weights < 0.0) or float(camera_weights.sum()) <= 0.0:
            raise ValueError("Camera motion weights must be non-negative and non-zero")
        camera_weights /= camera_weights.sum()
        camera_motion_score = np.clip(
            camera_weights[0] * linear_score
            + camera_weights[1] * angular_score,
            0.0,
            1.0,
        )
        hand_evidence = self._hand_evidence(
            hands,
            len(frame_indices),
            operation_cfg,
        )
        non_operation_score = PhaseSegmentationOps.combine_non_operation_score(
            camera_motion_score,
            hand_evidence["presence"],
            hand_evidence["motion"],
            hand_evidence["grasp"],
            hand_evidence["available"],
            float(operation_cfg.hand_presence_weight),
            float(operation_cfg.hand_motion_weight),
            float(operation_cfg.grasp_weight),
            float(operation_cfg.hand_operation_weight),
            float(operation_cfg.no_hand_weight),
        )
        mode, operation_confidence, non_operation_confidence = (
            self._classify_phases(non_operation_score, operation_cfg)
        )

        frames = tuple(
            PhaseFrame(
                frame_idx=int(frame_idx),
                timestamp_ns=int(timestamp),
                mode=int(mode[index]),
                linear_speed_mps=float(linear_speed[index]),
                angular_speed_rad_s=float(angular_speed[index]),
                yaw_unwrapped_deg=float(yaw[index]),
                operation_confidence=float(operation_confidence[index]),
                non_operation_confidence=float(non_operation_confidence[index]),
                camera_motion_score=float(camera_motion_score[index]),
                hand_presence_score=self._optional_score(
                    hand_evidence["presence"],
                    hand_evidence["available"],
                    index,
                ),
                hand_motion_score=self._optional_score(
                    hand_evidence["motion"],
                    hand_evidence["available"],
                    index,
                ),
                grasp_score=self._optional_score(
                    hand_evidence["grasp"],
                    hand_evidence["available"],
                    index,
                ),
                hand_evidence_available=bool(hand_evidence["available"][index]),
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
            "schema_version": PHASE_SCHEMA_VERSION,
            "world_frame": ARIA_MPS_WORLD_FRAME,
            "world_origin": ARIA_MPS_WORLD_ORIGIN,
            "initial_heading": ARIA_MPS_INITIAL_HEADING,
            "yaw_convention": ARIA_MPS_YAW_CONVENTION,
            "unit_dir": str(self.unit_dir),
            "video_path": str(self.video_path),
            "total_frames": len(frames),
            "duration_s": duration_s,
            "raw_vio_pose_coverage": float(trajectory.raw_pose_coverage),
            "hand_evidence_applied": hands is not None,
            "hand_input_available": bool(hands is not None),
            "hand_evidence_coverage": float(np.mean(hand_evidence["available"])),
            "detected_hand_ratio": float(np.mean(hand_evidence["presence"] > 0.0)),
            "mode_counts": {
                name: int(np.sum(mode == phase))
                for phase, name in PHASE_NAMES.items()
            },
            "operation_ratio": float(np.mean(mode == OPERATION_MODE)),
            "non_operation_ratio": float(np.mean(mode == NON_OPERATION_MODE)),
            "finished_ratio": float(np.mean(mode == FINISHED_MODE)),
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
    def _classify_phases(non_operation_score: np.ndarray, operation_cfg):
        scores = np.asarray(non_operation_score, dtype=np.float64).reshape(-1)
        if len(scores) < MIN_UNIT_FRAME_COUNT:
            raise ValueError(
                f"Phase classification requires at least {MIN_UNIT_FRAME_COUNT} frames"
            )

        middle_start = FORCED_NON_OPERATION_PREFIX_FRAMES
        middle_end = len(scores) - FINISHED_TAIL_FRAMES
        middle_mode = PhaseSegmentationOps.classify_binary_phases(
            scores[middle_start:middle_end],
            float(operation_cfg.non_operation_enter_threshold),
            float(operation_cfg.non_operation_exit_threshold),
            int(operation_cfg.non_operation_enter_frames),
            int(operation_cfg.non_operation_exit_frames),
            int(operation_cfg.min_non_operation_frames),
            initial_non_operation=True,
        )
        mode = np.full(len(scores), NON_OPERATION_MODE, dtype=np.int32)
        mode[middle_start:middle_end] = middle_mode
        mode[middle_end:] = FINISHED_MODE

        operation_confidence = 1.0 - scores
        non_operation_confidence = scores.copy()
        operation_confidence[:middle_start] = 0.0
        non_operation_confidence[:middle_start] = 1.0
        operation_confidence[middle_end:] = 0.0
        non_operation_confidence[middle_end:] = 0.0
        return mode, operation_confidence, non_operation_confidence

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
                    np.arctan2(frame.c2w[0, 2], -frame.c2w[2, 2])
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
            for index, interval in enumerate(dt, start=1):
                previous_rotation = vio_frames[index - 1].c2w[:3, :3]
                current_rotation = vio_frames[index].c2w[:3, :3]
                relative_rotation = previous_rotation.T @ current_rotation
                angular_speed[index] = Rotation.from_matrix(relative_rotation).magnitude() / interval
        return frame_indices, timestamps, linear_speed, angular_speed, np.degrees(yaw_rad)

    @staticmethod
    def _hand_evidence(hands: Hands | None, frame_count: int, cfg) -> dict[str, np.ndarray]:
        presence = np.zeros(frame_count, dtype=np.float64)
        motion = np.zeros(frame_count, dtype=np.float64)
        grasp = np.zeros(frame_count, dtype=np.float64)
        available = np.zeros(frame_count, dtype=bool)
        if hands is None:
            return {
                "presence": presence,
                "motion": motion,
                "grasp": grasp,
                "available": available,
            }
        if len(hands.hands) != frame_count:
            raise ValueError("Hand evidence and VIO trajectories must have equal lengths")

        for index, frame in enumerate(hands.hands):
            available[index] = True
            missing_probability = 1.0
            motion_scores = []
            grasp_scores = []
            for hand in (frame.hand_r, frame.hand_l):
                if hand is None or hand.confidence is None:
                    continue
                confidence = float(hand.confidence)
                if not np.isfinite(confidence) or confidence < float(cfg.min_hand_confidence):
                    continue
                confidence = float(np.clip(confidence, 0.0, 1.0))
                missing_probability *= 1.0 - confidence
                linear_speed = PhaseSegmentationGenerator._max_vector_speed(
                    hand,
                    ("midpoint_lin_vel_opt_world", "wrist_lin_vel_opt_world"),
                )
                angular_speed = PhaseSegmentationGenerator._max_vector_speed(
                    hand,
                    ("midpoint_ang_vel_opt_world", "wrist_ang_vel_opt_world"),
                )
                scores = []
                if linear_speed is not None:
                    scores.append(
                        PhaseSegmentationOps.sigmoid_score(
                            np.asarray([linear_speed]),
                            float(cfg.hand_linear_speed_reference_mps),
                            float(cfg.hand_linear_speed_scale_mps),
                        )[0]
                    )
                if angular_speed is not None:
                    scores.append(
                        PhaseSegmentationOps.sigmoid_score(
                            np.asarray([angular_speed]),
                            float(cfg.hand_angular_speed_reference_rad_s),
                            float(cfg.hand_angular_speed_scale_rad_s),
                        )[0]
                    )
                if scores:
                    motion_scores.append(max(scores))
                grasp_scores.append(confidence * hand.grasp_score)
            presence[index] = 1.0 - missing_probability
            motion[index] = max(motion_scores, default=0.0)
            grasp[index] = max(grasp_scores, default=0.0)

        window = int(cfg.feature_window_frames)
        presence = PhaseSegmentationOps.median_filter_values(presence, window)
        motion = PhaseSegmentationOps.median_filter_values(motion, window)
        grasp = PhaseSegmentationOps.median_filter_values(grasp, window)
        return {
            "presence": np.clip(presence, 0.0, 1.0),
            "motion": np.clip(motion, 0.0, 1.0),
            "grasp": np.clip(grasp, 0.0, 1.0),
            "available": available,
        }

    @staticmethod
    def _max_vector_speed(hand, field_names: tuple[str, ...]) -> float | None:
        speeds = []
        for field_name in field_names:
            value = getattr(hand, field_name, None)
            if value is None:
                continue
            vector = np.asarray(value, dtype=np.float64)
            if vector.shape == (3,) and np.all(np.isfinite(vector)):
                speeds.append(float(np.linalg.norm(vector)))
        return max(speeds) if speeds else None

    @staticmethod
    def _optional_score(values: np.ndarray, available: np.ndarray, index: int) -> float | None:
        return float(values[index]) if bool(available[index]) else None

    @staticmethod
    def _candidate_segments(frames: tuple[PhaseFrame, ...]) -> list[CandidateSegment]:
        mode = np.asarray([frame.mode for frame in frames], dtype=np.int32)
        segments = []
        for start, end, value in PhaseSegmentationOps.find_segments(mode):
            if value != OPERATION_MODE:
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
        self.vis_dir.mkdir(parents=True, exist_ok=True)
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
        operation_cfg = self.cfg.operation

        camera_score = [frame.camera_motion_score for frame in frames]
        non_operation_score = [frame.non_operation_confidence for frame in frames]
        hand_presence = [
            np.nan if frame.hand_presence_score is None else frame.hand_presence_score
            for frame in frames
        ]
        hand_motion = [
            np.nan if frame.hand_motion_score is None else frame.hand_motion_score
            for frame in frames
        ]

        figure, axes = plt.subplots(5, 1, figsize=(15, 12), sharex=True)
        axes[0].plot(time_s, linear_speed, color="#1976d2", linewidth=1.0)
        axes[0].axhline(
            operation_cfg.linear_speed_reference_mps,
            color="#d32f2f",
            linestyle="--",
        )
        axes[0].set_ylabel("v (m/s)")
        axes[1].plot(time_s, angular_speed, color="#388e3c", linewidth=1.0)
        axes[1].axhline(
            operation_cfg.angular_speed_reference_rad_s,
            color="#d32f2f",
            linestyle="--",
        )
        axes[1].set_ylabel("w (rad/s)")
        axes[2].plot(time_s, camera_score, label="camera", linewidth=1.0)
        axes[2].plot(time_s, hand_presence, label="hand presence", linewidth=1.0)
        axes[2].plot(time_s, hand_motion, label="hand motion", linewidth=1.0)
        axes[2].plot(
            time_s,
            non_operation_score,
            label="non-operation",
            color="#d32f2f",
            linewidth=1.2,
        )
        axes[2].axhline(
            operation_cfg.non_operation_enter_threshold,
            color="#d32f2f",
            linestyle="--",
            linewidth=0.8,
        )
        axes[2].axhline(
            operation_cfg.non_operation_exit_threshold,
            color="#388e3c",
            linestyle="--",
            linewidth=0.8,
        )
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].set_ylabel("evidence")
        axes[2].legend(loc="upper right", ncol=4, fontsize=8)
        axes[3].plot(time_s, yaw, color="#6a1b9a", linewidth=1.0)
        axes[3].set_ylabel("yaw (deg)")
        colors = {
            OPERATION_MODE: "#c8e6c9",
            NON_OPERATION_MODE: "#eeeeee",
            FINISHED_MODE: "#ffe0b2",
        }
        axes[4].scatter(time_s, modes, c=[colors[mode] for mode in modes], s=4)
        axes[4].set_yticks(sorted(PHASE_NAMES))
        axes[4].set_yticklabels([PHASE_NAMES[index] for index in sorted(PHASE_NAMES)])
        axes[4].set_xlabel("time (s)")
        axes[4].set_ylabel("phase")
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
            OPERATION_MODE: (100, 210, 120),
            NON_OPERATION_MODE: (190, 190, 190),
            FINISHED_MODE: (80, 180, 255),
        }[frame.mode]
        cv2.rectangle(image, (12, 12), (285, 142), (20, 20, 20), -1)
        cv2.rectangle(image, (12, 12), (285, 142), color, 2)
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
        cv2.putText(
            image,
            f"operation confidence: {frame.operation_confidence:.2f}",
            (25, 121),
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
