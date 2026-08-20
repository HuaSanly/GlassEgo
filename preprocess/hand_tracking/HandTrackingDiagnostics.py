import json
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from data_types.HandDiagnosticTypes import (
    HandCandidateDiagnostic,
    HandDiagnosticsResult,
    HandFrameDiagnostic,
)


STAGE_NAMES = (
    "detector",
    "hamer",
    "confidence_filtered",
    "interpolated",
    "final",
)
SIDES = ("left", "right")


class HandTrackingDiagnostics:
    """记录原始手部候选，并评估各阶段的检测表现。"""

    def __init__(self, unit_dir, cfg, detector_backend: str, frame_count: int):
        self.unit_dir = Path(unit_dir)
        self.cfg = cfg
        self.detector_backend = detector_backend
        self.frame_count = int(frame_count)
        self.output_dir = self.unit_dir / "preprocess" / "hand_diagnostics"
        self.result = HandDiagnosticsResult()

    def start_frame(self, frame_idx: int, timestamp_ns: int) -> HandFrameDiagnostic:
        frame = HandFrameDiagnostic(
            frame_idx=int(frame_idx),
            timestamp_ns=int(timestamp_ns),
            detector_backend=self.detector_backend,
        )
        self.result.frames.append(frame)
        return frame

    @staticmethod
    def add_candidate(
        frame: HandFrameDiagnostic,
        detection: dict,
        image_shape: tuple[int, int],
    ) -> HandCandidateDiagnostic:
        height, width = image_shape
        bbox = np.asarray(detection["bbox"], dtype=np.float64).reshape(4)
        bbox_width = max(0.0, float(bbox[2] - bbox[0]))
        bbox_height = max(0.0, float(bbox[3] - bbox[1]))
        bbox_area = bbox_width * bbox_height
        candidate = HandCandidateDiagnostic(
            candidate_idx=len(frame.candidates),
            side=str(detection["label"]).lower(),
            bbox=bbox.tolist(),
            bbox_width_px=bbox_width,
            bbox_height_px=bbox_height,
            bbox_area_px2=bbox_area,
            bbox_area_ratio=bbox_area / max(float(width * height), 1.0),
            detector_confidence=float(detection["confidence"]),
            vitpose_valid_keypoints_count=detection.get(
                "vitpose_valid_keypoints_count"
            ),
            whole_image_fallback=bool(
                detection.get("whole_image_fallback", False)
            ),
        )
        frame.candidates.append(candidate)
        return candidate

    @staticmethod
    def capture_detector_stage(
        frame: HandFrameDiagnostic,
        whole_image_fallback: bool,
    ) -> None:
        frame.whole_image_fallback = bool(whole_image_fallback)
        frame.detector_candidate_count = len(frame.candidates)
        left = any(candidate.side == "left" for candidate in frame.candidates)
        right = any(candidate.side == "right" for candidate in frame.candidates)
        frame.stages["detector"] = {
            "left": left,
            "right": right,
            "any": left or right,
        }

    def capture_hands_stage(self, stage: str, hands) -> None:
        if stage not in STAGE_NAMES[1:]:
            raise ValueError(f"Unsupported hand diagnostic stage: {stage}")
        if len(hands.hands) != len(self.result.frames):
            raise ValueError(
                "Hand diagnostics are not aligned with the hand sequence: "
                f"diagnostics={len(self.result.frames)}, hands={len(hands.hands)}"
            )
        for frame, hand_data in zip(self.result.frames, hands.hands):
            left = hand_data.hand_l is not None
            right = hand_data.hand_r is not None
            frame.stages[stage] = {
                "left": left,
                "right": right,
                "any": left or right,
            }

    def save(
        self,
        cam,
        hands,
        grasp_threshold: float,
        opt_velocity_limit: float,
    ) -> dict:
        if len(self.result.frames) != self.frame_count:
            raise ValueError(
                "Incomplete hand diagnostics: "
                f"expected={self.frame_count}, actual={len(self.result.frames)}"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_path = self.output_dir / "diagnostics.json"
        self.result.save_json(diagnostics_path)

        ground_truth = self._load_ground_truth()
        report = self._build_report(ground_truth)
        outputs = {
            "diagnostics_json": {"status": "completed"},
            "confidence_plot": {"status": "disabled"},
            "debug_video": {"status": "disabled"},
        }
        if bool(self.cfg.export_confidence_plot):
            try:
                self._save_confidence_plot(ground_truth)
                outputs["confidence_plot"] = {"status": "completed"}
            except Exception as error:
                outputs["confidence_plot"] = {
                    "status": "failed",
                    "error": str(error),
                }
                print(f"[HaMeR Diagnostics] Confidence plot failed: {error}")
        if bool(self.cfg.export_video):
            try:
                self._save_debug_video(
                    cam,
                    hands,
                    grasp_threshold,
                    opt_velocity_limit,
                )
                outputs["debug_video"] = {"status": "completed"}
            except Exception as error:
                outputs["debug_video"] = {
                    "status": "failed",
                    "error": str(error),
                }
                print(f"[HaMeR Diagnostics] Debug video failed: {error}")
        report["outputs"] = outputs
        if any(output["status"] == "failed" for output in outputs.values()):
            report["status"] = "completed_with_warnings"
        report_path = self.output_dir / "report.json"
        self._write_json_atomic(report_path, report)
        return report

    @staticmethod
    def _write_json_atomic(path: Path, document: dict) -> None:
        temporary_path = path.with_name(f"{path.name}.tmp")
        try:
            with temporary_path.open("w", encoding="utf-8") as stream:
                json.dump(document, stream, indent=2)
            temporary_path.replace(path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _load_ground_truth(self) -> dict:
        path = self.unit_dir / str(self.cfg.ground_truth_filename)
        if not path.is_file():
            return {
                "status": "missing",
                "path": str(path.relative_to(self.unit_dir)),
                "intervals": None,
                "presence": None,
            }

        document = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        if not isinstance(document, dict) or document.get("schema_version") != 1:
            raise ValueError(
                f"{path} must be a mapping with schema_version: 1"
            )

        intervals = {}
        presence = {}
        for side in SIDES:
            if side not in document or not isinstance(document[side], list):
                raise ValueError(f"{path}: {side} must be a list of frame intervals")
            normalized = self._normalize_intervals(document[side], path, side)
            side_presence = np.zeros(self.frame_count, dtype=bool)
            for start, end in normalized:
                side_presence[start:end + 1] = True
            intervals[side] = normalized
            presence[side] = side_presence
        presence["any"] = presence["left"] | presence["right"]
        return {
            "status": "available",
            "path": str(path.relative_to(self.unit_dir)),
            "intervals": intervals,
            "presence": presence,
        }

    def _normalize_intervals(self, values, path: Path, side: str) -> list[list[int]]:
        intervals = []
        for index, value in enumerate(values):
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 2
                or isinstance(value[0], bool)
                or isinstance(value[1], bool)
                or not isinstance(value[0], int)
                or not isinstance(value[1], int)
            ):
                raise ValueError(
                    f"{path}: {side}[{index}] must be [start_frame, end_frame]"
                )
            start, end = value
            if start < 0 or end < start or end >= self.frame_count:
                raise ValueError(
                    f"{path}: invalid {side} interval [{start}, {end}] "
                    f"for {self.frame_count} frames"
                )
            intervals.append([start, end])

        merged = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return merged

    def _build_report(self, ground_truth: dict) -> dict:
        all_candidates = [
            candidate
            for frame in self.result.frames
            for candidate in frame.candidates
        ]
        fallback_frames = sum(
            frame.whole_image_fallback for frame in self.result.frames
        )
        valid_keypoints = [
            candidate.vitpose_valid_keypoints_count
            for candidate in all_candidates
            if candidate.vitpose_valid_keypoints_count is not None
        ]

        stage_presence = {
            stage: {
                side: sum(
                    frame.stages.get(stage, {}).get(side, False)
                    for frame in self.result.frames
                )
                for side in (*SIDES, "any")
            }
            for stage in STAGE_NAMES
        }
        metrics = None
        if ground_truth["status"] == "available":
            metrics = {
                stage: {
                    side: self._stage_metrics(
                        stage,
                        side,
                        ground_truth["presence"][side],
                    )
                    for side in (*SIDES, "any")
                }
                for stage in STAGE_NAMES
            }

        return {
            "status": "completed",
            "schema_version": 2,
            "frame_count": self.frame_count,
            "detector_backend": self.detector_backend,
            "stage_definitions": {
                "detector": "detector candidates after internal filtering",
                "hamer": "best valid HaMeR reconstruction for each side",
                "confidence_filtered": "presence after final confidence filtering",
                "interpolated": "presence after short-gap interpolation",
                "final": "presence after short-track suppression",
            },
            "confidence_label_definition": (
                "A candidate is true_hand when its declared side is labeled "
                "present in that frame; interval labels do not evaluate bbox "
                "localization accuracy."
            ),
            "ground_truth": {
                "status": ground_truth["status"],
                "path": ground_truth["path"],
                "intervals": ground_truth["intervals"],
            },
            "summary": {
                "candidate_count": len(all_candidates),
                "hamer_success_count": sum(
                    candidate.hamer_succeeded for candidate in all_candidates
                ),
                "valid_reconstruction_count": sum(
                    candidate.reconstruction_valid for candidate in all_candidates
                ),
                "selected_candidate_count": sum(
                    candidate.selected for candidate in all_candidates
                ),
                "whole_image_fallback_frames": fallback_frames,
                "whole_image_fallback_rate": fallback_frames / self.frame_count,
                "depth_recovery_attempt_count": sum(
                    candidate.depth_recovery_attempted
                    for candidate in all_candidates
                ),
                "depth_recovery_success_count": sum(
                    candidate.depth_recovered for candidate in all_candidates
                ),
                "stage_presence_frames": stage_presence,
                "detector_candidates_per_frame": self._describe(
                    [frame.detector_candidate_count for frame in self.result.frames]
                ),
                "bbox_width_px": self._describe(
                    [candidate.bbox_width_px for candidate in all_candidates]
                ),
                "bbox_height_px": self._describe(
                    [candidate.bbox_height_px for candidate in all_candidates]
                ),
                "bbox_area_ratio": self._describe(
                    [candidate.bbox_area_ratio for candidate in all_candidates]
                ),
                "vitpose_valid_keypoints_count": self._describe(valid_keypoints),
                "final_confidence": self._describe(
                    [candidate.final_confidence for candidate in all_candidates
                     if candidate.final_confidence is not None]
                ),
                "geometry_confidence": self._describe(
                    [candidate.geometry_confidence for candidate in all_candidates
                     if candidate.geometry_confidence is not None]
                ),
                "iou_confidence": self._describe(
                    [candidate.iou_confidence for candidate in all_candidates
                     if candidate.iou_confidence is not None]
                ),
                "position_confidence": self._describe(
                    [candidate.position_confidence for candidate in all_candidates
                     if candidate.position_confidence is not None]
                ),
                "rotation_confidence": self._describe(
                    [candidate.rotation_confidence for candidate in all_candidates
                     if candidate.rotation_confidence is not None]
                ),
            },
            "metrics": metrics,
            "confidence_distributions": self._confidence_distributions(
                ground_truth
            ),
        }

    def _stage_metrics(
        self,
        stage: str,
        side: str,
        truth: np.ndarray,
    ) -> dict:
        predicted = np.asarray(
            [
                frame.stages.get(stage, {}).get(side, False)
                for frame in self.result.frames
            ],
            dtype=bool,
        )
        true_positive = int(np.sum(predicted & truth))
        false_negative = int(np.sum(~predicted & truth))
        false_positive = int(np.sum(predicted & ~truth))
        true_negative = int(np.sum(~predicted & ~truth))
        recall_denominator = true_positive + false_negative
        false_positive_denominator = false_positive + true_negative
        runs = self._false_positive_runs(
            predicted & ~truth,
            [frame.frame_idx for frame in self.result.frames],
        )
        lengths = [run["length_frames"] for run in runs]
        return {
            "tp": true_positive,
            "fn": false_negative,
            "fp": false_positive,
            "tn": true_negative,
            "recall": (
                true_positive / recall_denominator
                if recall_denominator
                else None
            ),
            "false_positive_rate": (
                false_positive / false_positive_denominator
                if false_positive_denominator
                else None
            ),
            "false_positive_runs": runs,
            "false_positive_run_lengths": self._describe(lengths),
        }

    @staticmethod
    def _false_positive_runs(
        mask: np.ndarray,
        frame_indices: list[int],
    ) -> list[dict]:
        runs = []
        start = None
        previous = None
        count = 0
        for frame_idx, is_false_positive in zip(frame_indices, mask):
            if start is not None and frame_idx != previous + 1:
                runs.append(
                    {
                        "start_frame": start,
                        "end_frame": previous,
                        "length_frames": count,
                    }
                )
                start = None
                count = 0
            if is_false_positive and start is None:
                start = frame_idx
                count = 1
            elif is_false_positive:
                count += 1
            elif not is_false_positive and start is not None:
                runs.append(
                    {
                        "start_frame": start,
                        "end_frame": previous,
                        "length_frames": count,
                    }
                )
                start = None
                count = 0
            previous = frame_idx
        if start is not None:
            runs.append(
                {
                    "start_frame": start,
                    "end_frame": previous,
                    "length_frames": count,
                }
            )
        return runs

    def _confidence_distributions(self, ground_truth: dict) -> dict:
        fields = (
            "detector_confidence",
            "hamer_confidence",
            "final_confidence",
            "geometry_confidence",
            "iou_confidence",
            "position_confidence",
            "rotation_confidence",
        )
        output = {}
        for field_name in fields:
            groups = {"all": [], "true_hand": [], "false_hand": []}
            for frame in self.result.frames:
                for candidate in frame.candidates:
                    value = getattr(candidate, field_name)
                    if value is None:
                        continue
                    if field_name == "final_confidence" and not candidate.reconstruction_valid:
                        continue
                    groups["all"].append(value)
                    if ground_truth["status"] == "available":
                        is_true = bool(
                            ground_truth["presence"][candidate.side][frame.frame_idx]
                        )
                        groups["true_hand" if is_true else "false_hand"].append(value)
            output[field_name] = {
                "all": self._describe(groups["all"]),
                "true_hand": (
                    self._describe(groups["true_hand"])
                    if ground_truth["status"] == "available"
                    else None
                ),
                "false_hand": (
                    self._describe(groups["false_hand"])
                    if ground_truth["status"] == "available"
                    else None
                ),
            }
        return output

    @staticmethod
    def _describe(values) -> dict:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "p10": None,
                "p90": None,
                "p95": None,
            }
        return {
            "count": int(values.size),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
        }

    def _save_confidence_plot(self, ground_truth: dict) -> None:
        import matplotlib.pyplot as plt

        fields = (
            ("detector_confidence", "Detector confidence"),
            ("hamer_confidence", "HaMeR confidence"),
            ("final_confidence", "Final confidence"),
            ("geometry_confidence", "Geometry confidence"),
            ("iou_confidence", "IoU confidence"),
            ("position_confidence", "Position confidence"),
            ("rotation_confidence", "Rotation confidence"),
        )
        figure, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.reshape(-1)
        try:
            for axis, (field_name, title) in zip(axes, fields):
                groups = {"all": [], "true": [], "false": []}
                for frame in self.result.frames:
                    for candidate in frame.candidates:
                        value = getattr(candidate, field_name)
                        if value is None:
                            continue
                        if (
                            field_name == "final_confidence"
                            and not candidate.reconstruction_valid
                        ):
                            continue
                        groups["all"].append(value)
                        if ground_truth["status"] == "available":
                            key = (
                                "true"
                                if ground_truth["presence"][candidate.side][
                                    frame.frame_idx
                                ]
                                else "false"
                            )
                            groups[key].append(value)
                if ground_truth["status"] == "available":
                    if groups["true"]:
                        axis.hist(
                            groups["true"],
                            bins=20,
                            alpha=0.65,
                            label="true hand",
                        )
                    if groups["false"]:
                        axis.hist(
                            groups["false"],
                            bins=20,
                            alpha=0.65,
                            label="false hand",
                        )
                    if groups["true"] or groups["false"]:
                        axis.legend()
                elif groups["all"]:
                    axis.hist(
                        groups["all"],
                        bins=20,
                        alpha=0.8,
                        label="unlabeled",
                    )
                    axis.legend()
                axis.set_title(title)
                axis.set_xlabel("confidence")
                axis.set_ylabel("candidate count")
                axis.grid(alpha=0.2)
            figure.tight_layout()
            save_path = self.output_dir / "confidence_distributions.png"
            figure.savefig(save_path, dpi=160)
        finally:
            plt.close(figure)

    def _save_debug_video(
        self,
        cam,
        hands,
        grasp_threshold: float,
        opt_velocity_limit: float,
    ) -> None:
        from hand_tracking.HandsOps import HandsOps

        if len(cam.cam) != len(self.result.frames) or len(hands.hands) != len(cam.cam):
            raise ValueError("Diagnostic video inputs are not frame-aligned")
        first = cam.cam[0].img
        if first is None:
            raise ValueError("The first camera frame is missing")
        height, width = first.shape[:2]
        vis_dir = self.output_dir / "vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        save_path = vis_dir / "hamer_hands_diagnostics.mp4"
        writer = cv2.VideoWriter(
            str(save_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(cam.fps),
            (width * 2, height),
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Unable to open VideoWriter: {save_path}")

        try:
            for frame_diag, cam_data, hand_data in tqdm(
                zip(self.result.frames, cam.cam, hands.hands),
                total=len(cam.cam),
                desc="Hand diagnostics video",
                mininterval=1.0,
            ):
                if cam_data.img is None:
                    raise RuntimeError(
                        f"Missing diagnostic video frame: {cam_data.idx}"
                    )
                image_bgr = cv2.cvtColor(cam_data.img, cv2.COLOR_RGB2BGR)
                raw = self._draw_raw_candidates(image_bgr.copy(), frame_diag)
                final = HandsOps.draw_aria_hands_skeleton(
                    image_bgr.copy(),
                    hand_data,
                    cam_data.k,
                    getattr(cam_data, "d", np.zeros(8)),
                    cam_data.c2w,
                    grasp_threshold=grasp_threshold,
                )
                final = HandsOps.draw_aria_hands_panel(
                    final,
                    frame_diag.frame_idx,
                    hand_data,
                    opt_v_limit=opt_velocity_limit,
                )
                self._draw_title(raw, "RAW DETECTOR / HAMER")
                self._draw_title(final, "FINAL POSTPROCESS")
                writer.write(np.hstack([raw, final]))
        finally:
            writer.release()
        if not save_path.is_file() or save_path.stat().st_size == 0:
            raise RuntimeError(f"Diagnostic video was not written: {save_path}")

    @staticmethod
    def _draw_title(image: np.ndarray, title: str) -> None:
        cv2.rectangle(image, (0, 0), (image.shape[1], 26), (20, 20, 20), -1)
        cv2.putText(
            image,
            title,
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_raw_candidates(
        image: np.ndarray,
        frame: HandFrameDiagnostic,
    ) -> np.ndarray:
        colors = {"left": (255, 160, 40), "right": (40, 180, 255)}
        for candidate in frame.candidates:
            color = colors.get(candidate.side, (180, 180, 180))
            x1, y1, x2, y2 = [int(round(value)) for value in candidate.bbox]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            hamer = (
                f"{candidate.hamer_confidence:.2f}"
                if candidate.hamer_confidence is not None
                else "-"
            )
            final = (
                f"{candidate.final_confidence:.2f}"
                if candidate.final_confidence is not None
                else "-"
            )
            valid = (
                str(candidate.vitpose_valid_keypoints_count)
                if candidate.vitpose_valid_keypoints_count is not None
                else "-"
            )
            flags = ""
            if candidate.whole_image_fallback:
                flags += " FB"
            if candidate.depth_recovered:
                flags += " DR"
            if candidate.selected:
                flags += " SEL"
            lines = (
                f"{candidate.side[0].upper()}#{candidate.candidate_idx} "
                f"d={candidate.detector_confidence:.2f} h={hamer} f={final}",
                f"kp={valid} box={candidate.bbox_width_px:.0f}x"
                f"{candidate.bbox_height_px:.0f}{flags}",
            )
            text_y = max(42, y1 - 22)
            for offset, text in enumerate(lines):
                cv2.putText(
                    image,
                    text,
                    (max(2, x1), text_y + offset * 17),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        stage_text = "  ".join(
            f"{stage[:4]}:"
            f"{'L' if frame.stages.get(stage, {}).get('left') else '-'}"
            f"{'R' if frame.stages.get(stage, {}).get('right') else '-'}"
            for stage in STAGE_NAMES
        )
        if frame.whole_image_fallback:
            stage_text = "WHOLE-IMAGE FALLBACK  " + stage_text
        cv2.rectangle(
            image,
            (0, image.shape[0] - 24),
            (image.shape[1], image.shape[0]),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            image,
            stage_text,
            (6, image.shape[0] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        return image
