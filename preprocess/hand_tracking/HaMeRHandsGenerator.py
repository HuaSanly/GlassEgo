"""Generate hand data using the HaMeR/OpenPose 21-keypoint convention.

CMC:腕掌关节
MCP:掌指关节
IP:指间关节
MCP:掌指关节
PIP:近端指间关节
DIP:远端指间关节
Tip:指尖。

Keypoint order:
    0 Wrist
    1-4 Thumb: CMC, MCP, IP, Tip
    5-8 Index: MCP, PIP, DIP, Tip
    9-12 Middle: MCP, PIP, DIP, Tip
    13-16 Ring: MCP, PIP, DIP, Tip
    17-20 Pinky: MCP, PIP, DIP, Tip
"""
import os
import gc
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from scipy.spatial.transform import Rotation as R
from typing import Optional
try:
    from data_types.HandsTypes import (
        Hands, HandsData, HandData, HandsJointAngles, MidpointFrameBuilder
    )
    from data_types.CamTypes import Cam
except ModuleNotFoundError:
    from preprocess.data_types.HandsTypes import (
        Hands, HandsData, HandData, HandsJointAngles, MidpointFrameBuilder
    )
    from preprocess.data_types.CamTypes import Cam
from preprocess.hand_tracking.HandsTrajectoryOptimizer import HandsTrajectoryOptimizer

class HaMeRHandsGenerator:

    def __init__(self, unit_dir, cfg, output_cfg, cam: Cam):
        # Model backends are optional until a hand stage is actually requested.
        # This keeps VIO, phase, and cached-hand workflows importable without
        # installing every detector implementation.
        try:
            from hand_tracking.HaMeRModel import HaMeRModel
            from hand_tracking.HandsOps import HandsOps
            from hand_tracking.HandTrackingDiagnostics import HandTrackingDiagnostics
        except ModuleNotFoundError:
            from preprocess.hand_tracking.HaMeRModel import HaMeRModel
            from preprocess.hand_tracking.HandsOps import HandsOps
            from preprocess.hand_tracking.HandTrackingDiagnostics import HandTrackingDiagnostics
        self._hamer_model_type = HaMeRModel
        self._hands_ops = HandsOps
        self._diagnostics_type = HandTrackingDiagnostics
        self.unit_dir = Path(unit_dir)
        if not self.unit_dir.is_dir():
            raise NotADirectoryError(f"Unit directory not found: {self.unit_dir}")
        self.preprocess_dir = self.unit_dir / "preprocess"
        self.cfg = cfg
        self.output_cfg = output_cfg
        self.cam = cam
        self._closed = False
        self._validate_scoring_config()

        backend = str(self.cfg.detector.backend).lower()
        device = str(self.cfg.detector.device)
        if backend not in {"auto", "vitpose", "mediapipe"}:
            raise ValueError(
                "hand_tracking.detector.backend must be auto, vitpose, or mediapipe"
            )

        if backend == "mediapipe":
            self.detector = self._load_mediapipe()(self.cfg.detector.mediapipe)
            self._detector_name = "MediaPipe"
        else:
            try:
                self.detector = self._load_vitpose()(
                    self.cfg.detector.vitpose,
                    device=device,
                )
                self._detector_name = "VitPose"
            except Exception as e:
                if backend == "vitpose":
                    raise
                print(
                    f"[HaMeR] ViTPose not available ({e}), "
                    "falling back to MediaPipe detector"
                )
                self.detector = self._load_mediapipe()(
                    self.cfg.detector.mediapipe
                )
                self._detector_name = "MediaPipe"

        if not bool(self.cfg.hamer.enabled):
            raise RuntimeError(
                "hand_tracking.hamer.enabled must be true; "
                "the current generator requires HaMeR 3D reconstruction"
            )
        self.hamer_model = self._hamer_model_type(
            device=str(self.cfg.hamer.device),
            hamer_hf_repo=str(self.cfg.hamer.hamer_hf_repo),
            mano_hf_repo=str(self.cfg.hamer.mano_hf_repo),
        )

        if not self.hamer_model.is_available:
            reason = self.hamer_model.initialization_error
            raise RuntimeError(
                "HaMeR model is unavailable; refusing to generate an all-empty "
                "hand trajectory. Check the local model assets or network access."
                + (f" Cause: {reason}" if reason is not None else "")
            )

        self.diagnostics = None
        if bool(self.cfg.diagnostics.enabled):
            self.diagnostics = self._diagnostics_type(
                unit_dir=self.unit_dir,
                cfg=self.cfg.diagnostics,
                detector_backend=self._detector_name,
                frame_count=len(self.cam.cam),
            )

        self.mid_frame_builder = MidpointFrameBuilder()
        self._score_state = {"right": None, "left": None}

    @staticmethod
    def _load_mediapipe():
        try:
            from hand_tracking.MediaPipeHandDetector import MediaPipeHandDetector
        except ModuleNotFoundError:
            from preprocess.hand_tracking.MediaPipeHandDetector import MediaPipeHandDetector
        return MediaPipeHandDetector

    @staticmethod
    def _load_vitpose():
        try:
            from hand_tracking.VitPoseHandDetector import VitPoseHandDetector
        except ModuleNotFoundError:
            from preprocess.hand_tracking.VitPoseHandDetector import VitPoseHandDetector
        return VitPoseHandDetector

    def cleanup(self) -> None:
        """释放单个视频手部处理期间持有的模型和帧缓存。"""
        if self._closed:
            return
        self._closed = True

        detector = getattr(self, "detector", None)
        self.detector = None
        if detector is not None and hasattr(detector, "cleanup"):
            detector.cleanup()

        hamer_model = getattr(self, "hamer_model", None)
        self.hamer_model = None
        if hamer_model is not None:
            hamer_model.cleanup()

        self.mid_frame_builder = None
        self._score_state = {"right": None, "left": None}
        self.diagnostics = None

        cam = self.cam
        self.cam = None
        if cam is not None:
            cam.cam.clear()
            cam.tss.clear()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _clip_score(value: float) -> float:
        return float(np.clip(float(value), 0.0, 1.0))

    def _score_value(self, path: str, default: float) -> float:
        value = self.cfg
        try:
            for part in path.split("."):
                value = getattr(value, part)
            return float(value)
        except (AttributeError, TypeError, ValueError):
            return float(default)

    def _validate_scoring_config(self) -> None:
        weights = (
            self._score_value("scoring.final.detector_weight", 0.30),
            self._score_value("scoring.final.geometry_weight", 0.25),
            self._score_value("scoring.final.iou_weight", 0.15),
            self._score_value("scoring.final.position_weight", 0.15),
            self._score_value("scoring.final.rotation_weight", 0.15),
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("hand_tracking.scoring.final weights must be non-negative")
        if not np.isclose(sum(weights), 1.0, atol=1e-6):
            raise ValueError("hand_tracking.scoring.final weights must sum to 1.0")

    @staticmethod
    def _bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
        first = np.asarray(first, dtype=np.float64).reshape(4)
        second = np.asarray(second, dtype=np.float64).reshape(4)
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        union = area_first + area_second - intersection
        return float(intersection / union) if union > 1e-9 else 0.0

    def _geometry_metrics(
        self,
        keypoints_3d: np.ndarray,
        detector_keypoints_2d: np.ndarray,
        k: np.ndarray,
        d: np.ndarray,
    ) -> tuple[float, float, float]:
        points_3d = np.asarray(keypoints_3d, dtype=np.float64)
        points_2d = np.asarray(detector_keypoints_2d, dtype=np.float64)
        if points_3d.shape != (21, 3) or points_2d.shape != (21, 2):
            return 0.0, float("inf"), 0.0
        valid = np.isfinite(points_3d).all(axis=1) & np.isfinite(points_2d).all(axis=1)
        positive_depth = valid & (points_3d[:, 2] > 1e-5)
        if not np.any(positive_depth):
            return 0.0, float("inf"), 0.0
        try:
            import cv2

            distortion = np.asarray(d if d is not None else np.zeros(8), dtype=np.float64).reshape(-1, 1)
            projected, _ = cv2.projectPoints(
                points_3d,
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                np.asarray(k, dtype=np.float64),
                distortion,
            )
            projected = projected.reshape(-1, 2)
            valid &= np.isfinite(projected).all(axis=1)
            if not np.any(valid):
                return 0.0, float("inf"), 0.0
            errors = np.linalg.norm(projected[valid] - points_2d[valid], axis=1)
            scale = max(self._score_value("scoring.geometry_error_scale_px", 30.0), 1e-6)
            reprojection_score = 1.0 - float(np.clip(np.median(errors) / scale, 0.0, 1.0))
            depth_score = float(np.mean(positive_depth[valid]))
            geometry_confidence = self._clip_score(
                0.7 * reprojection_score + 0.3 * depth_score
            )
            return geometry_confidence, float(np.median(errors)), depth_score
        except Exception:
            return 0.0, float("inf"), 0.0

    def _history_for_frame(self, side: str, frame_idx: int, timestamp_ns: int):
        state = self._score_state.get(side)
        if state is None:
            return None
        max_gap = int(self._score_value("scoring.temporal_max_gap_frames", 3))
        if frame_idx <= state["frame_idx"] or frame_idx - state["frame_idx"] > max_gap:
            return None
        if timestamp_ns <= state["timestamp_ns"]:
            return None
        if not np.isfinite(state["wrist_world"]).all():
            return None
        rotation = np.asarray(state["rotation_world"], dtype=np.float64)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            return None
        return state

    def _world_wrist_pose(self, kpts_cam: np.ndarray, c2w: np.ndarray, k: np.ndarray, h: int, w: int):
        hand = self._build_hand_data(
            kpts_cam,
            np.zeros((21, 2), dtype=np.float32),
            0.0,
            c2w,
            k,
            h,
            w,
            is_right=True,
        )
        if hand.wrist_pose is None:
            return None, None
        wrist_pose = np.asarray(hand.wrist_pose, dtype=np.float64)
        rotation_cam = wrist_pose[:3, :3]
        position_cam = wrist_pose[:3, 3]
        c2w = np.asarray(c2w, dtype=np.float64)
        rotation_world = c2w[:3, :3] @ rotation_cam
        position_world = c2w[:3, :3] @ position_cam + c2w[:3, 3]
        if not np.isfinite(position_world).all() or not np.isfinite(rotation_world).all():
            return None, None
        if not np.allclose(rotation_world.T @ rotation_world, np.eye(3), atol=1e-3):
            return None, None
        if not np.isclose(np.linalg.det(rotation_world), 1.0, atol=1e-3):
            return None, None
        return position_world, rotation_world

    def _candidate_temporal_features(
        self,
        side: str,
        detection: dict,
        frame_idx: int,
        timestamp_ns: int,
        position_world: Optional[np.ndarray],
        rotation_world: Optional[np.ndarray],
    ) -> dict:
        state = self._history_for_frame(side, frame_idx, timestamp_ns)
        features = {
            "history_available": state is not None,
            "iou_confidence": None,
            "position_confidence": None,
            "position_speed_mps": None,
            "rotation_confidence": None,
            "angular_speed_rad_s": None,
        }
        if state is None or position_world is None or rotation_world is None:
            features["history_available"] = False
            return features
        delta_t = (timestamp_ns - state["timestamp_ns"]) / 1e9
        if delta_t <= 0.0:
            features["history_available"] = False
            return features
        features["iou_confidence"] = self._bbox_iou(detection["bbox"], state["bbox"])
        speed = float(np.linalg.norm(position_world - state["wrist_world"]) / delta_t)
        max_speed = max(self._score_value("scoring.max_hand_speed_mps", 2.5), 1e-6)
        features["position_speed_mps"] = speed
        features["position_confidence"] = self._clip_score(1.0 - speed / max_speed)
        relative_rotation = state["rotation_world"].T @ rotation_world
        cosine = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
        angle = float(np.arccos(cosine))
        angular_speed = angle / delta_t
        max_angular_speed = max(
            self._score_value("scoring.max_hand_angular_speed_rad_s", 12.0),
            1e-6,
        )
        features["angular_speed_rad_s"] = angular_speed
        features["rotation_confidence"] = self._clip_score(
            1.0 - angular_speed / max_angular_speed
        )
        return features

    def _final_confidence(self, detector: float, geometry: float, temporal: dict) -> float:
        weights = {
            "detector": self._score_value("scoring.final.detector_weight", 0.30),
            "geometry": self._score_value("scoring.final.geometry_weight", 0.25),
            "iou": self._score_value("scoring.final.iou_weight", 0.15),
            "position": self._score_value("scoring.final.position_weight", 0.15),
            "rotation": self._score_value("scoring.final.rotation_weight", 0.15),
        }
        if temporal["history_available"]:
            return self._clip_score(
                weights["detector"] * detector
                + weights["geometry"] * geometry
                + weights["iou"] * temporal["iou_confidence"]
                + weights["position"] * temporal["position_confidence"]
                + weights["rotation"] * temporal["rotation_confidence"]
            )
        base_weight = weights["detector"] + weights["geometry"]
        return self._clip_score(
            (weights["detector"] * detector + weights["geometry"] * geometry)
            / max(base_weight, 1e-6)
        )

    def _update_score_state(self, side: str, candidate: Optional[dict], frame_idx: int) -> None:
        if candidate is None:
            state = self._score_state.get(side)
            if state is not None and frame_idx - state["frame_idx"] > int(
                self._score_value("scoring.temporal_max_gap_frames", 3)
            ):
                self._score_state[side] = None
            return
        self._score_state[side] = {
            "frame_idx": int(frame_idx),
            "bbox": np.asarray(candidate["detection"]["bbox"], dtype=np.float32).copy(),
            "timestamp_ns": int(candidate["timestamp_ns"]),
            "wrist_world": np.asarray(candidate["wrist_world"], dtype=np.float32).copy(),
            "rotation_world": np.asarray(candidate["rotation_world"], dtype=np.float32).copy(),
        }

    def get_hands_data(self)->Hands:
        """完整对外pipeline"""
        hands = Hands(
            mps_path=str(self.unit_dir),
            camera_frame=self.cam.camera_frame,
            world_frame=self.cam.world_frame,
            world_origin=self.cam.world_origin,
            initial_heading=self.cam.initial_heading,
        )
        dt = 1.0 / self.cam.fps

        for i,cam_data in enumerate(tqdm(self.cam.cam, desc="Hands", mininterval=1.0)):
            #图像获取
            img = cam_data.img  #rgb
            h_img, w_img = img.shape[:2]
            k = cam_data.k
            c2w = cam_data.c2w

            #手部检测
            if self._detector_name == "MediaPipe":
                timestamp_ms = int(i * 1000.0 / self.cam.fps)
                detections = self.detector.detect(img, timestamp_ms)
            else:
                detections = self.detector.detect(img)

            frame_diagnostic = None
            if self.diagnostics is not None:
                frame_diagnostic = self.diagnostics.start_frame(
                    cam_data.idx,
                    cam_data.ts,
                )

            hand_r = None
            hand_l = None

            fx = k[0, 0]
            fy = k[1, 1]
            focal = (fx + fy) / 2.0  #焦距

            candidates_by_side = {"right": [], "left": []}
            for hand in detections:
                label = hand['label']
                side = "right" if label == "Right" else "left"
                det_confidence = self._clip_score(hand['confidence'])
                candidate_diagnostic = None
                if frame_diagnostic is not None:
                    candidate_diagnostic = self.diagnostics.add_candidate(
                        frame_diagnostic,
                        hand,
                        (h_img, w_img),
                    )
                # 第 2 阶段：HaMeR 从裁剪中恢复 3D 网格
                hamer_result = self.hamer_model.predict_from_crop(
                    img, hand['bbox'],
                    is_right=hand['is_right_int'],
                    focal_length=focal,
                    camera_matrix=k,
                    distortion=cam_data.d,
                )

                if hamer_result is not None:
                    kpts_cam = np.asarray(hamer_result['joints_3d'], dtype=np.float32)
                    kpts_2d = np.asarray(hamer_result['joints_2d'], dtype=np.float32)
                    if candidate_diagnostic is not None:
                        candidate_diagnostic.hamer_succeeded = True

                    wrist_z = kpts_cam[0,2]  # (4, 4) 相机空间
                    if (
                        wrist_z < float(self.cfg.depth_recovery.wrist_min_z_m)
                        or wrist_z > float(self.cfg.depth_recovery.wrist_max_z_m)
                    ):
                        if candidate_diagnostic is not None:
                            candidate_diagnostic.depth_recovery_attempted = True
                        # HaMeR 的深度不可靠（可能是由于焦距
                        # 不匹配 — HaMeR 假设 f≈5000，但 Aria 的 f≈320）。
                        # 从像素大小+真实焦点重新估计绝对深度。
                        recovered = self._recover_absolute_3d_from_hamer(
                            kpts_cam,
                            hand['landmarks_2d'],
                            k,
                            h_img,
                            w_img,
                        )
                        if recovered is None:
                            if candidate_diagnostic is not None:
                                candidate_diagnostic.rejection_reason = (
                                    "depth_recovery_failed"
                                )
                            continue
                        kpts_cam = recovered
                        if candidate_diagnostic is not None:
                            candidate_diagnostic.depth_recovered = True
                    geometry_confidence, reprojection_error_px, positive_depth_ratio = self._geometry_metrics(
                        kpts_cam,
                        hand['landmarks_2d'],
                        k,
                        cam_data.d,
                    )
                    wrist_world, rotation_world = self._world_wrist_pose(
                        kpts_cam,
                        c2w,
                        k,
                        h_img,
                        w_img,
                    )
                    temporal = self._candidate_temporal_features(
                        side,
                        hand,
                        cam_data.idx,
                        cam_data.ts,
                        wrist_world,
                        rotation_world,
                    )
                    max_speed = max(
                        self._score_value("scoring.max_hand_speed_mps", 2.5),
                        1e-6,
                    )
                    if (
                        temporal["position_speed_mps"] is not None
                        and temporal["position_speed_mps"] > max_speed
                    ):
                        if candidate_diagnostic is not None:
                            candidate_diagnostic.rejection_reason = (
                                "position_speed_exceeded"
                            )
                        continue
                    final_confidence = self._final_confidence(
                        det_confidence,
                        geometry_confidence,
                        temporal,
                    )
                    candidate = {
                        "detection": hand,
                        "kpts_cam": kpts_cam,
                        "kpts_2d": kpts_2d,
                        "geometry_confidence": geometry_confidence,
                        "reprojection_error_px": reprojection_error_px,
                        "positive_depth_ratio": positive_depth_ratio,
                        "temporal": temporal,
                        "final_confidence": final_confidence,
                        "timestamp_ns": cam_data.ts,
                        "wrist_world": wrist_world,
                        "rotation_world": rotation_world,
                        "diagnostic": candidate_diagnostic,
                    }
                    if candidate_diagnostic is not None:
                        candidate_diagnostic.reconstruction_valid = True
                        candidate_diagnostic.geometry_confidence = geometry_confidence
                        candidate_diagnostic.final_confidence = final_confidence
                        candidate_diagnostic.iou_confidence = temporal["iou_confidence"]
                        candidate_diagnostic.position_confidence = temporal["position_confidence"]
                        candidate_diagnostic.rotation_confidence = temporal["rotation_confidence"]
                        candidate_diagnostic.reprojection_error_px = reprojection_error_px
                        candidate_diagnostic.positive_depth_ratio = positive_depth_ratio
                        candidate_diagnostic.position_speed_mps = temporal["position_speed_mps"]
                        candidate_diagnostic.angular_speed_rad_s = temporal["angular_speed_rad_s"]
                        candidate_diagnostic.history_available = temporal["history_available"]
                    candidates_by_side[side].append(candidate)
                elif candidate_diagnostic is not None:
                    candidate_diagnostic.rejection_reason = "hamer_failed"

            for side, candidates in candidates_by_side.items():
                if not candidates:
                    self._update_score_state(side, None, cam_data.idx)
                    continue
                selected = max(candidates, key=lambda item: item["final_confidence"])
                for candidate in candidates:
                    diagnostic = candidate["diagnostic"]
                    if diagnostic is None:
                        continue
                    if candidate is selected:
                        diagnostic.selected = True
                    else:
                        diagnostic.rejection_reason = "superseded_by_higher_final_confidence"
                h_data = self._build_hand_data(
                    selected["kpts_cam"],
                    selected["kpts_2d"],
                    selected["final_confidence"],
                    c2w,
                    k,
                    h_img,
                    w_img,
                    is_right=(side == "right"),
                )
                if side == "right":
                    hand_r = h_data
                else:
                    hand_l = h_data
                self._update_score_state(
                    side,
                    selected if selected["final_confidence"] >= float(self.cfg.postprocess.confidence_threshold) else None,
                    cam_data.idx,
                )

            if frame_diagnostic is not None:
                self.diagnostics.capture_detector_stage(
                    frame_diagnostic,
                    getattr(
                        self.detector,
                        "last_whole_image_fallback",
                        False,
                    ),
                )

            frame_data = HandsData(cam_data.idx, cam_data.ts, hand_r, hand_l)

            # 计算速度和中点坐标系
            self._assign_world_kinematics(frame_data, c2w)

            hands.hands.append(frame_data)
            hands.tss.append(cam_data.ts)
        # 第二阶段：数据清洗
        if self.diagnostics is not None:
            self.diagnostics.capture_hands_stage("hamer", hands)
        self._filter_by_confidence(
            hands,
            conf_th=float(self.cfg.postprocess.confidence_threshold),
        )
        if self.diagnostics is not None:
            self.diagnostics.capture_hands_stage(
                "confidence_filtered",
                hands,
            )
        if bool(self.cfg.postprocess.interpolation.enabled):
            HandsTrajectoryOptimizer._discard_incomplete_hands(hands)
            self._interpolate_hand_trajectories(
                hands,
                max_gap=int(
                    self.cfg.postprocess.interpolation.max_gap_frames
                ),
            )
        max_hand_speed = self._score_value("scoring.max_hand_speed_mps", 2.5)
        HandsTrajectoryOptimizer.remove_excessive_speed_hands(
            hands,
            max_hand_speed,
            "wrist_pose_raw_world",
        )
        if self.diagnostics is not None:
            self.diagnostics.capture_hands_stage("interpolated", hands)
        if bool(self.cfg.postprocess.short_track.enabled):
            self._suppress_short_hands(
                hands,
                min_frames=int(self.cfg.postprocess.short_track.min_frames),
            )
        if self.diagnostics is not None:
            self.diagnostics.capture_hands_stage("final", hands)
        self._smooth_grasp_detection(hands, size=self.cfg.grasp.smooth_window)
        # 第三阶段：运动学优化
        if bool(self.cfg.trajectory.enabled):
            optimizer = HandsTrajectoryOptimizer(self.cfg.trajectory)
            for _ in range(3):
                optimizer.run(hands)
                if not HandsTrajectoryOptimizer.remove_excessive_speed_hands(
                    hands,
                    max_hand_speed,
                    "wrist_pose_opt_world",
                ):
                    break
                if bool(self.cfg.postprocess.short_track.enabled):
                    self._suppress_short_hands(
                        hands,
                        min_frames=int(self.cfg.postprocess.short_track.min_frames),
                    )
        self._smooth_grasp_detection(hands, size=self.cfg.grasp.smooth_window)

        # 第四阶段：报告
        analysis_dir = self.preprocess_dir / "vis" / "hands"
        os.makedirs(analysis_dir, exist_ok=True)
        try:
            self._hands_ops.save_hands_analysis_plots_two(
                hands, str(analysis_dir), dt, self.cfg
            )
        except Exception as e:
            print(f"[HaMeR] Warning: analysis plots failed: {e}")
        self._hands_ops.print_summary_and_eval(hands)

        if self.diagnostics is not None:
            self.diagnostics.save(
                self.cam,
                hands,
                grasp_threshold=float(self.cfg.grasp.fallback_distance_m),
                opt_velocity_limit=float(
                    self.cfg.analysis.linear_velocity_limit_mps
                ),
            )

        if self.output_cfg.export_json:
            hands.save_hands_json(filename=self.output_cfg.json_filename)

        if self.output_cfg.export_video or self.output_cfg.export_gif:
            self._export_visualizations(hands)

        return hands

    def _export_visualizations(self, hands: Hands):
        import cv2
        from utils.utils_media import create_video_from_frames
        print("[HaMeR] Generating hand visualizations ...")
        vis_frames = []
        for idx in tqdm(range(len(self.cam.cam)), desc="HaMeR Vis"):
            cam_d = self.cam.cam[idx]
            if cam_d.img is not None:
                img = cv2.cvtColor(cam_d.img, cv2.COLOR_RGB2BGR)
            else:
                img_path = self.preprocess_dir / "temp_data" / f"{cam_d.idx:05d}" / "rgb.png"
                img = cv2.imread(str(img_path)) if img_path.is_file() else None
            if img is None:
                raise RuntimeError(f"Missing visualization frame: {cam_d.idx}")

            if idx < len(hands.hands):
                img = self._hands_ops.draw_aria_hands_skeleton(
                    img, hands.hands[idx],
                    cam_d.k, getattr(cam_d, 'd', np.zeros(8)), cam_d.c2w,
                    grasp_threshold=self.cfg.grasp.fallback_distance_m,
                )
                img = self._hands_ops.draw_aria_hands_panel(
                    img, idx, hands.hands[idx],
                    opt_v_limit=self.cfg.analysis.linear_velocity_limit_mps,
                )

            vis_frames.append(img)

        vis_dir = self.preprocess_dir / "vis"
        save_path = vis_dir / self.output_cfg.video_filename
        create_video_from_frames(
            vis_frames,
            str(save_path),
            self.cam.fps,
            export_gif=self.output_cfg.export_gif,
            ratio=self.output_cfg.gif_frame_ratio,
            export_video=self.output_cfg.export_video,
        )

    def _recover_absolute_3d_from_hamer(
            self,
            kpts_3d_hamer: np.ndarray,    # (21, 3) HaMeR 相机空间关节
            kpts_2d_mp: np.ndarray,        # (21, 2) MediaPipe 用于深度估计的 2D 检测
            k: np.ndarray,                 # (3, 3) 相机内参
            h_img: int, w_img: int,
        )->Optional[np.ndarray]:
            """"使用针孔模型重新估算HaMeR 3D关节的绝对深度"""
            wrist_2d = kpts_2d_mp[0]
            middle_mcp_2d = kpts_2d_mp[9]
    
            # 与 HaMeR 3D 关节的物理距离
            physical_dist = float(np.linalg.norm(kpts_3d_hamer[9] - kpts_3d_hamer[0]))
            if physical_dist < 0.01:
                physical_dist = float(
                    self.cfg.depth_recovery.wrist_middle_mcp_m
                )
    
            # 2D像素距离
            pixel_dist = float(np.linalg.norm(middle_mcp_2d - wrist_2d))
            if pixel_dist < float(self.cfg.depth_recovery.min_pixel_distance):
                return None
    
            fx = k[0, 0]
            fy = k[1, 1]
            focal = (fx + fy) / 2.0     #近似焦距

            z_wrist = focal * physical_dist / pixel_dist  #针孔模型相似三角形
    
            if (
                z_wrist < float(self.cfg.depth_recovery.wrist_min_z_m)
                or z_wrist > float(self.cfg.depth_recovery.wrist_max_z_m)
            ): #超出合理范围则估算失败
                return None
    
            # 反投影手腕得到 2D -> 3D 相机系下的点 
            cx, cy = k[0, 2], k[1, 2] #主点

            x_wrist = (wrist_2d[0] - cx) * z_wrist / fx
            y_wrist = (wrist_2d[1] - cy) * z_wrist / fy
            wrist_cam = np.array([x_wrist, y_wrist, z_wrist], dtype=np.float32)
    
            # 使用 HaMeR 相对于手腕的结构偏移
            offsets = kpts_3d_hamer - kpts_3d_hamer[0:1]
            kpts_cam = wrist_cam[np.newaxis, :] + offsets  #由计算出的手腕位置，加上HaMeR估算出来的准确3D相对姿态，得到其余20点相机系下的绝对坐标

            # 限制最小深度为0.01
            if np.any(kpts_cam[:, 2] < 0.01):
                kpts_cam[:, 2] = np.clip(kpts_cam[:, 2], 0.01, None) #clip：将数组内所有超出指定范围的数值，拉回范围边界
    
            return kpts_cam.astype(np.float32)
    def _build_hand_data(
        self,
        kpts_cam: np.ndarray,   # (21, 3) HaMeR/OpenPose order, camera coordinates
        kpts_2d: np.ndarray,    # (21, 2) HaMeR/OpenPose order, pixel coordinates
        confidence: float,
        c2w: np.ndarray,
        k: np.ndarray,
        h_img: int, w_img: int,
        is_right: bool,
    ) -> HandData:
        """从相机帧 21 个关键点构建 AriaHandData。"""

        # 相机坐标系中的手腕位姿（简单：使用手腕位置+手掌方向）
        wrist_pos_cam = kpts_cam[0]
        palm_center_cam = np.mean(kpts_cam[[5, 9, 13, 17]], axis=0)
        index_mcp_cam = kpts_cam[5]
        middle_mcp_cam = kpts_cam[9]

        # 构建手腕坐标系：Z = 手掌法向，Y = 手腕 -> 手掌方向
        v_wrist_palm = palm_center_cam - wrist_pos_cam
        v_wrist_palm_norm = np.linalg.norm(v_wrist_palm)
        if v_wrist_palm_norm < 1e-6:
            wrist_pose = None
        else:
            y_axis = v_wrist_palm / v_wrist_palm_norm
            v_lateral = index_mcp_cam - middle_mcp_cam
            x_axis = np.cross(y_axis, v_lateral)
            x_norm = np.linalg.norm(x_axis)
            if x_norm < 1e-6:
                wrist_pose = None
            else:
                x_axis /= x_norm
                z_axis = np.cross(x_axis, y_axis)
                z_axis /= (np.linalg.norm(z_axis) + 1e-6)
                y_axis = np.cross(z_axis, x_axis)

                wrist_pose = np.eye(4, dtype=np.float64)
                wrist_pose[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
                wrist_pose[:3, 3] = wrist_pos_cam

        # 抓取检测：基于比率（尺度不变）
        # 拇指尖 (Aria 0) 与食指尖 (Aria 1)，按手掌大小标准化
        thumb_tip = kpts_cam[4]
        index_tip = kpts_cam[8]
        wrist = kpts_cam[0]
        mid_mcp = kpts_cam[9]
        distance = float(np.linalg.norm(thumb_tip - index_tip))
        palm_size = float(np.linalg.norm(mid_mcp - wrist))
        grasp_ratio = distance / palm_size if palm_size > 1e-12 else None
        if palm_size > 0.01 and grasp_ratio is not None:
            closed_ratio = float(
                getattr(
                    self.cfg.grasp,
                    "ratio_closed_threshold",
                    float(self.cfg.grasp.ratio_threshold) * 0.2,
                )
            )
            open_ratio = float(
                getattr(
                    self.cfg.grasp,
                    "ratio_open_threshold",
                    float(self.cfg.grasp.ratio_threshold),
                )
            )
            grasp_state = self._score_from_interval(
                grasp_ratio, closed_ratio, open_ratio
            )
        else:
            open_distance = float(self.cfg.grasp.fallback_distance_m)
            closed_distance = float(
                getattr(
                    self.cfg.grasp,
                    "fallback_closed_distance_m",
                    open_distance * 0.45,
                )
            )
            grasp_state = self._score_from_interval(
                distance, closed_distance, open_distance
            )

        # 关节角度
        joint_angles = HandsJointAngles.from_keypoints_3d(kpts_cam)

        # 对 d2c 使用身份，因为我们没有用于基于图像的方法的设备->相机
        d2c = np.eye(4, dtype=np.float64)

        return HandData(
            d2c=d2c,
            c2w=c2w,
            is_right=is_right,
            confidence=confidence,
            wrist_pose=wrist_pose,
            palm_pose=wrist_pose,  # 近似：与手腕相同
            hand_keypoints_3d=kpts_cam,
            hand_keypoints_2d=kpts_2d,
            grasp_state=grasp_state,
            grasp_tip_distance_m=distance,
            grasp_palm_size_m=palm_size,
            grasp_ratio=grasp_ratio,
            joint_angles=joint_angles,
        )

    def _assign_world_kinematics(self, hands_data: HandsData, c2w: np.ndarray) -> None:
        """Convert one observed hand from camera coordinates to world geometry."""
        c2w = np.asarray(c2w, dtype=np.float64)
        rotation_c2w = c2w[:3, :3]
        translation_c2w = c2w[:3, 3]
        for hand_attr in ("hand_r", "hand_l"):
            hand = getattr(hands_data, hand_attr)
            if hand is None or hand.wrist_pose is None:
                continue

            wrist_rotation = rotation_c2w @ hand.wrist_pose[:3, :3]
            wrist_position = rotation_c2w @ hand.wrist_pose[:3, 3] + translation_c2w
            hand.wrist_pose_raw_world = self._make_pose(wrist_rotation, wrist_position)

            keypoints = np.asarray(hand.hand_keypoints_3d, dtype=np.float64)
            if keypoints.shape != (21, 3) or not np.isfinite(keypoints).all():
                continue
            keypoints_world = keypoints @ rotation_c2w.T + translation_c2w
            thumb_w, index_w = keypoints_world[4], keypoints_world[8]
            thumb_base_w, index_base_w = keypoints_world[2], keypoints_world[5]
            midpoint_w = 0.5 * (thumb_w + index_w)
            midpoint_rotation = self.mid_frame_builder.build(
                thumb_w=thumb_w,
                index_w=index_w,
                thumb_base_w=thumb_base_w,
                index_base_w=index_base_w,
                wrist_w=wrist_position,
                midpoint_w=midpoint_w,
            )
            if midpoint_rotation is None:
                continue

            hand.thumb_translation_raw_world = thumb_w
            hand.index_translation_raw_world = index_w
            hand.thumb_base_raw_world = thumb_base_w
            hand.index_base_raw_world = index_base_w
            hand.midpoint_translation_raw_world = midpoint_w
            hand.midpoint_pose_raw_world = self._make_pose(
                midpoint_rotation,
                midpoint_w,
            )
            hand.midpoint_orientation_raw_world = midpoint_rotation.flatten()

    @staticmethod
    def _make_pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = R.from_matrix(rotation).as_matrix()
        pose[:3, 3] = np.asarray(translation, dtype=np.float64)
        return pose

    def _filter_by_confidence(self, hands: Hands, conf_th: float = 0.3) -> None:
        for frame in hands.hands:
            for hand_attr in ("hand_r", "hand_l"):
                hand = getattr(frame, hand_attr)
                if hand is not None and hand.confidence < conf_th:
                    setattr(frame, hand_attr, None)

    def _interpolate_hand_trajectories(self, hands: Hands, max_gap: int = 6) -> None:
        """Fill only internal short gaps using world geometry and local c2w."""
        from scipy.spatial.transform import Slerp
        import cv2

        for hand_attr in ("hand_r", "hand_l"):
            present = np.array(
                [getattr(frame, hand_attr) is not None for frame in hands.hands],
                dtype=bool,
            )
            valid_indices = np.flatnonzero(present)
            for left, right in zip(valid_indices[:-1], valid_indices[1:]):
                gap = int(right - left - 1)
                if gap <= 0 or gap > max_gap:
                    continue
                start = getattr(hands.hands[left], hand_attr)
                end = getattr(hands.hands[right], hand_attr)
                if not (
                    HandsTrajectoryOptimizer.has_complete_raw_geometry(start)
                    and HandsTrajectoryOptimizer.has_complete_raw_geometry(end)
                ):
                    continue

                start_points = self._world_points(start)
                end_points = self._world_points(end)
                wrist_slerp = Slerp(
                    [0.0, 1.0],
                    R.from_matrix(
                        [
                            start.wrist_pose_raw_world[:3, :3],
                            end.wrist_pose_raw_world[:3, :3],
                        ]
                    ),
                )
                midpoint_slerp = Slerp(
                    [0.0, 1.0],
                    R.from_matrix(
                        [
                            start.midpoint_pose_raw_world[:3, :3],
                            end.midpoint_pose_raw_world[:3, :3],
                        ]
                    ),
                )
                left_ts = float(hands.hands[left].ts)
                right_ts = float(hands.hands[right].ts)
                if right_ts <= left_ts:
                    continue

                for frame_index in range(left + 1, right):
                    frame = hands.hands[frame_index]
                    ratio = float(
                        np.clip(
                            (float(frame.ts) - left_ts) / (right_ts - left_ts),
                            0.0,
                            1.0,
                        )
                    )
                    cam = self.cam.cam[frame_index]
                    world_points = (1.0 - ratio) * start_points + ratio * end_points
                    world_to_camera = np.linalg.inv(np.asarray(cam.c2w, dtype=np.float64))
                    camera_points = (
                        world_points @ world_to_camera[:3, :3].T
                        + world_to_camera[:3, 3]
                    )
                    projected, _ = cv2.projectPoints(
                        camera_points,
                        np.zeros(3),
                        np.zeros(3),
                        np.asarray(cam.k, dtype=np.float64),
                        np.asarray(cam.d if cam.d is not None else np.zeros(5), dtype=np.float64),
                    )
                    wrist_pose_world = self._make_pose(
                        wrist_slerp(ratio).as_matrix(),
                        (1.0 - ratio) * start.wrist_pose_raw_world[:3, 3]
                        + ratio * end.wrist_pose_raw_world[:3, 3],
                    )
                    midpoint_position = 0.5 * (world_points[4] + world_points[8])
                    midpoint_pose_world = self._make_pose(
                        midpoint_slerp(ratio).as_matrix(), midpoint_position
                    )
                    wrist_pose_camera = world_to_camera @ wrist_pose_world
                    hand = HandData(
                        d2c=np.array(start.d2c, copy=True) if start.d2c is not None else None,
                        c2w=np.array(cam.c2w, copy=True),
                        is_right=start.is_right,
                        confidence=(1.0 - ratio) * float(start.confidence or 0.0)
                        + ratio * float(end.confidence or 0.0),
                        tracking_state="interpolated",
                        wrist_pose=wrist_pose_camera,
                        palm_pose=wrist_pose_camera.copy(),
                        hand_keypoints_3d=camera_points,
                        hand_keypoints_2d=projected.reshape(21, 2),
                        grasp_state=(1.0 - ratio) * start.grasp_score
                        + ratio * end.grasp_score,
                        joint_angles=HandsJointAngles.from_keypoints_3d(camera_points),
                        wrist_pose_raw_world=wrist_pose_world,
                        midpoint_pose_raw_world=midpoint_pose_world,
                        midpoint_translation_raw_world=midpoint_position.copy(),
                        midpoint_orientation_raw_world=midpoint_pose_world[:3, :3].flatten(),
                    )
                    hand.thumb_translation_raw_world = world_points[4].copy()
                    hand.index_translation_raw_world = world_points[8].copy()
                    hand.thumb_base_raw_world = world_points[2].copy()
                    hand.index_base_raw_world = world_points[5].copy()
                    hand.grasp_tip_distance_m = float(
                        np.linalg.norm(camera_points[4] - camera_points[8])
                    )
                    hand.grasp_palm_size_m = float(
                        np.linalg.norm(camera_points[9] - camera_points[0])
                    )
                    hand.grasp_ratio = (
                        hand.grasp_tip_distance_m / hand.grasp_palm_size_m
                        if hand.grasp_palm_size_m > 1e-12
                        else None
                    )
                    setattr(frame, hand_attr, hand)

    @staticmethod
    def _world_points(hand: HandData) -> np.ndarray:
        camera_points = np.asarray(hand.hand_keypoints_3d, dtype=np.float64)
        c2w = np.asarray(hand.c2w, dtype=np.float64)
        return camera_points @ c2w[:3, :3].T + c2w[:3, 3]

    def _suppress_short_hands(self, hands: Hands, min_frames: int = 6) -> None:
        for hand_attr in ("hand_r", "hand_l"):
            present = [getattr(frame, hand_attr) is not None for frame in hands.hands]
            start = None
            for index, is_present in enumerate(present + [False]):
                if is_present and start is None:
                    start = index
                elif not is_present and start is not None:
                    if index - start < min_frames:
                        for frame_index in range(start, index):
                            setattr(hands.hands[frame_index], hand_attr, None)
                    start = None

    def _smooth_grasp_detection(self, hands: Hands, size: int = 5) -> None:
        from scipy.ndimage import uniform_filter1d

        for hand_attr in ("hand_r", "hand_l"):
            scores = np.array(
                [
                    getattr(frame, hand_attr).grasp_score
                    if getattr(frame, hand_attr) is not None
                    else 0.0
                    for frame in hands.hands
                ],
                dtype=np.float32,
            )
            smoothed = uniform_filter1d(scores, size=size)
            for index, frame in enumerate(hands.hands):
                hand = getattr(frame, hand_attr)
                if hand is not None:
                    hand.grasp_score = float(np.clip(smoothed[index], 0.0, 1.0))

    @staticmethod
    def _score_from_interval(value: float, closed: float, opened: float) -> float:
        """Map a distance-like value to grasp confidence."""
        if not np.isfinite(value):
            return 0.0
        if opened <= closed:
            raise ValueError("Grasp score interval must satisfy opened > closed")
        return float(np.clip((opened - value) / (opened - closed), 0.0, 1.0))
