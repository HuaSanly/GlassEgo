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
from typing import Optional, Tuple
from data_types.HandsTypes import Hands, HandsData, HandData, HandsJointAngles, MidpointFrameBuilder
from data_types.CamTypes import Cam


from hand_tracking.MediaPipeHandDetector import MediaPipeHandDetector
from hand_tracking.VitPoseHandDetector import VitPoseHandDetector
from hand_tracking.HaMeRModel import HaMeRModel
from hand_tracking.HandsOps import HandsOps
from hand_tracking.HandTrackingDiagnostics import HandTrackingDiagnostics
from preprocess.hand_tracking.HandsTrajectoryOptimizer import HandsTrajectoryOptimizer

class HaMeRHandsGenerator:

    def __init__(self, unit_dir, cfg, output_cfg, cam: Cam):
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
            self.detector = MediaPipeHandDetector(self.cfg.detector.mediapipe)
            self._detector_name = "MediaPipe"
        else:
            try:
                self.detector = VitPoseHandDetector(
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
                self.detector = MediaPipeHandDetector(
                    self.cfg.detector.mediapipe
                )
                self._detector_name = "MediaPipe"

        if not bool(self.cfg.hamer.enabled):
            raise RuntimeError(
                "hand_tracking.hamer.enabled must be true; "
                "the current generator requires HaMeR 3D reconstruction"
            )
        self.hamer_model = HaMeRModel(
            device=str(self.cfg.hamer.device),
            hamer_hf_repo=str(self.cfg.hamer.hamer_hf_repo),
            mano_hf_repo=str(self.cfg.hamer.mano_hf_repo),
        )

        if not self.hamer_model.is_available:
            print(
                "[HaMeR] WARNING: HaMeR model not available. "
                "Falling back to MediaPipe-only 3D recovery."
            )

        self.diagnostics = None
        if bool(self.cfg.diagnostics.enabled):
            self.diagnostics = HandTrackingDiagnostics(
                unit_dir=self.unit_dir,
                cfg=self.cfg.diagnostics,
                detector_backend=self._detector_name,
                frame_count=len(self.cam.cam),
            )

        # 用于速度计算的缓存
        self.prev_r_cache = None
        self.prev_l_cache = None
        self.prev_r_mid_cache = None
        self.prev_l_mid_cache = None
        self.prev_r_mid_R = None
        self.prev_l_mid_R = None
        self.mid_frame_builder = MidpointFrameBuilder()
        self._score_state = {"right": None, "left": None}

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

        self.prev_r_cache = None
        self.prev_l_cache = None
        self.prev_r_mid_cache = None
        self.prev_l_mid_cache = None
        self.prev_r_mid_R = None
        self.prev_l_mid_R = None
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
        hands = Hands(mps_path=str(self.unit_dir))
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
                )

                if hamer_result is not None:
                    kpts_cam = np.asarray(hamer_result['joints_3d'], dtype=np.float32)
                    kpts_2d = np.asarray(hamer_result['joints_2d'], dtype=np.float32)
                    hamer_confidence = self._clip_score(hamer_result['confidence'])
                    combined_confidence = self._clip_score(det_confidence * hamer_confidence)
                    if candidate_diagnostic is not None:
                        candidate_diagnostic.hamer_succeeded = True
                        candidate_diagnostic.hamer_confidence = hamer_confidence
                        candidate_diagnostic.combined_confidence = combined_confidence
                        candidate_diagnostic.final_confidence = combined_confidence

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
                    final_confidence = self._final_confidence(
                        det_confidence,
                        geometry_confidence,
                        temporal,
                    )
                    candidate = {
                        "detection": hand,
                        "kpts_cam": kpts_cam,
                        "kpts_2d": kpts_2d,
                        "base_confidence": combined_confidence,
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
            self._compute_and_assign_vel(frame_data, c2w, dt)

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
            self._interpolate_hand_trajectories(
                hands,
                max_gap=int(
                    self.cfg.postprocess.interpolation.max_gap_frames
                ),
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
            optimizer = HandsTrajectoryOptimizer(self.cfg.trajectory, dt)
            optimizer.run(hands)
        self._smooth_grasp_detection(hands, size=self.cfg.grasp.smooth_window)

        # 第四阶段：报告
        os.makedirs(self.preprocess_dir, exist_ok=True)
        try:
            HandsOps.save_hands_analysis_plots_two(
                hands, str(self.preprocess_dir), dt, self.cfg
            )
        except Exception as e:
            print(f"[HaMeR] Warning: analysis plots failed: {e}")
        HandsOps.print_summary_and_eval(hands)

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
                img_path = self.preprocess_dir / "all_data" / f"{cam_d.idx:05d}" / "rgb.png"
                img = cv2.imread(str(img_path)) if img_path.is_file() else None
            if img is None:
                raise RuntimeError(f"Missing visualization frame: {cam_d.idx}")

            if idx < len(hands.hands):
                img = HandsOps.draw_aria_hands_skeleton(
                    img, hands.hands[idx],
                    cam_d.k, getattr(cam_d, 'd', np.zeros(8)), cam_d.c2w,
                    grasp_threshold=self.cfg.grasp.fallback_distance_m,
                )
                img = HandsOps.draw_aria_hands_panel(
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
        if palm_size > 0.01:
            grasp_ratio = distance / palm_size
            grasp_state = (
                1 if grasp_ratio < float(self.cfg.grasp.ratio_threshold) else 0
            )
        else:
            grasp_threshold = float(self.cfg.grasp.fallback_distance_m)
            grasp_state = 1 if distance < grasp_threshold else 0

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
            joint_angles=joint_angles,
        )
    def _compute_and_assign_vel(self, hands_data: HandsData,
                                c2w: np.ndarray, dt: float) -> None:
        """计算世界坐标系中的位姿、速度和中点夹爪坐标系。"""
        # 旋转矩阵的鲁棒化，避免神经网络预测出的旋转矩阵不是正交矩阵
        def robust_rot(matrix):  
            try:
                return R.from_matrix(matrix)
            except ValueError:
                U, S, Vt = np.linalg.svd(matrix) # svd分解，U和Vt是正交矩阵，S是奇异值，任意矩阵M可分解为M = U @ S @ Vt
                d = np.linalg.det(U @ Vt)#计算行列式
                if d < 0: U[:, -1] *= -1      #若U @ Vt的行列式小于0，说明是一个反射矩阵，调整U的最后一列，使其变为正交矩阵
                return R.from_matrix(U @ Vt)   #构造最接近的旋转矩阵

        for is_right in [True, False]:
            h_data = hands_data.hand_r if is_right else hands_data.hand_l
            prev_cache = self.prev_r_cache if is_right else self.prev_l_cache
            prev_mid_cache = self.prev_r_mid_cache if is_right else self.prev_l_mid_cache
            prev_R = self.prev_r_mid_R if is_right else self.prev_l_mid_R

            if h_data and h_data.wrist_pose is not None:
                # 手腕 -> 世界
                p_cam = h_data.wrist_pose[:3, 3]
                r_cam = h_data.wrist_pose[:3, :3]
                p_world = (c2w[:3, :3] @ p_cam) + c2w[:3, 3]
                r_world = c2w[:3, :3] @ r_cam

                h_data.wrist_pose_raw_world = np.eye(4)
                h_data.wrist_pose_raw_world[:3, :3] = r_world
                h_data.wrist_pose_raw_world[:3, 3] = p_world

                if prev_cache is not None:
                    h_data.wrist_lin_vel_raw_world = (p_world - prev_cache['pos']) / dt  #差值估算瞬时速度
                    rel = prev_cache['rot'].T @ r_world  #上一帧到这一帧旋转姿态的相对旋转矩阵，蕴含了“需要转多少”
                    h_data.wrist_ang_vel_raw_world = robust_rot(rel).as_rotvec() / dt  # 计算角速度 .as_rotvec()将旋转矩阵转换为旋转向量，表示旋转轴和旋转角度
                
                #更新速度缓存
                cache_val = {'pos': p_world, 'rot': r_world}
                if is_right: self.prev_r_cache = cache_val
                else: self.prev_l_cache = cache_val

                # 中点夹爪坐标系
                if h_data.hand_keypoints_3d is not None and len(h_data.hand_keypoints_3d) >= 21:
                    thumb_w = (c2w[:3, :3] @ h_data.hand_keypoints_3d[4]) + c2w[:3, 3] #拇指尖
                    index_w = (c2w[:3, :3] @ h_data.hand_keypoints_3d[8]) + c2w[:3, 3] #食指尖
                    thumb_base_w = (c2w[:3, :3] @ h_data.hand_keypoints_3d[2]) + c2w[:3, 3] #拇指根
                    index_base_w = (c2w[:3, :3] @ h_data.hand_keypoints_3d[5]) + c2w[:3, 3] #食指根

                    h_data.thumb_translation_raw_world = thumb_w
                    h_data.index_translation_raw_world = index_w
                    h_data.thumb_base_raw_world = thumb_base_w
                    h_data.index_base_raw_world = index_base_w

                    midpoint_w = (thumb_w + index_w) / 2.0 #取拇指尖和食指尖的中点作为中点夹爪坐标系的原点
                    h_data.midpoint_translation_raw_world = midpoint_w

                    R_mid = self.mid_frame_builder.build(
                        thumb_w=thumb_w, index_w=index_w,
                        thumb_base_w=thumb_base_w, index_base_w=index_base_w,
                        wrist_w=p_world, midpoint_w=midpoint_w, prev_R=prev_R,
                    )
                    if R_mid is None:
                        R_mid = prev_R if prev_R is not None else r_world.copy()

                    h_data.midpoint_pose_raw_world = np.eye(4)  #将旋转矩阵和位置拼接成变换矩阵
                    h_data.midpoint_pose_raw_world[:3, :3] = R_mid
                    h_data.midpoint_pose_raw_world[:3, 3] = midpoint_w
                    h_data.midpoint_orientation_raw_world = R_mid.flatten() #压平成1*9的一维数据，为了给模型输入的

                    #算中点的速度
                    if prev_mid_cache is not None: 
                        h_data.midpoint_lin_vel_raw_world = (midpoint_w - prev_mid_cache['pos']) / dt
                        rel = prev_mid_cache['rot'].T @ R_mid
                        h_data.midpoint_ang_vel_raw_world = robust_rot(rel).as_rotvec() / dt

                    cache_mid = {'pos': midpoint_w, 'rot': R_mid}
                    if is_right:
                        self.prev_r_mid_cache = cache_mid
                        self.prev_r_mid_R = R_mid
                    else:
                        self.prev_l_mid_cache = cache_mid
                        self.prev_l_mid_R = R_mid
    # 时域清洗
    def _filter_by_confidence(self, hands: Hands, conf_th: float = 0.3) -> None:
        for frame_data in hands.hands:
            for attr in ["hand_r", "hand_l"]:
                h = getattr(frame_data, attr)
                if h and (h.confidence < conf_th):
                    setattr(frame_data, attr, None)
    def _interpolate_hand_trajectories(self, hands: Hands, max_gap: int = 3) -> None:
        from scipy.spatial.transform import Slerp
        for attr in ["hand_r", "hand_l"]:
            presence = [getattr(h, attr) is not None for h in hands.hands] #有效值列表 例：[True, False, True, True]
            indices = np.where(presence)[0]   #取出有效的下标数组 例：[0,2,3]
            if len(indices) < 2:
                continue
            for start_i, end_i in zip(indices[:-1], indices[1:]): #错位滑动遍历，即[(0,2),(2,3)],0-2之间都是无效帧，可以插值
                gap = end_i - start_i - 1 #有几帧是无效的
                if 0 < gap <= max_gap: 
                    h_start = getattr(hands.hands[start_i], attr)  #拿到有效的首尾手部数据
                    h_end = getattr(hands.hands[end_i], attr)
                    if h_start.wrist_pose is None or h_end.wrist_pose is None:
                        continue
                    steps = np.linspace(0, 1, gap + 2)[1:-1] #生成插值步长，去掉首尾，即将0-1之间均匀分成gap+2份，取中间的gap份，+2是因为和首尾之间还有两个空隙
                    for j, t in enumerate(steps):
                        fill_idx = start_i + j + 1 #要填充的无效值下标
                        h_new = HandData(
                            d2c=h_start.d2c, c2w=h_start.c2w,
                            is_right=h_start.is_right,
                            confidence=(1.0 - t) * h_start.confidence + t * h_end.confidence,
                        )
                        # 插入手腕位姿
                        pos_interp = (1.0 - t) * h_start.wrist_pose[:3, 3] + t * h_end.wrist_pose[:3, 3] #手腕位置线性插值
                    
                        try:
                            rots = R.from_matrix([h_start.wrist_pose[:3, :3], h_end.wrist_pose[:3, :3]])  #已知的两个首尾旋转矩阵转换为一个旋转对象列表
                            slerp = Slerp([0, 1], rots) #创建一个旋转插值器
                            rot_interp = slerp(t).as_matrix() #在t处插值，并转换回旋转矩阵
                        except Exception:
                            rot_interp = h_start.wrist_pose[:3, :3] #若计Slerp计算失败，则使用首帧的旋转矩阵作为插值结果
                        
                        T_interp = np.eye(4) #创建一个4x4的单位阵
                        T_interp[:3, :3] = rot_interp
                        T_interp[:3, 3] = pos_interp
                        h_new.wrist_pose = T_interp
                        h_new.palm_pose = T_interp

                        # 21关键点直接线性插值
                        if h_start.hand_keypoints_3d is not None and h_end.hand_keypoints_3d is not None:
                            h_new.hand_keypoints_3d = (1.0 - t) * h_start.hand_keypoints_3d + t * h_end.hand_keypoints_3d
                        if h_start.hand_keypoints_2d is not None and h_end.hand_keypoints_2d is not None:
                            h_new.hand_keypoints_2d = (1.0 - t) * h_start.hand_keypoints_2d + t * h_end.hand_keypoints_2d

                        # 抓取状态和首保持一致
                        h_new.grasp_state = h_start.grasp_state
                        # 把新手部数据插入这个列表对象
                        setattr(hands.hands[fill_idx], attr, h_new)
    def _suppress_short_hands(self, hands: Hands, min_frames: int = 5) -> None:
            for attr in ["hand_r", "hand_l"]:
                presence = [getattr(h, attr) is not None for h in hands.hands]
                count, segments = 0, []
                for i, is_present in enumerate(presence):
                    if is_present:
                        count += 1
                    else:
                        if 0 < count < min_frames:
                            segments.append((i - count, i))
                        count = 0
                if 0 < count < min_frames:
                    segments.append((len(presence) - count, len(presence)))
                for start, end in segments: #清理所有待删除片段
                    for i in range(start, end):
                        setattr(hands.hands[i], attr, None)
    def _smooth_grasp_detection(self, hands: Hands, size: int = 5) -> None:
        """抓取状态平滑"""
        from scipy.ndimage import uniform_filter1d
        for attr in ["hand_r", "hand_l"]:
            states = []
            for h in hands.hands:
                hand = getattr(h, attr)
                states.append(hand.grasp_state if hand else 0)
            g = np.array(states, dtype=np.float32)
            g = uniform_filter1d(g, size=size) #一维均匀滤波，对每个位置取相邻size个数的平均值
            g = (g > 0.5).astype(int) #  (g > 0.5) 这一步把大于0.5的数转换为浮点数组，然后.astype转换为int类型

            # 抓取状态闪烁抑制
            flicker_max = self.cfg.grasp.flicker_max_len
            for flip_val in [0, 1]:
                count = 0
                for i in range(len(g)):
                    if g[i] == flip_val:
                        count += 1
                    else:
                        if 0 < count <= flicker_max:
                            for j in range(i - count, i):  #翻转短片段
                                g[j] = 1 - flip_val
                        count = 0
            #处理后结果写入
            for i, h in enumerate(hands.hands):
                hand = getattr(h, attr)
                if hand:
                    hand.grasp_state = int(g[i])
