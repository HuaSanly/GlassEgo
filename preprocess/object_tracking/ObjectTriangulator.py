# -*- coding: utf-8 -*-
# @FileName: ObjectTriangulator.py

"""
====================================================================================================
Object Triangulation & PCA Pose Pipeline (ObjectTriangulator.py)
====================================================================================================

Description:
    Convert CoTracker 2D object tracks into 3D object keypoints using VIO camera poses,
    then estimate a 6-DOF object frame with the HumanEgo PCA branches.

Technical Specifics:
    - Multi-view DLT with camera intrinsics removed.
    - Point-only bundle adjustment with Huber loss.
    - PCA pose estimation with pca1 / pca2 only; VLM / OrientAnything is intentionally excluded.
====================================================================================================
"""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
from omegaconf import OmegaConf
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from utils.utils_vis import draw_glass_rect


@dataclass(frozen=True)
class ObjectTriangulatorConfig:
    """HumanEgo CamTriangulator PCA 分支默认参数。"""

    pose_method: str = "pca2"
    step: int = 1
    smooth_window: int = 7
    smooth_polyorder: int = 2
    ba_f_scale: float = 3.0
    axes_len_m: float = 0.12
    point_radius_m: float = 0.005
    line_radius_m: float = 0.001


def estimate_frame_pca1(
    pts_cam: np.ndarray,
    is_anchor: bool = True,
    anchor_center_cam: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, dict]:
    """HumanEgo PCA1：星形/关系式物体坐标系估计。"""
    assert pts_cam.ndim == 2 and pts_cam.shape[1] == 3
    t = pts_cam.mean(axis=0)

    q = pts_cam - t[None, :]
    c = q.T @ q
    evals, evecs = np.linalg.eigh(c)

    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    v1 = evecs[:, 0]
    v2 = evecs[:, 1]
    v3 = evecs[:, 2]

    cam_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    cam_right = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    if is_anchor:
        y_axis = v2.copy()
        if np.dot(y_axis, cam_up) > 0:
            y_axis = -y_axis

        lam_a, lam_b = evals[0], evals[1]
        anisotropy = (lam_a - lam_b) / (lam_a + 1e-12)

        if anisotropy > 0.15:
            x_axis = v1.copy()
            if np.dot(x_axis, cam_right) < 0:
                x_axis = -x_axis
            method_used = f"anchor_pca (aniso:{anisotropy:.2f})"
        else:
            x_proj = cam_right - np.dot(cam_right, y_axis) * y_axis
            x_axis = x_proj / (np.linalg.norm(x_proj) + 1e-12)
            method_used = f"anchor_symmetric (aniso:{anisotropy:.2f})"

    else:
        if anchor_center_cam is None:
            raise ValueError("anchor_center_cam MUST be provided for context objects!")

        y_axis = v3.copy()
        if np.dot(y_axis, cam_up) > 0:
            y_axis = -y_axis

        vec_to_anchor = anchor_center_cam - t
        x_proj = vec_to_anchor - np.dot(vec_to_anchor, y_axis) * y_axis
        norm_x = np.linalg.norm(x_proj)

        if norm_x > 1e-4:
            x_axis = x_proj / norm_x
            method_used = "context_relational_aligned"
        else:
            x_proj = cam_right - np.dot(cam_right, y_axis) * y_axis
            x_axis = x_proj / (np.linalg.norm(x_proj) + 1e-12)
            method_used = "context_stacked_fallback"

    x_axis = x_axis - np.dot(x_axis, y_axis) * y_axis
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-12)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-12)

    r_mat = np.stack([x_axis, y_axis, z_axis], axis=1)
    if np.linalg.det(r_mat) < 0:
        x_axis = -x_axis
        r_mat = np.stack([x_axis, y_axis, z_axis], axis=1)

    T_o2c = np.eye(4, dtype=np.float64)
    T_o2c[:3, :3] = r_mat
    T_o2c[:3, 3] = t

    info = {
        "pca_evals": evals.tolist(),
        "method": method_used,
    }
    return T_o2c, info


def estimate_frame_pca2(
    pts_cam: np.ndarray,
    is_anchor: bool = True,
    anchor_center_cam: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, dict]:
    """HumanEgo PCA2：object-centric + relational 姿态估计。"""
    assert pts_cam.ndim == 2 and pts_cam.shape[1] == 3
    t = pts_cam.mean(axis=0)

    q = pts_cam - t[None, :]
    c = q.T @ q
    evals, evecs = np.linalg.eigh(c)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    v_long, v_mid, v_short = evecs[:, 0], evecs[:, 1], evecs[:, 2]

    cam_down = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    cam_right = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    is_vertical = abs(np.dot(v_long, cam_down)) > abs(np.dot(v_long, cam_right))

    if is_vertical:
        y_axis = v_long.copy()
        if np.dot(y_axis, cam_down) < 0:
            y_axis = -y_axis

        if is_anchor:
            if abs(np.dot(v_mid, cam_right)) > abs(np.dot(v_short, cam_right)):
                x_axis = v_mid.copy()
            else:
                x_axis = v_short.copy()
            if np.dot(x_axis, cam_right) < 0:
                x_axis = -x_axis
            method_used = "pca_centric_vertical_anchor"
        else:
            if anchor_center_cam is None:
                raise ValueError("anchor_center_cam MUST be provided for context objects!")
            vec_to_anchor = anchor_center_cam - t
            if abs(np.dot(v_mid, vec_to_anchor)) > abs(np.dot(v_short, vec_to_anchor)):
                x_axis = v_mid.copy()
            else:
                x_axis = v_short.copy()
            if np.dot(x_axis, vec_to_anchor) < 0:
                x_axis = -x_axis
            method_used = "pca_centric_vertical_context"

    else:
        x_axis = v_long.copy()

        if is_anchor:
            if np.dot(x_axis, cam_right) < 0:
                x_axis = -x_axis
            method_used = "pca_centric_horizontal_anchor"
        else:
            if anchor_center_cam is None:
                raise ValueError("anchor_center_cam MUST be provided for context objects!")
            vec_to_anchor = anchor_center_cam - t
            if np.dot(x_axis, vec_to_anchor) < 0:
                x_axis = -x_axis
            method_used = "pca_centric_horizontal_context"

        if abs(np.dot(v_mid, cam_down)) > abs(np.dot(v_short, cam_down)):
            y_axis = v_mid.copy()
        else:
            y_axis = v_short.copy()
        if np.dot(y_axis, cam_down) < 0:
            y_axis = -y_axis

    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-12)
    r_mat = np.stack([x_axis, y_axis, z_axis], axis=1)

    T_o2c = np.eye(4, dtype=np.float64)
    T_o2c[:3, :3] = r_mat
    T_o2c[:3, 3] = t

    info = {
        "pca_evals": evals.tolist(),
        "method": method_used,
    }
    return T_o2c, info


class ObjectTriangulatorEngine:
    """DLT 三角化与 BA 精修。"""

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else ObjectTriangulatorConfig()

    @staticmethod
    def triangulate_dlt(poses, Ks, obs_uv, vis):
        """Multi-view DLT triangulation with normalized coordinates (K removed)."""
        A = []
        for i in range(len(poses)):
            if vis[i] == 0:
                continue
            u, v = obs_uv[i]
            fx, fy = Ks[i][0, 0], Ks[i][1, 1]
            cx, cy = Ks[i][0, 2], Ks[i][1, 2]
            un = (u - cx) / fx
            vn = (v - cy) / fy

            T_w2c = np.linalg.inv(poses[i])
            P = T_w2c[:3, :]

            A.append(un * P[2, :] - P[0, :])
            A.append(vn * P[2, :] - P[1, :])

        if len(A) < 4:
            return np.zeros(3), 0.0

        _, s, vh = np.linalg.svd(np.array(A))
        cond = s[0] / s[-1] if s[-1] != 0 else 0.0

        X = vh[-1]
        p3d = X[:3] / X[3]
        return p3d, cond

    def ba_refine(self, p3d, poses, Ks, obs_uv, vis):
        """Point-only Bundle Adjustment refinement with Huber loss."""

        def res(x):
            pts = []
            for i in range(len(poses)):
                if vis[i] == 0:
                    continue
                T_w2c = np.linalg.inv(poses[i])
                pc = T_w2c[:3, :3] @ x + T_w2c[:3, 3]
                if pc[2] < 1e-4:
                    continue
                uv = Ks[i] @ pc
                pts.extend((uv[:2] / uv[2]) - obs_uv[i])
            return np.array(pts)

        sol = least_squares(
            res,
            p3d,
            loss="huber",
            f_scale=float(self.cfg.ba_f_scale),
            max_nfev=60,
        )
        return sol.x


class ObjectTriangulator:
    """Generator 调用的物体 3D 三角化与 PCA 位姿模块。"""

    def __init__(self, unit_dir: str | Path, cfg=None):
        self.unit_dir = Path(unit_dir).expanduser().resolve()
        self.cfg = cfg if cfg is not None else ObjectTriangulatorConfig()
        self.output_dir = self.unit_dir / "preprocess" / "objects" / "triangulation"
        self.result_path = self.output_dir / "object_3d_results.json"
        self.qa_path = self.output_dir / "object_3d_vis.png"
        self.ply_path = self.output_dir / "object_3d_vis.ply"
        self.engine = ObjectTriangulatorEngine(self.cfg)

    def triangulate(self, tracks_document: dict, frames_bgr: list) -> tuple[dict, np.ndarray]:
        """把 CoTracker tracks 转为 3D 点云和 PCA 物体位姿。"""
        frame_indices = [int(item) for item in tracks_document["frames"]]
        if len(frame_indices) != len(frames_bgr):
            raise ValueError(
                "frames_bgr must align with CoTracker frames: "
                f"images={len(frames_bgr)}, tracks={len(frame_indices)}"
            )

        calibration = self._load_calibration()
        vio_frames = self._load_vio_frames()
        poses = [vio_frames[index] for index in frame_indices]
        K = self._camera_matrix(calibration)
        Ks = [K for _ in poses]

        cam0_c2w = poses[0]
        T_w2c0 = np.linalg.inv(cam0_c2w)
        R_w2c0, t_w2c0 = T_w2c0[:3, :3], T_w2c0[:3, 3]

        objects = {}
        anchor_center_cam = None
        obj_keys = sorted(
            key for key in tracks_document.get("objects", {}).keys()
            if key.startswith("obj")
        )
        if not obj_keys:
            raise ValueError("No obj* tracks found for triangulation")

        for obj_key in obj_keys:
            is_anchor = obj_key == obj_keys[0]
            object_tracks = tracks_document["objects"][obj_key]
            tracks = np.asarray(object_tracks["tracks"], dtype=np.float64)
            visibility = np.asarray(object_tracks["visibility"], dtype=np.float64)
            tracks = self._smooth_tracks(tracks)

            indices = self._subsample_indices(tracks.shape[0])
            sub_poses = [poses[index] for index in indices]
            sub_Ks = [Ks[index] for index in indices]
            sub_tracks = tracks[indices]
            sub_vis = visibility[indices]

            points_world = []
            conditions = []
            for point_idx in range(tracks.shape[1]):
                p3d_init, condition = self.engine.triangulate_dlt(
                    sub_poses,
                    sub_Ks,
                    sub_tracks[:, point_idx],
                    sub_vis[:, point_idx],
                )
                if np.allclose(p3d_init, 0.0) or not np.all(np.isfinite(p3d_init)):
                    continue
                p3d_refined = self.engine.ba_refine(
                    p3d_init,
                    sub_poses,
                    sub_Ks,
                    sub_tracks[:, point_idx],
                    sub_vis[:, point_idx],
                )
                if np.allclose(p3d_refined, 0.0) or not np.all(np.isfinite(p3d_refined)):
                    continue
                points_world.append(p3d_refined)
                conditions.append(float(condition))

            if len(points_world) < 3:
                raise ValueError(f"{obj_key} has fewer than 3 triangulated points")

            pts_world = np.asarray(points_world, dtype=np.float64)
            pts_cam0 = (R_w2c0 @ pts_world.T + t_w2c0[:, None]).T
            T_o2c0, info = self._estimate_pose(
                pts_cam0,
                is_anchor=is_anchor,
                anchor_center_cam=anchor_center_cam,
            )
            if is_anchor:
                anchor_center_cam = T_o2c0[:3, 3]

            T_o2w = cam0_c2w @ T_o2c0
            self._validate_transform(T_o2c0, f"{obj_key}.object_to_cam0_matrix")
            self._validate_transform(T_o2w, f"{obj_key}.object_to_world_matrix")

            objects[obj_key] = {
                "points_3d_world": pts_world.tolist(),
                "points_3d_cam0": pts_cam0.tolist(),
                "object_to_cam0_matrix": T_o2c0.tolist(),
                "object_to_world_matrix": T_o2w.tolist(),
                "center_world": T_o2w[:3, 3].tolist(),
                "pose_info": info,
                "triangulated_points": int(len(pts_world)),
                "dlt_condition_mean": float(np.mean(conditions)) if conditions else 0.0,
            }

        document = {
            "schema_version": 1,
            "method": "humanego_camtriangulator_pca",
            "pose_method": str(self.cfg.pose_method),
            "frames": frame_indices,
            "cam0_frame_idx": int(frame_indices[0]),
            "cam0_c2w": cam0_c2w.tolist(),
            "camera_intrinsics": K.tolist(),
            "objects": objects,
            "outputs": {
                "results": str(self.result_path),
                "qa": str(self.qa_path),
                "ply": str(self.ply_path),
            },
        }
        qa = self._draw_qa(frames_bgr[-1].copy(), np.linalg.inv(poses[-1]), K, document)
        return document, qa

    def save_outputs(self, document: dict, qa_bgr: np.ndarray) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write_json(self.result_path, document)
        cv2.imwrite(str(self.qa_path), qa_bgr)
        self._export_ply(document)

    def _estimate_pose(
        self,
        pts_cam: np.ndarray,
        is_anchor: bool,
        anchor_center_cam: np.ndarray | None,
    ) -> tuple[np.ndarray, dict]:
        method = str(self.cfg.pose_method).lower()
        if method == "pca1":
            return estimate_frame_pca1(
                pts_cam,
                is_anchor=is_anchor,
                anchor_center_cam=anchor_center_cam,
            )
        if method == "pca2":
            return estimate_frame_pca2(
                pts_cam,
                is_anchor=is_anchor,
                anchor_center_cam=anchor_center_cam,
            )
        raise ValueError(f"Unsupported object pose method: {self.cfg.pose_method}")

    def _smooth_tracks(self, tracks: np.ndarray) -> np.ndarray:
        smooth_window = int(self.cfg.smooth_window)
        smooth_polyorder = int(self.cfg.smooth_polyorder)
        if smooth_window % 2 == 0:
            smooth_window += 1
        if smooth_window > len(tracks):
            smooth_window = len(tracks) if len(tracks) % 2 == 1 else len(tracks) - 1
        if smooth_window <= smooth_polyorder or smooth_window < 3:
            return tracks.copy()

        out = tracks.copy()
        for point_idx in range(out.shape[1]):
            for dim in range(2):
                out[:, point_idx, dim] = savgol_filter(
                    out[:, point_idx, dim],
                    smooth_window,
                    smooth_polyorder,
                )
        return out

    def _subsample_indices(self, total_frames: int) -> list[int]:
        step = max(1, int(self.cfg.step))
        indices = np.arange(0, total_frames, step).tolist()
        last_idx = total_frames - 1
        if last_idx not in indices:
            indices.append(last_idx)
        return sorted(indices)

    def _load_calibration(self):
        calibration_path = self.unit_dir / "calibration.yaml"
        if not calibration_path.is_file():
            raise FileNotFoundError(f"Calibration not found: {calibration_path}")
        return OmegaConf.load(calibration_path)

    def _camera_matrix(self, calibration) -> np.ndarray:
        intrinsics = OmegaConf.select(calibration, "camera.intrinsics", default=[])
        if len(intrinsics) != 4:
            raise ValueError("camera.intrinsics must contain [fx, fy, cx, cy]")
        fx, fy, cx, cy = (float(value) for value in intrinsics)
        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def _load_vio_frames(self) -> dict[int, np.ndarray]:
        pose_path = self.unit_dir / "preprocess" / "vio" / "poses.json"
        if not pose_path.is_file():
            raise FileNotFoundError(f"VIO poses not found: {pose_path}")
        with pose_path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
        if document.get("schema_version") != 2:
            raise ValueError("Unsupported VIO pose schema")

        poses = {}
        for item in document.get("frames", []):
            if not item.get("valid", False):
                continue
            c2w = np.asarray(item["c2w"], dtype=np.float64)
            self._validate_transform(c2w, "c2w")
            poses[int(item["frame_idx"])] = c2w
        if not poses:
            raise ValueError("VIO poses contain no valid frames")
        return poses

    def _draw_qa(self, img_bgr: np.ndarray, w2c: np.ndarray, K: np.ndarray, document: dict):
        def project(p_world):
            pc = w2c[:3, :3] @ p_world + w2c[:3, 3]
            if pc[2] < 1e-4:
                return None
            uv = K @ pc
            return int(uv[0] / uv[2]), int(uv[1] / uv[2])

        for obj_key, data in document["objects"].items():
            for point in data["points_3d_world"]:
                uv = project(np.asarray(point, dtype=np.float64))
                if uv is not None:
                    cv2.circle(img_bgr, uv, 4, (0, 255, 0), -1, cv2.LINE_AA)

            T_o2w = np.asarray(data["object_to_world_matrix"], dtype=np.float64)
            origin = T_o2w[:3, 3]
            uv0 = project(origin)
            if uv0 is None:
                continue
            cv2.circle(img_bgr, uv0, 5, (0, 215, 255), -1, cv2.LINE_AA)
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
            labels = ["X", "Y", "Z"]
            for axis_idx in range(3):
                endpoint = origin + T_o2w[:3, axis_idx] * float(self.cfg.axes_len_m)
                uv1 = project(endpoint)
                if uv1 is None:
                    continue
                cv2.arrowedLine(
                    img_bgr,
                    uv0,
                    uv1,
                    colors[axis_idx],
                    2,
                    tipLength=0.2,
                    line_type=cv2.LINE_AA,
                )
                cv2.putText(
                    img_bgr,
                    labels[axis_idx],
                    (uv1[0] + 5, uv1[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    colors[axis_idx],
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                img_bgr,
                obj_key,
                (uv0[0] + 10, uv0[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        draw_glass_rect(img_bgr, (10, 10), (380, 70))
        cv2.putText(
            img_bgr,
            f"OBJECT TRIANGULATOR ({len(document['objects'])} OBJS)",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img_bgr,
            f"Pose: {document['pose_method']}",
            (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return img_bgr

    def _export_ply(self, document: dict) -> None:
        geos = []
        obj_colors = [
            [0.2, 0.9, 0.2],
            [0.2, 0.8, 0.9],
            [0.9, 0.2, 0.8],
            [0.9, 0.2, 0.2],
        ]
        for index, (obj_key, data) in enumerate(document["objects"].items()):
            pts = np.asarray(data["points_3d_cam0"], dtype=np.float64)
            T_o2c0 = np.asarray(data["object_to_cam0_matrix"], dtype=np.float64)
            origin = T_o2c0[:3, 3]
            base_color = obj_colors[index % len(obj_colors)]

            for point in pts:
                sphere = o3d.geometry.TriangleMesh.create_sphere(
                    radius=float(self.cfg.point_radius_m)
                )
                sphere.translate(point)
                sphere.paint_uniform_color(base_color)
                geos.append(sphere)

            center_sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=float(self.cfg.point_radius_m) * 1.6
            )
            center_sphere.translate(origin)
            center_sphere.paint_uniform_color([1.0, 0.8, 0.0])
            geos.append(center_sphere)

            axis_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            for axis_idx in range(3):
                axis_vec = T_o2c0[:3, axis_idx]
                arrow = self._create_arrow(
                    origin,
                    origin + axis_vec * float(self.cfg.axes_len_m),
                    axis_colors[axis_idx],
                )
                geos.append(arrow)

        combined = o3d.geometry.TriangleMesh()
        for geo in geos:
            combined += geo
        o3d.io.write_triangle_mesh(str(self.ply_path), combined)

    @staticmethod
    def _create_arrow(start, end, color):
        vec = end - start
        length = np.linalg.norm(vec)
        if length < 1e-6:
            return o3d.geometry.TriangleMesh()
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.002,
            cone_radius=0.005,
            cylinder_height=length * 0.8,
            cone_height=length * 0.2,
        )
        arrow.paint_uniform_color(color)
        z_axis = np.array([0, 0, 1], dtype=np.float64)
        rotation_matrix = ObjectTriangulator._get_rotation_matrix(
            z_axis,
            vec / length,
        )
        arrow.rotate(rotation_matrix, center=(0, 0, 0))
        arrow.translate(start)
        return arrow

    @staticmethod
    def _get_rotation_matrix(vec1, vec2):
        a = (vec1 / (np.linalg.norm(vec1) + 1e-12)).reshape(3)
        b = (vec2 / (np.linalg.norm(vec2) + 1e-12)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        if s < 1e-8:
            return np.eye(3)
        kmat = np.array(
            [
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0],
            ],
            dtype=np.float64,
        )
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2 + 1e-12))

    @staticmethod
    def _validate_transform(transform: np.ndarray, label: str) -> None:
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError(f"{label} must be a finite 4x4 matrix")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
            raise ValueError(f"{label} must be homogeneous")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError(f"{label} rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
            raise ValueError(f"{label} rotation determinant must be +1")

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
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
