# -*- coding: utf-8 -*-
# @FileName: CoTracker.py

"""
====================================================================================================
Object CoTracker3 Offline Tracking Pipeline (CoTracker.py)
====================================================================================================

Description:
    Track object keypoints across a sampled frame sequence using CoTracker3.
    This module does not scan data units and does not expose a CLI; it is called by
    ObjectTrackingGenerator as one stage of the object_tracking pipeline.

Technical Specifics:
    - Letterbox frames to a square inference resolution.
    - Track all object points together, then split results back by object key.
    - Use the HumanEgo-style bidirectional chunk flow around the reference frame.
====================================================================================================
"""

import gc
import time
from contextlib import nullcontext
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from cotracker.predictor import CoTrackerPredictor
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from utils.utils_vis import draw_glass_rect


@dataclass(frozen=True)
class CoTrackerConfig:
    """CoTracker3 默认运行参数。"""

    cotracker_res: int = 640
    cotracker_chunk_size: int = 100
    cotracker_viz_trail_len: int = 20
    hf_repo_id: str = "facebook/cotracker3"
    checkpoint_name: str = "scaled_offline.pth"


class CoTrackerEngine:
    """CoTracker3 模型与核心推理逻辑。"""

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else CoTrackerConfig()
        self.device = self._get_device()
        print(f"║ [CoTracker] Initializing on {self.device}...")

        ckpt = hf_hub_download(
            repo_id=self.cfg.hf_repo_id,
            filename=self.cfg.checkpoint_name,
        )
        self.model = CoTrackerPredictor(checkpoint=ckpt).to(self.device)

    def _get_device(self):
        """选择可用推理设备。"""
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _letterbox(self, img_bgr):
        """等比例缩放并补边到正方形，同时返回反变换所需元数据。"""
        h, w = img_bgr.shape[:2]
        res = int(self.cfg.cotracker_res)
        scale = res / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (new_w, new_h))

        pad_w, pad_h = res - new_w, res - new_h
        pad_l, pad_t = pad_w // 2, pad_h // 2

        img_sq = cv2.copyMakeBorder(
            resized,
            pad_t,
            pad_h - pad_t,
            pad_l,
            pad_w - pad_l,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        meta = {
            "scale": scale,
            "pad_l": pad_l,
            "pad_t": pad_t,
            "orig_w": w,
            "orig_h": h,
        }
        return img_sq, meta

    def _unletterbox_points(self, tracks_sq, meta):
        """把 CoTracker 正方形输入坐标映射回原始视频坐标。"""
        s, px, py = meta["scale"], meta["pad_l"], meta["pad_t"]
        out = tracks_sq.copy()
        out[..., 0] = (out[..., 0] - px) / (s + 1e-12)
        out[..., 1] = (out[..., 1] - py) / (s + 1e-12)
        return out

    def _process_single_chunk(self, frames_sq, queries_at_first_frame):
        """处理一个连续 chunk，query 均位于 chunk 的第 0 帧。"""
        video_np = np.stack(frames_sq).astype(np.uint8)
        video_tensor = (
            torch.from_numpy(video_np)
            .permute(0, 3, 1, 2)[None]
            .float()
            .to(self.device)
        )

        # CoTracker sparse query 格式为 [t, x, y]；这里 t=0 表示 chunk 起点。
        qs = [[0, x, y] for (x, y) in queries_at_first_frame]
        queries_tensor = torch.tensor([qs], device=self.device).float()

        autocast_ctx = (
            torch.amp.autocast("cuda")
            if self.device == "cuda"
            else nullcontext()
        )
        with torch.no_grad():
            with autocast_ctx:
                pred_tracks, pred_vis = self.model(video_tensor, queries=queries_tensor)

        tracks = pred_tracks[0].detach().cpu().numpy()
        vis = pred_vis[0].detach().cpu().numpy()

        del video_tensor, queries_tensor, pred_tracks, pred_vis
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return tracks, vis

    def run_inference(self, frames_bgr, init_kpts, ref_idx: int):
        """在采样帧序列上执行双向离线跟踪。"""
        num_frames = len(frames_bgr)
        chunk_size = int(self.cfg.cotracker_chunk_size)
        if num_frames < 2:
            raise ValueError("CoTracker requires at least two frames")
        if chunk_size < 2:
            raise ValueError("cotracker_chunk_size must be at least 2")
        if not init_kpts:
            raise ValueError("CoTracker requires at least one keypoint")

        if ref_idx < 0:
            ref_idx = num_frames + ref_idx
        ref_idx = max(0, min(ref_idx, num_frames - 1))

        all_frames_sq = []
        lb_meta = None
        for img in tqdm(
            frames_bgr,
            desc="CoTracker prepare",
            unit="frame",
            dynamic_ncols=True,
        ):
            sq, meta = self._letterbox(img)
            all_frames_sq.append(sq)
            if lb_meta is None:
                lb_meta = meta

        s, px, py = lb_meta["scale"], lb_meta["pad_l"], lb_meta["pad_t"]
        init_queries_sq = [[x * s + px, y * s + py] for (x, y) in init_kpts]

        final_tracks_sq = np.zeros((num_frames, len(init_kpts), 2), dtype=np.float32)
        final_vis = np.zeros((num_frames, len(init_kpts)), dtype=np.float32)

        print(f"║ [CoTracker] Tracking Forward from frame {ref_idx}...", flush=True)
        curr_queries = init_queries_sq
        forward_starts = range(ref_idx, num_frames - 1, chunk_size - 1)
        for start_f in tqdm(
            forward_starts,
            desc="CoTracker forward",
            unit="chunk",
            dynamic_ncols=True,
        ):
            end_f = min(start_f + chunk_size, num_frames)
            chunk_frames = all_frames_sq[start_f:end_f]

            t_chunk, v_chunk = self._process_single_chunk(chunk_frames, curr_queries)

            final_tracks_sq[start_f:end_f] = t_chunk
            final_vis[start_f:end_f] = v_chunk
            curr_queries = t_chunk[-1].tolist()

        if ref_idx > 0:
            print(f"║ [CoTracker] Tracking Backward from frame {ref_idx}...", flush=True)
            backward_indices = list(range(ref_idx, -1, -1))
            curr_queries = init_queries_sq

            backward_starts = range(0, len(backward_indices) - 1, chunk_size - 1)
            for i in tqdm(
                backward_starts,
                desc="CoTracker backward",
                unit="chunk",
                dynamic_ncols=True,
            ):
                idx_chunk = backward_indices[i : i + chunk_size]
                chunk_frames = [all_frames_sq[idx] for idx in idx_chunk]

                t_chunk, v_chunk = self._process_single_chunk(chunk_frames, curr_queries)

                for local_i, global_idx in enumerate(idx_chunk):
                    final_tracks_sq[global_idx] = t_chunk[local_i]
                    final_vis[global_idx] = v_chunk[local_i]

                curr_queries = t_chunk[-1].tolist()

        tracks = self._unletterbox_points(final_tracks_sq, lb_meta)
        return tracks, final_vis

    def cleanup(self):
        """释放模型和显存。"""
        if hasattr(self, "model"):
            self.model.to("cpu")
            del self.model
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()


class CoTrackerVisualizer:
    """渲染 CoTracker 点轨迹和简单 HUD。"""

    def __init__(self):
        self.colors = [
            (52, 152, 219),
            (46, 204, 113),
            (231, 76, 60),
            (241, 196, 15),
            (155, 89, 182),
        ]

    def render_frame(self, img_bgr, tracks, vis, frame_idx, trail_len):
        canvas = img_bgr.copy()
        _, n_points, _ = tracks.shape

        draw_glass_rect(
            canvas,
            (canvas.shape[1] - 200, 10),
            (canvas.shape[1] - 10, 80),
        )
        cv2.putText(
            canvas,
            "COTRACKER",
            (canvas.shape[1] - 190, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 215, 0),
            1,
            cv2.LINE_AA,
        )

        visible_cnt = np.sum(vis[frame_idx] > 0)
        cv2.putText(
            canvas,
            f"Points: {visible_cnt}/{n_points}",
            (canvas.shape[1] - 190, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        for point_idx in range(n_points):
            if vis[frame_idx, point_idx] == 0:
                continue

            col = self.colors[point_idx % len(self.colors)]
            c_bgr = (col[2], col[1], col[0])

            start_t = max(0, frame_idx - trail_len)
            pts = []
            for t in range(start_t, frame_idx + 1):
                if vis[t, point_idx] > 0:
                    pts.append(tracks[t, point_idx].astype(np.int32))
                else:
                    if len(pts) > 1:
                        cv2.polylines(canvas, [np.array(pts)], False, c_bgr, 2, cv2.LINE_AA)
                    pts = []
            if len(pts) > 1:
                cv2.polylines(canvas, [np.array(pts)], False, c_bgr, 2, cv2.LINE_AA)

            x, y = int(tracks[frame_idx, point_idx, 0]), int(tracks[frame_idx, point_idx, 1])
            cv2.circle(canvas, (x, y), 4, c_bgr, -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 6, (255, 255, 255), 1, cv2.LINE_AA)

        return canvas


class CoTracker:
    """Generator 调用的 CoTracker 模块类。"""

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else CoTrackerConfig()
        self.visualizer = CoTrackerVisualizer()

    def track(self, frames_bgr, frame_indices, keypoints_by_object, ref_idx: int):
        """跟踪多个 object 的 2D keypoints。"""
        init_kpts = []
        object_slices = {}
        cursor = 0
        for object_key, points in keypoints_by_object.items():
            pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
            init_kpts.extend(pts.tolist())
            object_slices[object_key] = (cursor, cursor + len(pts))
            cursor += len(pts)

        engine = CoTrackerEngine(self.cfg)
        start_time = time.perf_counter()
        try:
            tracks, visibility = engine.run_inference(frames_bgr, init_kpts, ref_idx)
        finally:
            engine.cleanup()
        elapsed = time.perf_counter() - start_time

        objects = {}
        for object_key, (start, end) in object_slices.items():
            objects[object_key] = {
                "tracks": tracks[:, start:end, :].tolist(),
                "visibility": visibility[:, start:end].astype(float).tolist(),
                "source_points": np.asarray(
                    keypoints_by_object[object_key],
                    dtype=np.float32,
                ).tolist(),
            }

        vis_frames = []
        for frame_idx, frame in tqdm(
            enumerate(frames_bgr),
            total=len(frames_bgr),
            desc="CoTracker visualization",
            unit="frame",
            dynamic_ncols=True,
        ):
            vis_frames.append(
                self.visualizer.render_frame(
                    frame,
                    tracks,
                    visibility,
                    frame_idx,
                    int(self.cfg.cotracker_viz_trail_len),
                )
            )
        document = {
            "method": "cotracker3_offline",
            "ref_sequence_index": int(ref_idx),
            "ref_frame": int(frame_indices[ref_idx]),
            "frames": [int(item) for item in frame_indices],
            "objects": objects,
            "inference_seconds": elapsed,
            "num_points": len(init_kpts),
        }
        return document, vis_frames
