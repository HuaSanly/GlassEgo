# -*- coding: utf-8 -*-
# @FileName: KptsSelector.py

"""
====================================================================================================
Object Keypoints Selection Pipeline (KptsSelector.py)
====================================================================================================

Description:
    Extract robust, equidistant 2D keypoints from an object segmentation mask.
    The selected points are intended as CoTracker query points in the next stage.

Technical Specifics:
    - Mask cleanup: morphological closing and erosion.
    - Contour extraction: largest external object contour.
    - Keypoint sampling: deterministic, equidistant contour sampling.
====================================================================================================
"""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class KptsSelectorConfig:
    """HumanEgo KptsSelector 默认参数。"""

    kpts_patch_close: bool = True
    kpts_close_kernel: int = 7
    kpts_patch_erode: bool = True
    kpts_erode_kernel: int = 5
    kpts_erode_iters: int = 1
    kpts_use_inner_edge: bool = True
    kpts_edge_erode_kernel: int = 5
    kpts_min_edge_pixels: int = 50
    kpts_n_bands: int = 10


class KptsSelector:
    """从单张二值 mask 中选择用于跟踪的 2D 关键点。"""

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else KptsSelectorConfig()

    def _clamp_odd(self, k: int) -> int:
        """OpenCV 形态学 kernel 尺寸使用奇数更稳定。"""
        k = int(k)
        return k if (k % 2 == 1) else (k + 1)

    def _morph_kernel(self, k: int) -> np.ndarray:
        """生成椭圆形形态学 kernel。"""
        k = self._clamp_odd(k)
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    def preprocess_mask(self, mask_255: np.ndarray) -> np.ndarray:
        """对二值 mask 做基础清理：填洞、去边缘毛刺。"""
        m = (mask_255 > 127).astype(np.uint8) * 255

        if self.cfg.kpts_patch_close:
            k = self._morph_kernel(self.cfg.kpts_close_kernel)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

        if self.cfg.kpts_patch_erode:
            k = self._morph_kernel(self.cfg.kpts_erode_kernel)
            m = cv2.erode(m, k, iterations=int(self.cfg.kpts_erode_iters))
        return m

    def select_points(self, mask_raw: np.ndarray) -> List[Tuple[int, int]]:
        """沿最大外轮廓等距采样关键点。"""
        mask_pp = self.preprocess_mask(mask_raw)

        if self.cfg.kpts_use_inner_edge:
            k = self._morph_kernel(self.cfg.kpts_edge_erode_kernel)
            mask_eroded = cv2.erode(mask_pp, k, iterations=1)
            if np.count_nonzero(mask_eroded) > self.cfg.kpts_min_edge_pixels:
                mask_pp = mask_eroded

        contours, _ = cv2.findContours(
            mask_pp,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        if not contours:
            print("║ [KptsSelector] Warning: No contours found.")
            return []

        c = max(contours, key=cv2.contourArea).squeeze(1)
        if c.ndim < 2 or c.shape[0] < 10:
            print("║ [KptsSelector] Warning: Contour area too small.")
            return []

        # 以最上方点作为起点，保证同一 mask 每次输出顺序稳定。
        top_idx = np.lexsort((c[:, 0], c[:, 1]))[0]
        c = np.roll(c, -top_idx, axis=0)

        diffs = np.diff(c, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        dist_close = np.linalg.norm(c[-1] - c[0])
        dists = np.append(dists, dist_close)

        cum_dists = np.concatenate(([0], np.cumsum(dists)))
        total_len = cum_dists[-1]
        n_points = int(self.cfg.kpts_n_bands) * 2

        target_dists = np.linspace(0, total_len, n_points, endpoint=False)
        selected_points = []

        for td in target_dists:
            idx = np.searchsorted(cum_dists, td) - 1
            idx = np.clip(idx, 0, len(c) - 1)

            if idx == len(c) - 1:
                p1, p2 = c[idx], c[0]
                d1, d2 = cum_dists[idx], total_len
            else:
                p1, p2 = c[idx], c[idx + 1]
                d1, d2 = cum_dists[idx], cum_dists[idx + 1]

            if d2 == d1:
                pt = p1
            else:
                alpha = (td - d1) / (d2 - d1)
                pt = p1 + alpha * (p2 - p1)

            selected_points.append((int(round(pt[0])), int(round(pt[1]))))

        return selected_points

    def select_from_mask(
        self,
        mask_raw: np.ndarray,
        image_bgr: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """选择关键点并返回可视化图。

        Args:
            mask_raw: uint8[H,W]，0/255 二值 mask。
            image_bgr: 可选的 OpenCV BGR 背景图。

        Returns:
            points: int32[N,2]，像素坐标 xy。
            vis_bgr: BGR 可视化图。
        """
        points = self.select_points(mask_raw)
        points_array = np.asarray(points, dtype=np.int32).reshape(-1, 2)

        if image_bgr is not None:
            vis_bgr = image_bgr.copy()
        else:
            h, w = mask_raw.shape
            vis_bgr = np.zeros((h, w, 3), dtype=np.uint8)
            vis_bgr[mask_raw > 0] = (50, 50, 50)

        vis_bgr = draw_points(vis_bgr, points)
        return points_array, vis_bgr


def draw_points(img_bgr: np.ndarray, points: List[Tuple[int, int]]) -> np.ndarray:
    """在 BGR 图上画出关键点和编号。"""
    colors = [
        (52, 152, 219),
        (46, 204, 113),
        (231, 76, 60),
        (241, 196, 15),
        (155, 89, 182),
        (52, 73, 94),
        (26, 188, 156),
        (230, 126, 34),
        (149, 165, 166),
        (192, 57, 43),
        (39, 174, 96),
        (41, 128, 185),
    ]
    out = img_bgr.copy()
    for i, (x, y) in enumerate(points):
        color = colors[i % len(colors)]
        c_bgr = (color[2], color[1], color[0])

        cv2.circle(out, (x, y), 5, c_bgr, -1, cv2.LINE_AA)
        cv2.circle(out, (x, y), 7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            out,
            str(i),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_DUPLEX,
            0.6,
            c_bgr,
            2,
            cv2.LINE_AA,
        )
    return out
