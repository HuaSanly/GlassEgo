"""Stereo disparity and metric depth for rectified :class:`StereoFrame` data."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Mapping

import cv2
import numpy as np

from .stereo_types import StereoFrame


@dataclass(frozen=True)
class StereoDepthResult:
    disparity_px: np.ndarray
    depth_m: np.ndarray
    valid_mask: np.ndarray


class StereoDepthEstimator:
    """Estimate depth in the left rectified optical frame using StereoSGBM."""

    def __init__(
        self,
        cfg: Mapping | None = None,
        logger: logging.Logger | None = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.cfg = dict(cfg or {})
        self.min_disparity = float(self.cfg.get("min_disparity", 0.0))
        self.num_disparities = int(self.cfg.get("num_disparities", 128))
        self.block_size = int(self.cfg.get("block_size", 5))
        self.min_depth_m = float(self.cfg.get("min_depth_m", 0.1))
        self.max_depth_m = float(self.cfg.get("max_depth_m", 5.0))
        self.lr_max_error_px = float(self.cfg.get("lr_max_error_px", 1.0))
        self.uniqueness_ratio = int(self.cfg.get("uniqueness_ratio", 10))
        self.speckle_window_size = int(self.cfg.get("speckle_window_size", 100))
        self.speckle_range = int(self.cfg.get("speckle_range", 2))
        if self.num_disparities <= 0 or self.num_disparities % 16:
            raise ValueError("stereo.num_disparities must be a positive multiple of 16")
        if self.block_size <= 0 or self.block_size % 2 == 0:
            raise ValueError("stereo.block_size must be a positive odd integer")
        if self.min_depth_m <= 0 or self.max_depth_m <= self.min_depth_m:
            raise ValueError("stereo depth bounds are invalid")
        if self.lr_max_error_px < 0:
            raise ValueError("stereo.lr_max_error_px must be non-negative")

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("stereo images must have shape HxWx3")
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _projections(frame: StereoFrame) -> tuple[np.ndarray, np.ndarray]:
        p_left = np.asarray(frame.left_projection, dtype=np.float64)
        p_right = np.asarray(frame.right_projection, dtype=np.float64)
        if p_left.shape != (3, 4) or p_right.shape != (3, 4):
            raise ValueError("StereoFrame projections must be 3x4 matrices")
        for matrix in (p_left, p_right):
            if not np.all(np.isfinite(matrix)):
                raise ValueError("StereoFrame projections must be finite")
        if p_left[0, 0] <= 0 or p_left[1, 1] <= 0:
            raise ValueError("left projection has invalid focal lengths")
        if p_right[0, 0] <= 0 or p_right[1, 1] <= 0:
            raise ValueError("right projection has invalid focal lengths")
        return p_left, p_right

    def _matcher(self, min_disparity: int) -> cv2.StereoSGBM:
        channels = 1
        block = self.block_size
        return cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=self.num_disparities,
            blockSize=block,
            P1=8 * channels * block * block,
            P2=32 * channels * block * block,
            disp12MaxDiff=-1,
            uniquenessRatio=self.uniqueness_ratio,
            speckleWindowSize=self.speckle_window_size,
            speckleRange=self.speckle_range,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def _q_matrix(
        self,
        p_left: np.ndarray,
        p_right: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        fx_left = float(p_left[0, 0])
        fx_right = float(p_right[0, 0])
        if not np.isclose(fx_left, fx_right, rtol=1e-3, atol=1e-3):
            raise ValueError("rectified stereo projections must share focal length")
        tx = float(p_right[0, 3] / fx_right)
        if abs(tx) <= 1e-9:
            raise ValueError("right projection does not contain a stereo baseline")
        cx_left = float(p_left[0, 2])
        cx_right = float(p_right[0, 2])
        cy_left = float(p_left[1, 2])
        q = np.array(
            [
                [1.0, 0.0, 0.0, -cx_left],
                [0.0, 1.0, 0.0, -cy_left],
                [0.0, 0.0, 0.0, fx_left],
                [0.0, 0.0, -1.0 / tx, (cx_left - cx_right) / tx],
            ],
            dtype=np.float64,
        )
        return q, cx_left - cx_right, tx

    def estimate(self, frame: StereoFrame) -> StereoDepthResult:
        if not isinstance(frame, StereoFrame):
            raise TypeError("frame must be a StereoFrame")
        if frame.left.shape[:2] != frame.right.shape[:2]:
            raise ValueError("left and right stereo images must have equal shape")
        p_left, p_right = self._projections(frame)
        q, principal_offset, _ = self._q_matrix(p_left, p_right)
        left_gray = self._gray(frame.left)
        right_gray = self._gray(frame.right)

        # SGBM's pixel disparity includes the principal-point offset caused by
        # different crop origins; the configured range is the physical range.
        matcher_min = int(round(self.min_disparity + principal_offset))
        left_raw = self._matcher(matcher_min).compute(left_gray, right_gray)
        left_disp = left_raw.astype(np.float32) / 16.0

        valid = np.isfinite(left_disp)
        effective_disp = left_disp - principal_offset
        valid &= effective_disp >= self.min_disparity

        if self.lr_max_error_px > 0:
            right_min = -matcher_min - self.num_disparities
            right_raw = self._matcher(right_min).compute(right_gray, left_gray)
            right_disp = right_raw.astype(np.float32) / 16.0
            height, width = left_disp.shape
            rows, cols = np.indices((height, width))
            right_cols = np.rint(cols - left_disp).astype(np.int32)
            inside = (right_cols >= 0) & (right_cols < width)
            sampled = np.full_like(left_disp, np.nan, dtype=np.float32)
            safe_rows = rows[inside]
            safe_cols = right_cols[inside]
            sampled[inside] = right_disp[safe_rows, safe_cols]
            finite_sampled = np.isfinite(sampled)
            with np.errstate(invalid="ignore"):
                consistent = np.abs(left_disp + sampled) <= self.lr_max_error_px
            valid &= finite_sampled & consistent

        disparity = left_disp.astype(np.float32, copy=True)
        disparity[~valid] = np.nan

        # reprojectImageTo3D uses Q's signed baseline and handles cx_left-cx_right.
        safe_disparity = np.nan_to_num(disparity, nan=0.0)
        points = cv2.reprojectImageTo3D(safe_disparity, q).astype(np.float32)
        depth = points[:, :, 2]
        valid &= np.isfinite(depth)
        valid &= depth >= self.min_depth_m
        valid &= depth <= self.max_depth_m
        disparity[~valid] = np.nan
        depth = depth.astype(np.float32, copy=False)
        depth[~valid] = np.nan
        return StereoDepthResult(
            disparity_px=disparity,
            depth_m=depth,
            valid_mask=valid,
        )

    def points_from_mask(
        self,
        frame: StereoFrame,
        mask: np.ndarray,
        result: StereoDepthResult | None = None,
    ) -> np.ndarray:
        if not isinstance(frame, StereoFrame):
            raise TypeError("frame must be a StereoFrame")
        mask = np.asarray(mask).astype(bool, copy=False)
        if mask.shape != frame.left.shape[:2]:
            raise ValueError("mask must match StereoFrame.left spatial shape")
        result = result if result is not None else self.estimate(frame)
        if result.depth_m.shape != mask.shape or result.valid_mask.shape != mask.shape:
            raise ValueError("depth result must match the frame spatial shape")
        p_left, _ = self._projections(frame)
        valid = mask & result.valid_mask & np.isfinite(result.depth_m)
        rows, cols = np.nonzero(valid)
        if rows.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        z = result.depth_m[rows, cols]
        x = (cols.astype(np.float32) - float(p_left[0, 2])) * z / float(p_left[0, 0])
        y = (rows.astype(np.float32) - float(p_left[1, 2])) * z / float(p_left[1, 1])
        return np.column_stack((x, y, z)).astype(np.float32)


__all__ = [
    "StereoDepthEstimator",
    "StereoDepthResult",
]
