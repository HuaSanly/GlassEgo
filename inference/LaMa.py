"""In-memory LaMa inpainting adapter for stereo perception."""

from __future__ import annotations

import logging
from typing import Mapping

import cv2
import numpy as np

from .stereo_types import StereoFrame


class LaMaInpaint:
    """Run LaMa on the left image and keep all results in memory."""

    def __init__(
        self,
        cfg: Mapping | None = None,
        session=None,
        logger: logging.Logger | None = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.cfg = dict(cfg or {})
        self.input_size = int(self.cfg.get("input_size", 512))
        self.mask_dilation = int(self.cfg.get("mask_dilation", 5))
        model = self.cfg.get("model", {})
        if not isinstance(model, Mapping):
            raise TypeError("lama.model must be a mapping")
        self.repo_id = str(self.cfg.get("repo_id", model.get("repo_id", "")))
        self.filename = str(self.cfg.get("filename", model.get("filename", "")))
        if self.input_size <= 0:
            raise ValueError("lama.input_size must be positive")
        if self.mask_dilation <= 0 or self.mask_dilation % 2 == 0:
            raise ValueError("lama.mask_dilation must be a positive odd integer")

        self.session = session
        if self.session is None:
            if not self.repo_id or not self.filename:
                raise ValueError(
                    "lama.model.repo_id and lama.model.filename are required"
                )
            try:
                import onnxruntime as ort
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise RuntimeError(
                    "LaMa requires onnxruntime and huggingface_hub"
                ) from exc
            model_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.filename,
            )
            self.session = ort.InferenceSession(
                model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )

    def inpaint(self, frame: StereoFrame, mask: np.ndarray) -> np.ndarray:
        if not isinstance(frame, StereoFrame):
            raise TypeError("frame must be a StereoFrame")
        image_bgr = frame.left
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("StereoFrame.left must have shape HxWx3")
        if mask is None or np.asarray(mask).shape != image_bgr.shape[:2]:
            raise ValueError("LaMa mask must match StereoFrame.left spatial shape")

        mask_u8 = (np.asarray(mask) > 0).astype(np.uint8) * 255
        if not np.any(mask_u8):
            return image_bgr.copy()

        kernel = np.ones(
            (self.mask_dilation, self.mask_dilation),
            dtype=np.uint8,
        )
        mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
        height, width = image_bgr.shape[:2]
        image_input = cv2.resize(
            image_bgr,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32) / 255.0
        mask_input = cv2.resize(
            mask_u8,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_NEAREST,
        )
        inputs = {
            self.session.get_inputs()[0].name: image_input.transpose(2, 0, 1)[None],
            self.session.get_inputs()[1].name: (
                (mask_input > 127.5).astype(np.float32)[None, None]
            ),
        }
        output = np.asarray(self.session.run(None, inputs)[0]).squeeze(0)
        if output.ndim != 3:
            raise ValueError(f"LaMa returned unexpected output shape {output.shape}")
        output = output.transpose(1, 2, 0)
        if output.max() <= 1.1:
            output *= 255.0
        output = np.clip(output, 0, 255).astype(np.uint8)
        output = cv2.resize(output, (width, height), interpolation=cv2.INTER_LINEAR)
        blend = cv2.GaussianBlur(
            mask_u8.astype(np.float32) / 255.0,
            (5, 5),
            0,
        )[..., None]
        return np.asarray(
            output * blend + image_bgr * (1.0 - blend),
            dtype=np.uint8,
        )

    def close(self) -> None:
        self.session = None


__all__ = ["LaMaInpaint"]
