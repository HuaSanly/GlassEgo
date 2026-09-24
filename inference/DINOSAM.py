"""In-memory Grounding DINO + SAM2 segmentation for stereo perception."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, Mapping

import cv2
import numpy as np

from .stereo_types import StereoFrame


@dataclass(frozen=True)
class MaskResult:
    """One prompt's binary mask and aggregate detection confidence."""

    mask: np.ndarray
    confidence: float


class DinoSamSegmenter:
    """Run DINO-SAM2 on the left image of a :class:`StereoFrame`.

    Heavy model dependencies are imported lazily so the rest of the perception
    package can be tested with a fake segmenter on machines without SAM2.
    """

    def __init__(
        self,
        cfg: Mapping | None = None,
        logger: logging.Logger | None = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.cfg = dict(cfg or {})
        self.device = None
        self.processor = None
        self.dino_model = None
        self.predictor = None
        self._torch = None
        self._pil_image = None
        self._closed = False

        try:
            import torch
            from huggingface_hub import hf_hub_download
            from PIL import Image
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )

            self._torch = torch
            self._pil_image = Image
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            dino_model_id = self.cfg.get("dino_model_id", "IDEA-Research/grounding-dino-base")
            self.processor = AutoProcessor.from_pretrained(dino_model_id)
            self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                dino_model_id
            ).to(self.device)
            self.dino_model.eval()

            checkpoint = hf_hub_download(
                repo_id=self.cfg.get("sam2_repo_id", "facebook/sam2-hiera-large"),
                filename=self.cfg.get("sam2_checkpoint_name", "sam2_hiera_large.pt"),
            )
            sam_model = build_sam2(
                self.cfg.get("sam2_config", "sam2_hiera_l.yaml"),
                checkpoint,
                device=self.device,
            )
            self.predictor = SAM2ImagePredictor(sam_model)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _combined_mask(masks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        values = np.asarray(masks)
        if values.ndim < 2:
            return np.zeros(shape, dtype=bool)
        if values.ndim > 2:
            values = np.any(values, axis=tuple(range(values.ndim - 2)))
        if values.shape != shape:
            raise ValueError(
                f"SAM2 returned mask shape {values.shape}, expected {shape}"
            )
        return values.astype(bool, copy=False)

    def _predict_prompt(
        self,
        image_bgr: np.ndarray,
        prompt: str,
    ) -> tuple[np.ndarray, float]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = self._pil_image.fromarray(image_rgb)
        width, height = image_pil.size

        inputs = self.processor(
            images=image_pil,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)
        with self._torch.no_grad():
            outputs = self.dino_model(**inputs)

        logits = outputs.logits.sigmoid()[0]
        boxes = outputs.pred_boxes[0]
        selected = logits.max(-1).values > float(self.cfg.get("box_threshold", 0.3))
        if not bool(selected.any()):
            return np.zeros(image_bgr.shape[:2], dtype=bool), 0.0

        filtered_logits = logits[selected]
        filtered_boxes = boxes[selected]
        confidences = filtered_logits.max(-1).values.detach().cpu().numpy()
        pixel_boxes = filtered_boxes * self._torch.tensor(
            [width, height, width, height],
            device=self.device,
        )
        cx, cy, box_width, box_height = pixel_boxes.unbind(-1)
        input_boxes = self._torch.stack(
            [
                cx - 0.5 * box_width,
                cy - 0.5 * box_height,
                cx + 0.5 * box_width,
                cy + 0.5 * box_height,
            ],
            dim=-1,
        ).detach().cpu().numpy()

        masks, _, _ = self.predictor.predict(
            box=input_boxes,
            multimask_output=False,
        )
        mask = self._combined_mask(masks, image_bgr.shape[:2])
        return mask, float(np.mean(confidences))

    def segment(
        self,
        frame: StereoFrame,
        prompts: Dict[str, str],
    ) -> Dict[str, MaskResult]:
        """Segment all non-empty prompts on ``frame.left`` in one image pass."""
        if not isinstance(frame, StereoFrame):
            raise TypeError("frame must be a StereoFrame")
        image = frame.left
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("StereoFrame.left must have shape HxWx3")

        results: Dict[str, MaskResult] = {}
        if not prompts:
            return results

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)
        try:
            for key, prompt in prompts.items():
                prompt = str(prompt).strip()
                if not prompt:
                    continue
                mask, confidence = self._predict_prompt(image, prompt)
                results[str(key)] = MaskResult(mask=mask, confidence=confidence)
        finally:
            self.predictor.reset_predictor()
        return results

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        predictor = self.predictor
        self.predictor = None
        if predictor is not None:
            try:
                predictor.reset_predictor()
                predictor.model.to("cpu")
            except (AttributeError, RuntimeError):
                pass
        model = self.dino_model
        self.dino_model = None
        if model is not None:
            try:
                model.to("cpu")
            except (AttributeError, RuntimeError):
                pass
        self.processor = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


__all__ = ["DinoSamSegmenter", "MaskResult"]
