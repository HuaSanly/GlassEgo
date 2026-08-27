"""Arm-mask inpainting for the unified preprocessing artifact layout."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from utils.utils_media import create_video_from_frames
from utils.utils_artifact_store import FrameArtifactStore


class LaMaEngine:
    """Thin ONNX Runtime wrapper kept lazy so non-LaMa stages stay importable."""

    def __init__(self, cfg):
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "LaMa requires onnxruntime and huggingface_hub"
            ) from exc

        model_path = hf_hub_download(
            repo_id=str(cfg.model.repo_id),
            filename=str(cfg.model.filename),
        )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_size = int(cfg.input_size)
        self.mask_dilation = int(cfg.mask_dilation)
        if self.input_size <= 0:
            raise ValueError("lama.input_size must be positive")
        if self.mask_dilation <= 0 or self.mask_dilation % 2 == 0:
            raise ValueError("lama.mask_dilation must be a positive odd integer")

    def inpaint(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("LaMa expects a BGR image with shape HxWx3")
        if mask is None or mask.shape[:2] != image_bgr.shape[:2]:
            raise ValueError("LaMa mask must have the same spatial shape as the image")

        height, width = image_bgr.shape[:2]
        kernel = np.ones((self.mask_dilation, self.mask_dilation), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        image_input = cv2.resize(
            image_bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA
        ).astype(np.float32) / 255.0
        mask_input = cv2.resize(
            mask,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)
        inputs = {
            self.session.get_inputs()[0].name: image_input.transpose(2, 0, 1)[None],
            self.session.get_inputs()[1].name: (mask_input > 127.5).astype(np.float32)[
                None, None
            ],
        }
        output = np.asarray(self.session.run(None, inputs)[0]).squeeze(0)
        output = output.transpose(1, 2, 0)
        if output.max() <= 1.1:
            output *= 255.0
        output = np.clip(output, 0, 255).astype(np.uint8)
        output = cv2.resize(output, (width, height), interpolation=cv2.INTER_LINEAR)
        blend_mask = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (5, 5), 0)[..., None]
        return np.asarray(
            output * blend_mask + image_bgr * (1.0 - blend_mask), dtype=np.uint8
        )


class LaMaGenerator:
    """Generate ``rgb_WoArm.png`` for context and training frames."""

    def __init__(
        self,
        unit_dir: str | Path,
        cfg,
        store: FrameArtifactStore | None = None,
        engine: LaMaEngine | None = None,
    ):
        self.unit_dir = Path(unit_dir).expanduser().resolve()
        self.cfg = cfg
        self.store = store or FrameArtifactStore(self.unit_dir)
        self.engine = engine

    def run(
        self,
        frame_images: dict[int, np.ndarray],
        training_frames: set[int],
        context_frames: set[int],
        fps: float = 30.0,
    ) -> dict:
        if not bool(getattr(self.cfg, "enabled", True)):
            return {"status": "disabled", "frames": 0, "path": None}

        frame_indices = sorted(set(training_frames) | set(context_frames))
        if not frame_indices:
            raise ValueError("LaMa requires at least one frame")
        if self.engine is None:
            self.engine = LaMaEngine(self.cfg)

        outputs = []
        visual_frames = []
        for frame_idx in frame_indices:
            image = frame_images.get(int(frame_idx))
            if image is None:
                raise FileNotFoundError(f"Missing RGB frame for LaMa: {frame_idx}")
            is_training = int(frame_idx) in training_frames
            frame_dir = self.store.frame_dir(frame_idx, is_training)
            image_path = frame_dir / "rgb.png"
            mask_path = frame_dir / "mask_arm.png"
            if not image_path.is_file():
                self.store.write_image(frame_idx, "rgb.png", image, is_training)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Missing arm mask for LaMa: {mask_path}")
            result = self.engine.inpaint(image, mask)
            output_path = self.store.write_image(
                frame_idx, "rgb_WoArm.png", result, is_training
            )
            outputs.append(self.store.relative_path(output_path))
            visual_frames.append(np.hstack((image, result)))

        video_path = self.store.vis_dir / "lama_vis.mp4"
        output_cfg = getattr(self.cfg, "output", None)
        if bool(getattr(output_cfg, "export_video", False)):
            create_video_from_frames(
                visual_frames,
                video_path,
                fps=float(fps),
                export_gif=bool(getattr(output_cfg, "export_gif", False)),
                ratio=int(getattr(output_cfg, "gif_frame_ratio", 10)),
                export_video=True,
            )
        report = {
            "status": "completed",
            "frames": len(outputs),
            "training_frames": len(training_frames),
            "context_frames": len(context_frames),
            "outputs": outputs,
            "video": self.store.relative_path(video_path)
            if video_path.is_file()
            else None,
        }
        report_path = self.store.module_vis_dir("lama") / "report.json"
        self._atomic_write_json(report_path, report)
        return {**report, "path": self.store.relative_path(report_path)}

    @staticmethod
    def _atomic_write_json(path: Path, document: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        )
        temporary_path = Path(handle.name)
        try:
            with handle:
                json.dump(document, handle, indent=2, ensure_ascii=True)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
