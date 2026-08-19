# -*- coding: utf-8 -*-
# @FileName: DINOSAM.py

"""
====================================================================================================
Project Aria DINO-SAM2 Segmentation Pipeline (DINOSAM.py)
====================================================================================================

Description:
    This script processes RGB frames using Grounding DINO for object detection and SAM2
    for mask generation. It supports multi-object prompts and generates combined masks
    for downstream tasks.

Technical Specifics:
    - Grounding DINO: Text-to-Box detection.
    - SAM2: Box-to-Mask segmentation.
====================================================================================================
"""

import gc
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from utils.utils_vis import (
    C_CYAN,
    C_GOLD,
    C_GRAY,
    C_GREEN,
    C_WHITE,
    draw_glass_rect,
    draw_status_bar,
)


@dataclass(frozen=True)
class DINOSAMConfig:
    """DINO-SAM2 默认运行参数。"""

    # Grounding DINO 负责根据文本 prompt 找 bbox。
    dino_model_id: str = "IDEA-Research/grounding-dino-base"
    # SAM2 负责把 DINO 给出的 bbox 进一步细化成像素级 mask。
    sam2_repo_id: str = "facebook/sam2-hiera-large"
    sam2_checkpoint_name: str = "sam2_hiera_large.pt"
    sam2_config: str = "sam2_hiera_l.yaml"
    # DINO bbox 置信度阈值；低于该阈值的候选框不会交给 SAM2。
    box_threshold: float = 0.3


class DINOSAMEngine:
    """DINO 与 SAM2 的常驻模型封装。

    这个类只负责模型生命周期和单个 prompt 的核心推理，不关心视频、
    数据单元目录或结果落盘布局。上层 `DINOSAM` 会复用同一个 engine，
    避免每帧重复加载模型。
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 首次运行时 HuggingFace 会下载权重；之后从本机缓存加载。
        print(f"║ [System] Initializing Models on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.cfg.dino_model_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.cfg.dino_model_id
        ).to(self.device)

        ckpt_path = hf_hub_download(
            repo_id=self.cfg.sam2_repo_id,
            filename=self.cfg.sam2_checkpoint_name,
        )
        self.predictor = SAM2ImagePredictor(
            build_sam2(self.cfg.sam2_config, ckpt_path, device=self.device)
        )

    def predict_frame_internal(self, image_np, text_prompt):
        """在已经 set_image 的当前帧上，对单个 prompt 做 DINO+SAM2。

        输入约定：`image_np` 是 OpenCV BGR `uint8` 图像。
        输出约定：
            mask: uint8[H,W]，取值 0/255；无检测时为 None。
            boxes: float[N,4]，像素坐标 xyxy；无检测时为 None。
        """
        # Grounding DINO 使用 PIL RGB；仓库内视频流和可视化统一用 OpenCV BGR。
        image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        W, H = image_pil.size

        # DINO 输出归一化 cxcywh boxes 和每个文本 token 的 logits。
        inputs = self.processor(
            images=image_pil,
            text=text_prompt,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)

        logits = outputs.logits.sigmoid()[0]
        boxes = outputs.pred_boxes[0]

        # 取每个候选框的最高 token 置信度，并用 box_threshold 过滤。
        mask_filter = logits.max(-1)[0] > self.cfg.box_threshold
        filtered_logits = logits[mask_filter]
        filtered_boxes = boxes[mask_filter]

        if len(filtered_boxes) == 0:
            return None, 0.0, None, None

        confidences = filtered_logits.max(-1)[0].cpu().numpy()
        avg_conf = np.mean(confidences)

        # DINO 的 box 是归一化 cx,cy,w,h；SAM2 需要像素 xyxy。
        pixel_boxes = filtered_boxes * torch.Tensor([W, H, W, H]).to(self.device)
        cx, cy, w, h = pixel_boxes.unbind(-1)
        x1, y1 = cx - 0.5 * w, cy - 0.5 * h
        x2, y2 = cx + 0.5 * w, cy + 0.5 * h
        input_boxes = torch.stack([x1, y1, x2, y2], dim=-1).cpu().numpy()

        # predictor.set_image(img) 在外层只调用一次；这里对不同 prompt 复用同一帧 embedding。
        masks, _, _ = self.predictor.predict(box=input_boxes, multimask_output=False)

        # 一个 prompt 可能对应多个 box/mask，这里合并为该 prompt 的单张二值 mask。
        combined_mask = np.any(masks.squeeze(), axis=0) if masks.ndim > 3 else masks.squeeze()
        if combined_mask.ndim > 2:
            combined_mask = np.any(combined_mask, axis=0)

        return (combined_mask.astype(np.uint8) * 255), avg_conf, input_boxes, confidences

    def cleanup(self):
        """Release resources."""
        print("║ [Cleanup] Releasing Engine Resources...")
        if hasattr(self, "dino_model"):
            self.dino_model.to("cpu")
            del self.dino_model
        if hasattr(self, "predictor"):
            self.predictor.model.to("cpu")
            del self.predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class DINOSAM:
    """DINO-SAM 单帧/多 prompt 推理管理器。

    这一层保留 HumanEgo 的使用方式：一张图上可以跑多个 prompt，
    每个 prompt 单独保存 mask，同时生成一张所有 prompt 的合并 mask。
    """

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else DINOSAMConfig()
        self.engine = DINOSAMEngine(self.cfg)

    def _render_vis(self, img, mask, boxes, box_confs, avg_conf, latency, text_prompt):
        """渲染调试图：左侧原图+bbox，右侧当前累计 mask 和状态面板。"""
        left_vis = img.copy()
        num_objects = 0
        if boxes is not None:
            num_objects = len(boxes)
            for box, b_conf in zip(boxes, box_confs):
                bx1, by1, bx2, by2 = box.astype(int)
                cv2.rectangle(left_vis, (bx1, by1), (bx2, by2), C_GREEN, 2)
                cv2.putText(
                    left_vis,
                    f"{b_conf:.2f}",
                    (bx1, by1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    C_GREEN,
                    1,
                    cv2.LINE_AA,
                )

        mask_vis = (
            mask
            if mask is not None
            else np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        )
        heatmap_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

        draw_glass_rect(heatmap_vis, (10, 10), (350, 240))
        font = cv2.FONT_HERSHEY_SIMPLEX

        header_col = C_GREEN if avg_conf > self.cfg.box_threshold else C_GRAY
        cv2.putText(
            heatmap_vis,
            "DINO-SAM2 ANALYZER",
            (20, 30),
            font,
            0.5,
            C_GOLD,
            1,
            cv2.LINE_AA,
        )
        status = "[SIGNAL ACTIVE]" if avg_conf > 0 else "[SEARCHING...]"
        cv2.putText(
            heatmap_vis,
            status,
            (20, 50),
            font,
            0.4,
            header_col,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            heatmap_vis,
            f"OBJECTS: {num_objects}",
            (20, 120),
            font,
            0.5,
            C_GOLD,
            1,
            cv2.LINE_AA,
        )

        draw_status_bar(
            heatmap_vis,
            (20, 170),
            300,
            avg_conf,
            1.0,
            f"Conf: {avg_conf:.2f}",
            C_CYAN,
        )
        cv2.putText(
            heatmap_vis,
            f"LATENCY: {latency * 1000:.1f}ms",
            (20, 210),
            font,
            0.4,
            C_WHITE,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            heatmap_vis,
            f"PROMPT: {text_prompt[:20]}...",
            (20, 225),
            font,
            0.3,
            C_GRAY,
            1,
            cv2.LINE_AA,
        )

        return cv2.hconcat([left_vis, heatmap_vis])

    def process_single(self, img, prompt, save_path=None):
        """处理单图单 prompt。

        该函数主要用于快速调试；正式视频入口会走 `process_and_save()`。
        """
        if isinstance(img, str):
            image_np = cv2.imread(img)
        else:
            image_np = img

        if image_np is None:
            return None

        self.engine.predictor.set_image(image_np)
        t_start = time.perf_counter()
        mask, avg_conf, boxes, box_confs = self.engine.predict_frame_internal(
            image_np,
            prompt,
        )

        if self.engine.device == "cuda":
            torch.cuda.synchronize()
        latency = time.perf_counter() - t_start

        mask_out = mask if mask is not None else np.zeros(image_np.shape[:2], dtype=np.uint8)
        if save_path:
            cv2.imwrite(save_path, mask_out)
        return mask_out

    def process_and_save(self, image, prompts_dict, output_dir):
        """处理单图多 prompt，并把 mask 写入指定帧目录。

        输出文件：
            mask_<key>.png: 每个 prompt 一张独立 mask。
            mask_arm_and_obj.png: 所有 prompt 的并集 mask，沿用 HumanEgo 命名。
        返回值供 Generator 组装 JSON 和保存可视化。
        """
        if isinstance(image, (str, os.PathLike, Path)):
            img = cv2.imread(str(image))
        else:
            img = image
        if img is None:
            return None, 0, []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        combined_all_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        self.engine.predictor.set_image(img)
        last_vis = None
        prompts_count = 0
        results = []

        for key, prompt in prompts_dict.items():
            # key 只作为稳定对象 ID 使用，例如 obj1/obj2/arm；prompt 才给 DINO。
            if not prompt.strip():
                continue
            prompts_count += 1

            t_start = time.perf_counter()
            mask, avg_conf, boxes, box_confs = self.engine.predict_frame_internal(
                img,
                prompt,
            )

            if self.engine.device == "cuda":
                torch.cuda.synchronize()
            latency = time.perf_counter() - t_start

            mask_out = (
                mask
                if mask is not None
                else np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            )
            mask_path = output_dir / f"mask_{key}.png"
            cv2.imwrite(str(mask_path), mask_out)
            combined_all_mask = cv2.bitwise_or(combined_all_mask, mask_out)

            # 统一成空数组，避免 JSON 组装阶段处理 None 分支。
            boxes_out = (
                boxes.astype(np.float32)
                if boxes is not None
                else np.empty((0, 4), dtype=np.float32)
            )
            confs_out = (
                box_confs.astype(np.float32)
                if box_confs is not None
                else np.empty((0,), dtype=np.float32)
            )
            results.append(
                {
                    "key": key,
                    "prompt": prompt,
                    "confidence": float(avg_conf),
                    "boxes": boxes_out,
                    "confidences": confs_out,
                    "mask_path": mask_path,
                }
            )

            last_vis = self._render_vis(
                img,
                combined_all_mask,
                boxes,
                box_confs,
                avg_conf,
                latency,
                prompt,
            )

        combined_path = output_dir / "mask_arm_and_obj.png"
        cv2.imwrite(str(combined_path), combined_all_mask)
        return last_vis, prompts_count, results

    def cleanup(self):
        self.engine.cleanup()
