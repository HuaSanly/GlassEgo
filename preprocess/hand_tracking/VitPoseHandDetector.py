
import gc
import os
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from easy_ViTPose import VitInference


VITPOSE_AVAILABLE = False

# COCO-WholeBody 中的手部关键点索引范围（133 个关键点）
LEFT_HAND_SLICE = slice(91, 112)   # 21个要点
RIGHT_HAND_SLICE = slice(112, 133)  # 21个要点

HF_REPO = "JunkyByte/easy_ViTPose"
# 喜欢大的，退回到较小的版本
VITPOSE_VARIANTS = [
    ("h", "torch/wholebody/vitpose-h-wholebody.pth"),
    ("l", "torch/wholebody/vitpose-l-wholebody.pth"),
    ("b", "torch/wholebody/vitpose-b-wholebody.pth"),
    ("s", "torch/wholebody/vitpose-s-wholebody.pth"),
]
YOLO_HF_PATH = "yolov8/yolov8s.pt"

class VitPoseHandDetector:
    """
    通过 ViTPose 全身关键点进行手部检测（官方 HaMeR 演示方法）。
    Pipeline: YOLO 身体检测 → ViTPose 全身 (133 kpts) → 来自手部关键点的手部 bbox。

    这与使用 ViTDet+ViTPose 的官方 HaMeR demo.py 相匹配。
    我们使用 easy_ViTPose （无 mmpose/detectron2 依赖性）和 YOLO 进行人体检测。

    COCO-WholeBody 关键点指数：
      0-16：身体 (17)
      17-22：脚 (6)
      23-90：脸 (68)
      91-111：左手 (21)
      112-132：右手 (21)
    """
    def __init__(self, cfg, device: str = "cuda"):
        self.cfg = cfg
        self.last_whole_image_fallback = False
        vitpose_model = None
        model_name = None
        for variant, hf_path in VITPOSE_VARIANTS:
            try:
                vitpose_model = hf_hub_download(
                    repo_id=HF_REPO, filename=hf_path,
                )
                model_name = variant
                break
            except Exception:
                continue

        if vitpose_model is None:
            raise FileNotFoundError(
                f"Could not download any ViTPose wholebody model from {HF_REPO}. "
                "Check your internet connection or install manually."
            )
        # 下载YOLOv8s进行人体检测
        yolo_model = hf_hub_download(
            repo_id=HF_REPO, filename=YOLO_HF_PATH,
        )

        self.model = VitInference(
            model=vitpose_model,
            yolo=yolo_model,
            model_name=model_name,
            dataset="wholebody",
            device=device,
        )
        self._model_name = model_name
        print(f"[ViTPose] Loaded ViTPose-{model_name.upper()} wholebody + YOLOv8s (from HuggingFace Hub)")
        VitPoseHandDetector.VITPOSE_AVAILABLE = True

    def cleanup(self) -> None:
        """释放 ViTPose 和 YOLO 模型。"""
        model = getattr(self, "model", None)
        self.model = None
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def detect(self, img_rgb: np.ndarray) -> list:

        h_img, w_img = img_rgb.shape[:2]
        kpt_conf_thr = float(self.cfg.kpt_conf_threshold)
        min_valid_kpts = int(self.cfg.min_valid_kpts)
        bbox_pad_ratio = float(self.cfg.bbox_pad_ratio)
        whole_image_fallback = False
        keypoints = self.model.inference(img_rgb)  # {person_id: (133, 3)}
        if len(keypoints) == 0:
            whole_image_fallback = True
            from easy_ViTPose.vit_utils.inference import pad_image
            img_pad, (left_pad, top_pad) = pad_image(
                img_rgb,
                float(self.cfg.fallback_aspect_ratio),
            )
            raw_kpts = self.model._inference(img_pad)[0]  # 推理 (133, 3) [y, x, 配置] 
            raw_kpts[:, :2] -= [top_pad, left_pad] #去掉补边造成的偏移
            keypoints = {0: raw_kpts}

            # 以自我为中心的置信度阈值较低（没有身体背景 → 噪音更大）
            kpt_conf_thr = min(
                kpt_conf_thr,
                float(self.cfg.fallback_kpt_conf_threshold),
            )
        self.last_whole_image_fallback = whole_image_fallback
        
        detections = []
        for pid, kpts in keypoints.items():
            # kpts 形状：(133, 3)，其中每行是 [y, x, 置信度]
            candidates = []
            for hand_slice, label, is_right_int in [
                (LEFT_HAND_SLICE,  "Left",  0),
                (RIGHT_HAND_SLICE, "Right", 1),
            ]:
                hand_kpts = kpts[hand_slice]  # 取出手部关键点 (21, 3)
                confidence = hand_kpts[:, 2]  # 提取置信度
                valid = confidence > kpt_conf_thr

                if valid.sum() < min_valid_kpts:
                    continue

                # 提取 [x, y] 格式的 2D 地标
                # easy_ViTPose 返回[y, x, conf]，转换为[x, y]
                landmarks_2d = np.stack([hand_kpts[:, 1], hand_kpts[:, 0]], axis=1).astype(np.float32)

                # 从有效关键点构建 bbox
                valid_pts = landmarks_2d[valid]
                x_min, y_min = valid_pts.min(axis=0)
                x_max, y_max = valid_pts.max(axis=0)
                pad_x = (x_max - x_min) * bbox_pad_ratio
                pad_y = (y_max - y_min) * bbox_pad_ratio
                bbox = np.array([
                    max(0, x_min - pad_x),
                    max(0, y_min - pad_y),
                    min(w_img, x_max + pad_x),
                    min(h_img, y_max + pad_y),
                ], dtype=np.float32)

                # 置信度 = 每个关键点置信度的平均值（有效的）
                mean_conf = float(confidence[valid].mean())

                # 跳过微小的检测（可能来自低置信度关键点的噪音）
                bbox_w = bbox[2] - bbox[0]
                bbox_h = bbox[3] - bbox[1]
                if (
                    bbox_w < float(self.cfg.min_bbox_size_px)
                    or bbox_h < float(self.cfg.min_bbox_size_px)
                ):
                    continue

                candidates.append({
                    'bbox': bbox,
                    'label': label,
                    'is_right_int': is_right_int,
                    'confidence': mean_conf,
                    'landmarks_2d': landmarks_2d,
                    'world_landmarks': None,  # ViTPose 不输出 3D 世界地标
                    'vitpose_valid_keypoints_count': int(valid.sum()),
                    'whole_image_fallback': whole_image_fallback,
                })

            # --- 对重叠的 left/right 检测进行重复数据删除 ---
            # ViTPose 始终输出左右手关键点组
            # （COCO-WholeBody 133 格式）。在以自我为中心的观点中，只有一只手
            # 可能是可见的，但两组都在同一图像区域上射击。
            # 如果两个检测的 IoU 较高，则仅保留置信度较高的一个。
            if len(candidates) == 2:
                b1, b2 = candidates[0]['bbox'], candidates[1]['bbox']
                # 计算 IoU
                xi1 = max(b1[0], b2[0]); yi1 = max(b1[1], b2[1])
                xi2 = min(b1[2], b2[2]); yi2 = min(b1[3], b2[3])
                inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
                a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
                union = a1 + a2 - inter
                iou = inter / max(union, 1e-6)

                if iou > float(self.cfg.overlap_iou_threshold): #交并比较高
                    # 检测到左右手是同一只手——保持更高的置信度
                    best = max(candidates, key=lambda c: c['confidence'])
                    detections.append(best)
                else:
                    detections.extend(candidates)
            else:
                detections.extend(candidates)

        return detections
