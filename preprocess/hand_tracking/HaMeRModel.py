"""
====================================================================================================
HaMeR 3D手部网格恢复器
====================================================================================================

描述：
    使用 HaMeR（手部网格恢复）进行基于图像的手部检测和 3D 网格恢复。
输入：
        - RGB 图像（H, W, 3），uint8
        - 手部边界框 [x1, y1, x2, y2]，像素坐标
        - is_right：1=右手，0=左手
        - focal_length：裁剪图像的近似焦距（像素）
输出：
        - joints_3d：相机空间中的 3D 关节（以米为单位），形状 (21, 3)
====================================================================================================
"""
from pathlib import Path


import numpy as np
import torch

from typing import Optional, Tuple

from hamer.models import HAMER
from hamer.configs import get_config
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full


class HaMeRModel:

    HAMER_AVAILABLE = False
    HAMER_HF_REPO = "Leo-TX/hamer"
    MANO_HF_REPO = "warmshao/WiLoR-mini"

    def __init__(self, device: str = "cuda"):

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.cfg = None

        try:
            assets = self._download_hamer_assets()
            mano_path = self._download_mano()
            self.model, self.cfg = self._load_hamer(assets, mano_path)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.HAMER_AVAILABLE = True
            print(f"[HaMeR] Model loaded on {self.device}")

        except Exception as e:
            print(f"[HaMeR] WARNING: Failed to load HaMeR model: {e}")
            self.model = None

    @classmethod
    def _download_hamer_assets(cls) -> dict[str, str]:
        from huggingface_hub import hf_hub_download

        filenames = ("hamer.ckpt", "model_config.yaml", "mano_mean_params.npz")
        return {
            filename: hf_hub_download(repo_id=cls.HAMER_HF_REPO, filename=filename)
            for filename in filenames
        }

    @classmethod
    def _download_mano(cls) -> str:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=cls.MANO_HF_REPO,
            filename="pretrained_models/MANO_RIGHT.pkl",
        )

    @staticmethod
    def _load_hamer(assets: dict[str, str], mano_path: str):
        model_cfg = get_config(assets["model_config.yaml"], update_cachedir=False)
        model_cfg.defrost()
        model_cfg.MANO.MODEL_PATH = str(Path(mano_path).parent)
        model_cfg.MANO.MEAN_PARAMS = assets["mano_mean_params.npz"]

        if model_cfg.MODEL.BACKBONE.TYPE == "vit" and "BBOX_SHAPE" not in model_cfg.MODEL:
            assert model_cfg.MODEL.IMAGE_SIZE == 256
            model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
            model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
        model_cfg.freeze()

        model = HAMER.load_from_checkpoint(
            assets["hamer.ckpt"], strict=False, cfg=model_cfg
        )
        return model, model_cfg

    @property
    def is_available(self) -> bool:
        return self.model is not None
    @staticmethod
    def _compute_hamer_confidence(
        pred_kpts_3d_rel: np.ndarray,
        joints_cam: np.ndarray,
        joints_2d: np.ndarray,
        img_w: int,
        img_h: int,
        bbox: np.ndarray,
        scaled_focal_length,
    ) -> float:
        """
        根据重建质量计算 HaMeR 的每次检测置信度。

        使用三个信号：
          1.深度合理性：手腕Z应为0.1-2.0m
          2. 2D覆盖：投影关节应覆盖大部分检测bbox
          3. 3D紧凑性：MANO空间中的手部关键点应具有合理的分布

        返回：对 [0.1, 0.99] 的置信度
        """
        try:
            # 1. 深度合理性（手腕 Z）
            wrist_z = float(joints_cam[0, 2])
            if wrist_z < 0.05 or wrist_z > 3.0:
                return 0.15
            depth_score = 1.0
            if wrist_z < 0.1:
                depth_score = 0.5
            elif wrist_z > 2.0:
                depth_score = 0.6

            # 2. 2D 覆盖：关节应跨越 bbox 的合理部分
            bx1, by1, bx2, by2 = bbox[:4]
            bbox_w = max(bx2 - bx1, 1.0)
            bbox_h = max(by2 - by1, 1.0)
            j2d_valid = joints_2d[(joints_2d[:, 0] > 0) & (joints_2d[:, 1] > 0)]
            if len(j2d_valid) > 5:
                j_span_x = j2d_valid[:, 0].max() - j2d_valid[:, 0].min()
                j_span_y = j2d_valid[:, 1].max() - j2d_valid[:, 1].min()
                coverage = (j_span_x / bbox_w + j_span_y / bbox_h) / 2.0
                coverage_score = float(np.clip(coverage, 0.1, 1.0))
            else:
                coverage_score = 0.3

            # 3. 3D紧凑性：MANO 手部尺寸应约为0.15-0.25m
            hand_span = float(np.linalg.norm(
                pred_kpts_3d_rel.max(axis=0) - pred_kpts_3d_rel.min(axis=0)
            ))
            if hand_span < 0.05 or hand_span > 0.5:
                compact_score = 0.3
            else:
                compact_score = 1.0

            confidence = 0.95 * depth_score * coverage_score * compact_score
            return float(np.clip(confidence, 0.1, 0.99))

        except Exception:
            return 0.50
    #禁用梯度计算
    @torch.no_grad()
    def predict_from_crop(
        self,
        img_rgb: np.ndarray,
        bbox: np.ndarray,
        is_right: int = 1,
        focal_length: float = 500.0,
    ) -> Optional[dict]:
        """
        对手部图像裁剪运行 HaMeR 推理。

        参数：
            img_rgb：完整的RGB图像（高、宽、3），uint8。
            bbox: 像素坐标中的边界框 [x1, y1, x2, y2]。
            is_right：1为右手，0为左手。
            focal_length：作物的大致焦距。

        返回：
            字典：
                'joints_3d': (21, 3) 相机空间 3D 关节（以米为单位）
                'joints_2d': (21, 2) 以像素坐标投影的 2D 关节
                'confidence'：浮点重建置信度
            如果推理失败则返回 None。
        """
        if not self.HAMER_AVAILABLE:
            return None

        try:
            x1, y1, x2, y2 = bbox.astype(int)
            bbox_size = max(x2 - x1, y2 - y1)

            if bbox_size < 10: #太小的框忽略
                return None

            img_h, img_w = img_rgb.shape[:2]

            # ViTDetDataset 期望 BGR 图像、框 (N,4)、右 (N,) 为 0/1
            dataset = ViTDetDataset(
                self.cfg,
                img_cv2=img_rgb[:, :, ::-1],  # RGB -> BGR
                boxes=np.array([[x1, y1, x2, y2]], dtype=np.float32),
                right=np.array([is_right], dtype=np.float32),
            )

            if len(dataset) == 0:
                return None

            # 使用 DataLoader 进行正确的批处理（处理 numpy→张量 + 排序规则）
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=1, shuffle=False, num_workers=0
            )
            batch = next(iter(dataloader))
            batch = recursive_to(batch, self.device)

            out = self.model(batch)

            # 提取预测（遵循 demo.py 逻辑）
            pred_cam = out['pred_cam']  # (1, 3) 张量
            pred_keypoints_3d = out['pred_keypoints_3d'][0].cpu().numpy()  # (21, 3)

            # 左手镜像 x 轴（与 demo.py 相同：乘数 = 2*right - 1）
            multiplier = (2 * batch['right'] - 1)  # +1 为右，-1 为左
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]

            # 将裁剪相机转换为全图像相机翻译
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = (
                self.cfg.EXTRA.FOCAL_LENGTH / self.cfg.MODEL.IMAGE_SIZE * img_size.max()
            )
            pred_cam_t_full = cam_crop_to_full(
                pred_cam, box_center, box_size, img_size, scaled_focal_length
            ).detach().cpu().numpy()[0]  # (3,) = [tx, ty, tz]

            # 对于左手，取消镜像关键点的 x 轴
            if is_right == 0:
                pred_keypoints_3d[:, 0] = -pred_keypoints_3d[:, 0]

            # 全图像相机空间中的 3D 关节：
            # pred_keypoints_3d 相对于 MANO 空间中的手根
            # pred_cam_t_full 给出相机翻译 [tx, ty, tz]
            joints_cam = pred_keypoints_3d + pred_cam_t_full[np.newaxis, :]

            # 使用透视投影投影 3D -> 2D
            joints_2d = np.zeros((21, 2), dtype=np.float32)
            if np.all(joints_cam[:, 2] > 0):
                fl = float(scaled_focal_length.cpu()) if isinstance(scaled_focal_length, torch.Tensor) else float(scaled_focal_length)
                joints_2d[:, 0] = joints_cam[:, 0] / joints_cam[:, 2] * fl + img_w / 2.0
                joints_2d[:, 1] = joints_cam[:, 1] / joints_cam[:, 2] * fl + img_h / 2.0

            # 置信度：合并模型质量信号。
            # HaMeR 不输出显式的每个关键点置信度。
            # 相反，计算基于重投影的质量分数
            # 3D→2D 投影一致性：比较 joints_2d （来自 HaMeR 3D）
            # 使用作物 center/size 来衡量预测的效果
            # 与检测到的边界框匹配。重建损失
            # 通过关键点分布和深度稳定性提供有用的代理。
            confidence = self._compute_hamer_confidence(
                pred_keypoints_3d, joints_cam, joints_2d,
                img_w, img_h, bbox, scaled_focal_length
            )

            return {
                'joints_3d': joints_cam.astype(np.float32),
                'joints_2d': joints_2d.astype(np.float32),
                'confidence': confidence,
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None
