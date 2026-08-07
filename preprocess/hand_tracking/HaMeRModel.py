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
import os


import numpy as np
import torch

from typing import Optional, Tuple

from hamer.models import HAMER
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full
from hamer.models import load_hamer


class HaMeRModel:

    HAMER_AVAILABLE = False
    HAMER_HF_REPO = "Leo-TX/hamer"
    # MediaPipe hand_landmarker.task (Apache 2.0, (c) Google)
    # 托管于：https://huggingface.co/Leo-TX/mediapipe-hand
    MEDIAPIPE_HF_REPO = "Leo-TX/mediapipe-hand"
    HAMER_CACHE_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "hamer",
    )
    def __init__(self, device: str = "cuda"):

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.cfg = None

        try:
            # 确保存在检查点 + MANO 文件（如果需要，请从 HF 下载）
            ckpt_path = self._ensure_hamer_ckpts()
            self._ensure_mano()
            self.HAMER_AVAILABLE = True
            self.model, self.cfg = load_hamer(ckpt_path)
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"[HaMeR] Model loaded on {self.device}")

        except Exception as e:
            print(f"[HaMeR] WARNING: Failed to load HaMeR model: {e}")
            self.model = None
    def _ensure_hamer_ckpts(self) -> str:
        """
        确保 HaMeR 检查点和配置文件可用。
        从 HuggingFace Hub 下载到 ~/.cache/hamer/（如果不存在）。
        还确保 MANO_RIGHT.pkl 存在（来自 WiLoR 的 HF 存储库）。

        HaMeR 由 UC Regents / Georgios Pavlakos 在 MIT 许可下发布。
        参见：https://github.com/geopavlakos/hamer

        返回：
            hamer.ckpt 的路径
        """
        cache_dir = self.HAMER_CACHE_DIR

        local_ckpt = os.path.join(cache_dir, "hamer_ckpts", "checkpoints", "hamer.ckpt")
        if os.path.isfile(local_ckpt):
            # 本地文件存在，确保 MANO 也存在
            return local_ckpt

        # 从 HuggingFace 中心下载
        try:
            from huggingface_hub import hf_hub_download
            print("[HaMeR] Downloading checkpoints from HuggingFace Hub...")

            hf_files = {
                "hamer.ckpt":          os.path.join(cache_dir, "hamer_ckpts", "checkpoints", "hamer.ckpt"),
                "model_config.yaml":   os.path.join(cache_dir, "hamer_ckpts", "model_config.yaml"),
                "dataset_config.yaml": os.path.join(cache_dir, "hamer_ckpts", "dataset_config.yaml"),
                "mano_mean_params.npz": os.path.join(cache_dir, "data", "mano_mean_params.npz"),
            }

            for hf_path, local_path in hf_files.items():
                if not os.path.isfile(local_path):
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    cached = hf_hub_download(repo_id=self.HAMER_HF_REPO, filename=hf_path)
                    # 从 HF 缓存到预期位置的符号链接
                    if not os.path.isfile(local_path):
                        os.symlink(cached, local_path)
                    print(f"  ✓ {os.path.basename(hf_path)}")

            print("[HaMeR] Checkpoint download complete")
        except Exception as e:
            print(f"[HaMeR] WARNING: HuggingFace download failed: {e}")
            print(f"[HaMeR] Please manually place hamer.ckpt at: {local_ckpt}")

        return local_ckpt
    def _ensure_mano(self) -> None:
        """
        确保 MANO_RIGHT.pkl 存在于 cache_dir/data/mano/. 中
        Sources (in order): WiLoR的HF缓存→从warmshao/WiLoR-mini.下载

        注意：MANO 许可证禁止重新分发，因此我们从
        原作者的发行版（WiLoR 将其捆绑在其许可证下）。
        """
        cache_dir = self.HAMER_CACHE_DIR
        mano_dst = os.path.join(cache_dir, "data", "mano", "MANO_RIGHT.pkl")
        if os.path.isfile(mano_dst):
            return

        os.makedirs(os.path.dirname(mano_dst), exist_ok=True)

        # 尝试在 WiLoR 的 HuggingFace 缓存中查找它
        import glob
        patterns = [
            os.path.expanduser("~/.cache/huggingface/hub/models--warmshao--WiLoR-mini/snapshots/*/pretrained_models/MANO_RIGHT.pkl"),
        ]
        for pat in patterns:
            matches = glob.glob(pat)
            if matches:
                import shutil
                shutil.copy2(matches[0], mano_dst)
                print(f"[HaMeR] Copied MANO_RIGHT.pkl from WiLoR cache")
                return

        # 尝试触发 WiLoR 下载
        try:
            from huggingface_hub import hf_hub_download
            cached = hf_hub_download(
                repo_id="warmshao/WiLoR-mini",
                subfolder="pretrained_models",
                filename="MANO_RIGHT.pkl",
            )
            import shutil
            shutil.copy2(cached, mano_dst)
            print(f"[HaMeR] Downloaded MANO_RIGHT.pkl via WiLoR's HF repo")
        except Exception:
            print(f"[HaMeR] WARNING: MANO_RIGHT.pkl not found at {mano_dst}")
            print(f"  Please download from https://mano.is.tue.mpg.de/ and place it there.")

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
