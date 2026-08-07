
import os
import cv2
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Any

@dataclass
class CamData:
    """
    表示单个相机帧的空间和视觉状态。

    属性：
        idx (int): 序列内的全局帧索引。
        ts (int): 更正了捕获时间戳（以纳秒为单位）。
        img (np.ndarray): BGR 格式的已处理图像数组（H、W、3）。
        fov (float): 垂直视野（以度为单位）。
        h (int): 图像高度（以像素为单位）。
        w (int): 图像宽度（以像素为单位）。
        k (np.ndarray): 3x3 固有矩阵。
        d (np.ndarray): 畸变系数（通常针对针孔归零）。
        c2w (np.ndarray): 4x4 外部矩阵：相机到世界 (T_c2w)。
        c2d (np.ndarray): 4x4 外部矩阵：相机到设备 (T_c2d)。
        d2w (np.ndarray): 4x4 外部矩阵：设备到世界 (T_d2w)。
    """
    idx: int = 0
    ts: int = 0
    img: np.ndarray = None
    fov: float = 0.0
    h: int = 0
    w: int = 0
    k: np.ndarray = None
    d: np.ndarray = None
    c2w: np.ndarray = None
    c2d: np.ndarray = None
    d2w: np.ndarray = None

@dataclass
class Cam:
    """
    AriaCamData 的序列级容器，包括导出和序列化逻辑。

    属性：
        tss (List[int]): 所有帧的时间戳列表。
        cam (List[AriaCamData]): 每帧数据物体的列表。
        fps (int): 计算出的每秒平均帧数。
        first_ts (int)：第一帧的时间戳。
        fov (float): 垂直视场。
        h (int): 全局图像高度。
        w (int): 全局图像宽度。
        k (np.ndarray): 全局内参矩阵（来自第一帧）。
        d (np.ndarray): 全局失真系数。
        c2d (np.ndarray): 相机到设备的转换。
        mps_path (str)：MPS 数据的根目录路径。
    """
    tss: List[int] = field(default_factory=list)
    cam: List[CamData] = field(default_factory=list)

    fps: int = 0
    first_ts: int = 0
    fov: float = 0.0
    h: int = 0
    w: int = 0
    k: np.ndarray = None
    d: np.ndarray = None
    c2d: np.ndarray = None

    mps_path: str = None


    def __len__(self) -> int:
        """返回存储的帧数。"""
        return len(self.tss)


    @staticmethod
    def _safe_list(arr: Any) -> Any:
        """
        将 numpy 数组转换为嵌套列表以进行 JSON 序列化。

        参数：
            arr (Any): 输入数组或列表。
        返回：
            Any: 如果输入是 np.ndarray，则显示 Python 列表，否则显示原始输入。
        """
        return arr.tolist() if isinstance(arr, np.ndarray) else arr


    def save_aria_cam_json(self, label: str) -> None:
        """
        将单个帧图像和每帧 JSON 元数据保存到文件系统。
        数据保存在[mps_path]/aria/all_data/[idx]/中。

        参数：
            label (str): 相机流的标识符（例如，'rgb'）。
        """
        for idx in range(len(self.tss)):
            # 为特定框架定义目录
            frame_dir = os.path.join(self.mps_path, "preprocess", "all_data", f"{idx:05d}")
            os.makedirs(frame_dir, exist_ok=True)
            img_path = os.path.join(frame_dir, f"{label}.png")

            cam = self.cam[idx]
            # 保存处理后的图像
            cv2.imwrite(img_path, cam.img)

            # 编译每帧元数据
            json_data = {
                "idx": cam.idx,
                "ts": cam.ts,
                "fov": cam.fov,
                "h": cam.h,
                "w": cam.w,
                "k": self._safe_list(cam.k),
                "d": self._safe_list(cam.d),
                "c2w": self._safe_list(cam.c2w),
                "c2d": self._safe_list(cam.c2d),
                "d2w": self._safe_list(cam.d2w),
                f"{label}_path": os.path.join("preprocess", "all_data", f"{idx:05d}", f"{label}.png"),
                "fps": self.fps
            }

            # 写入单独的 JSON 元数据
            with open(os.path.join(frame_dir, f"aria_cam_{label}.json"), 'w') as f:
                json.dump(json_data, f, indent=4)

        # 触发摘要配置保存
        self._save_aria_cam_config_json(label)


    def _save_aria_cam_config_json(self, label: str) -> None:
        """
        保存摄像机流的全局配置摘要。

        参数：
            label (str): 相机流的标识符。
        """
        save_path = os.path.join(self.mps_path, "preprocess", f"aria_cam_{label}_config.json")
        summary_data = {
            "total_frames": len(self),
            "fps": self.fps,
            "first_ts": self.first_ts,
            "h": self.h,
            "w": self.w,
            "k": self._safe_list(self.k),
            "d": self._safe_list(self.d),
            "c2d": self._safe_list(self.c2d),
        }
        with open(save_path, 'w') as f:
            json.dump(summary_data, f, indent=4)
        print(f"[***] JSON Summary saved to: {save_path}")


    def save_aria_cam_video_orig(self, export_video: bool, export_gif: bool, label: str) -> None:
        """
        将帧聚合到视频或 GIF 中以实现可视化目的。

        参数：
            export_video (bool)：是否生成MP4文件。
            export_gif (bool)：是否生成GIF文件。
            label (str): 相机流的标识符。
        """
        frames_all = []
        for idx in range(len(self.tss)):
            cam = self.cam[idx]
            frames_all.append(cam.img)

        # if export_video:
        #     save_path = os.path.join(self.mps_path, "preprocess", "vis", f"aria_cam_{label}.mp4")
        #     os.makedirs(os.path.dirname(save_path), exist_ok=True)

        #     create_video_from_frames(
        #         frames=frames_all,
        #         save_path=save_path,
        #         fps=self.fps,
        #         export_gif=export_gif,
        #         ratio=10
        #     )