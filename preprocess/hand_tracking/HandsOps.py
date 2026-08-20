# -*- coding: utf-8 -*-
# @FileName: HandsOps.py

"""
====================================================================================================
HaMeR 手部追踪可视化和分析操作 (HandsOps.py)
====================================================================================================

描述：
    该模块提供了一系列用于可视化和分析的静态实用方法
    处理 HaMeRHandsGenerator 生成的 OpenPose 21 点手部跟踪数据。
    它在预处理的 HandsData / Hands 结构上运行，涵盖两个主要领域：
    (1) 覆盖在相机图像上的实时每帧可视化，以及 (2) 离线批处理
    比较整个序列中原始运动轨迹与优化运动轨迹的分析图。

核心功能：
    1.骨骼可视化：使用 HaMeR/OpenPose 顺序在相机图像上绘制 2D 手部骨骼
        21 个关键点布局，具有每个手指的颜色编码和抓握状态视觉反馈。
    2. 位姿轴渲染：将 3D wrist/midpoint 位姿轴 (X/Y/Z) 投影到相机图像上
        使用OpenCV的projectPoints，具有装饰性基盘和激光笔效果。
    3. HUD 叠加：呈现信息平视显示器，显示拇指食指捏合情况
        每帧上的距离、手腕中点臂长和抓握状态指示器。
    4.面板UI：绘制每手信息面板（置信度、关节角条、速度计）
        在可视化框架的底角。
    5. 分析图：生成两个综合的每手 (left/right) matplotlib 图形
        覆盖 20 多个子面板：position/velocity/acceleration 轨迹、3D/2D 空间
        路径、姿态偏差（原始与优化）、方向轴和统计文本报告。

生成的输出：
    📊 aria_hands_analysis_r.png -- 右手全运动分析报告图。
    📊 aria_hands_analysis_l.png -- 左手全运动分析报告图。

技术细节：
    - 所有优化 ("Opt") 数据均指通过 Savitzky-Golay / EMA 平滑的轨迹
      在管道上游应用过滤 (HandsTrajectoryOptimizer.py)。
    - 原始 ("Raw") 数据是轨迹平滑之前的 HaMeR 手部结果。
    - 绘图中的颜色约定：红色 = 原始，绿色 = 优化。
    - 3D 坐标系由输入数据的 c2w 决定；2D 坐标为图像像素坐标。
====================================================================================================
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import Any, Dict, List, Optional, Tuple
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import uniform_filter1d

from utils.utils_vis import draw_glass_rect
from data_types.HandsTypes import HandsData, Hands


# ---------------------------------------------------------------------------
# 分析情节布局和风格常量。
# 这些控制图形大小、网格布局、字体大小、线条样式和
# 生成的离线报告数字的 alpha 值
# save_hands_analysis_plots_two()。
# ---------------------------------------------------------------------------
ANALYSIS_FIG_DPI = 300
ANALYSIS_GRID_ROWS = 7
ANALYSIS_GRID_COLS = 4
ANALYSIS_FIGSIZE = (34, 26)
ANALYSIS_TITLE_FONTSIZE = 18
ANALYSIS_SUBTITLE_FONTSIZE = 11
ANALYSIS_LINEWIDTH_MAIN = 2.0
ANALYSIS_LINEWIDTH_AUX = 1.2
ANALYSIS_ALPHA_RAW = 0.28
ANALYSIS_ALPHA_OPT = 0.95
ANALYSIS_ALPHA_FILL = 0.18
ANALYSIS_XTICK_NUM = 8


# HaMeRHandsGenerator outputs the OpenPose 21-keypoint hand convention.
WRIST_INDEX = 0
THUMB_TIP_INDEX = 4
INDEX_TIP_INDEX = 8
FINGER_KEYPOINT_GROUPS = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)
FINGER_INDEX_BY_KEYPOINT = {
    keypoint_index: finger_index
    for finger_index, keypoint_group in enumerate(FINGER_KEYPOINT_GROUPS)
    for keypoint_index in keypoint_group
}
HAND_KEYPOINT_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


class HandsOps:
    """
    用于 HaMeR/OpenPose 手部跟踪可视化和运动学分析的静态实用程序类。

    所有方法都是@staticmethod——此类充当分组的命名空间
    相关操作。不维护实例状态。

    方法组：
        分析助手：_set_frame_time_ticks、_safe_nan_array、
                             _collect_series_single_hand, _interpolate_and_smooth,
                             _compute_pose_deviation, _compute_acc_from_vel_mag
        分析图：save_hands_analysis_plots_two、print_summary_and_eval
        框架可视化：_draw_skeleton、_draw_axis、_draw_aria_hand_panel、
                             _draw_advanced_hud, _draw_opt_wrist_thumb_index_only,
                             draw_aria_hands_panel, draw_aria_hands_skeleton
    """

    @staticmethod
    def _set_frame_time_ticks(ax: Any, frames: np.ndarray, dt: float) -> None:
        """
        将 matplotlib 轴上的 x 轴刻度标签设置为 "frame index + elapsed time (s)"。

        每个刻度显示两行：顶部的整数帧编号和相应的帧编号
        时间戳 (frame_id * dt)（以秒为单位），提供直观的时间上下文。

        参数：
            ax     (matplotlib.axes.Axes): 要修改的子图轴。
            frames (np.ndarray)          :  帧索引的一维数组，形状 (N,)。
            dt     (float)               : 每帧的时间间隔，以秒为单位 (1 / fps)。

        返回：
            None
        """
        n = len(frames)
        if n <= 1:
            return
        k = max(2, ANALYSIS_XTICK_NUM)
        tick_ids = np.linspace(0, n - 1, k).astype(int)
        tick_ids = np.unique(tick_ids)

        ax.set_xticks(frames[tick_ids])
        labels = []
        for fid in frames[tick_ids]:
            t = fid * dt
            labels.append(f"{int(fid)}\n{t:.2f}s")
        ax.set_xticklabels(labels, fontsize=9)


    @staticmethod
    def _safe_nan_array(vals: Any) -> np.ndarray:
        """
        安全地将任意值或序列转换为 float numpy 数组。

        通过将标量输入包装在 1 元素数组中来处理标量输入，确保
        下游数组操作（例如索引、nanmean）始终有效。

        参数：
            vals (Any): 数值的标量、列表或类似数组。

        返回：
            np.ndarray：一个 float64 numpy 数组。标量变为形状 (1,)。
        """
        arr = np.array(vals, dtype=float)
        if arr.ndim == 0:
            arr = np.array([arr], dtype=float)
        return arr


    @staticmethod
    def _collect_series_single_hand(aria_hands: Hands, is_right: bool) -> Dict:
        """
        从 Hands 序列中提取一只手的所有每帧时间序列数据，
        返回准备用于分析绘图的平面数组字典。

        迭代每一帧，增强信心，抓取状态，raw/optimized 3D
        位置、位姿矩阵、速度、2D 投影和方向轴。
        丢失的帧表示为 NaN。平滑的拇指-食指距离曲线
        (dist_opt) 和 2D 中点轨迹 (mid2d_opt) 也在此处计算。

        参数：
            aria_hands (Hands)：带有每帧 HandsData 的完整手部跟踪序列。
            is_right (bool) : True 收集右手；左手为假。

        返回：
            Dict: 包含以下 numpy 数组的字典（来自 locals()）：
                - frames          (N,)    : 帧索引数组[0..N-1]。
                - conf            (N,)    : 跟踪每帧的置信度 [0, 1]。
                - grasp           (N,)    : 二进制抓取状态（0=打开，1=关闭）。
                - dist_raw (N,)：原始拇指索引 3D 距离（以米为单位）。
                - dist_opt (N,)：平滑的拇指索引距离（以米为单位）。
                - mid_raw_xyz (N, 3) ：原始中点世界位置 (m)。
                - mid_opt_xyz (N, 3) ：优化中点世界位置 (m)。
                - wrist_raw_xyz (N, 3) ：原始手腕世界位置 (m)。
                - wrist_opt_xyz (N, 3) ：优化手腕世界位置 (m)。
                - wrist_pose_raw/opt （列表[4x4 或无]）：每帧手腕位姿矩阵。
                - mid_pose_raw/opt （List[4x4 或 None]）：每帧中点位姿矩阵。
                - wrist/mid_lin/ang_raw/opt (N,)：速度大小（m/s、rad/s）。
                - d_mid2w_raw/opt (N,)：中点到手腕的距离（以米为单位）。
                - mid2d_raw (N, 2) ：图像像素中的原始 2D 中点。
                - mid2d_opt (N, 2) ：平滑图像像素中的 2D 中点。
                - mid_ax_raw/opt (N, 3, 3)：世界坐标系中的方向轴列。
        """
        hand_attr = "hand_r" if is_right else "hand_l"
        n = len(aria_hands)
        frames = np.arange(n)
        conf, grasp = np.zeros(n), np.zeros(n)

        # 位置和位姿容器（NaN = 缺失帧）
        dist_raw = np.full(n, np.nan)
        mid_raw_xyz, mid_opt_xyz = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
        wrist_raw_xyz, wrist_opt_xyz = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
        wrist_pose_raw, wrist_pose_opt = [None] * n, [None] * n
        mid_pose_raw, mid_pose_opt = [None] * n, [None] * n
        mid2d_raw, mid2d_opt = np.full((n, 2), np.nan), np.full((n, 2), np.nan)

        # 速度容器（NaN = 缺失帧）
        wrist_lin_raw, wrist_lin_opt = np.full(n, np.nan), np.full(n, np.nan)
        wrist_ang_raw, wrist_ang_opt = np.full(n, np.nan), np.full(n, np.nan)
        mid_lin_raw, mid_lin_opt = np.full(n, np.nan), np.full(n, np.nan)
        mid_ang_raw, mid_ang_opt = np.full(n, np.nan), np.full(n, np.nan)
        d_mid2w_raw, d_mid2w_opt = np.full(n, np.nan), np.full(n, np.nan)
        mid_ax_raw, mid_ax_opt = np.full((n, 3, 3), np.nan), np.full((n, 3, 3), np.nan)

        for i, frame_data in enumerate(aria_hands.hands):
            h = getattr(frame_data, hand_attr)
            if h is None: continue

            conf[i] = float(h.confidence) if h.confidence is not None else 0.0
            grasp[i] = int(h.grasp_state)

            if (
                h.hand_keypoints_3d is not None
                and len(h.hand_keypoints_3d) > INDEX_TIP_INDEX
            ):
                dist_raw[i] = np.linalg.norm(
                    h.hand_keypoints_3d[THUMB_TIP_INDEX]
                    - h.hand_keypoints_3d[INDEX_TIP_INDEX]
                )

            if h.midpoint_translation_raw_world is not None: mid_raw_xyz[i, :] = h.midpoint_translation_raw_world
            if h.midpoint_translation_opt_world is not None: mid_opt_xyz[i, :] = h.midpoint_translation_opt_world
            if h.wrist_pose_raw_world is not None:
                wrist_raw_xyz[i, :] = h.wrist_pose_raw_world[:3, 3]
                wrist_pose_raw[i] = h.wrist_pose_raw_world
            if h.wrist_pose_opt_world is not None:
                wrist_opt_xyz[i, :] = h.wrist_pose_opt_world[:3, 3]
                wrist_pose_opt[i] = h.wrist_pose_opt_world

            mid_pose_raw[i] = h.midpoint_pose_raw_world
            mid_pose_opt[i] = h.midpoint_pose_opt_world

            if h.midpoint_pose_raw_world is not None and h.wrist_pose_raw_world is not None:
                d_mid2w_raw[i] = np.linalg.norm(h.midpoint_pose_raw_world[:3, 3] - h.wrist_pose_raw_world[:3, 3])
            if h.midpoint_pose_opt_world is not None and h.wrist_pose_opt_world is not None:
                d_mid2w_opt[i] = np.linalg.norm(h.midpoint_pose_opt_world[:3, 3] - h.wrist_pose_opt_world[:3, 3])

            # 速度幅度提取（无 → NaN 以保持数组对齐）
            wrist_lin_raw[i] = np.linalg.norm(h.wrist_lin_vel_raw_world) if h.wrist_lin_vel_raw_world is not None else np.nan
            wrist_lin_opt[i] = np.linalg.norm(h.wrist_lin_vel_opt_world) if h.wrist_lin_vel_opt_world is not None else np.nan
            wrist_ang_raw[i] = np.linalg.norm(h.wrist_ang_vel_raw_world) if h.wrist_ang_vel_raw_world is not None else np.nan
            wrist_ang_opt[i] = np.linalg.norm(h.wrist_ang_vel_opt_world) if h.wrist_ang_vel_opt_world is not None else np.nan
            mid_lin_raw[i] = np.linalg.norm(h.midpoint_lin_vel_raw_world) if h.midpoint_lin_vel_raw_world is not None else np.nan
            mid_lin_opt[i] = np.linalg.norm(h.midpoint_lin_vel_opt_world) if h.midpoint_lin_vel_opt_world is not None else np.nan
            mid_ang_raw[i] = np.linalg.norm(h.midpoint_ang_vel_raw_world) if h.midpoint_ang_vel_raw_world is not None else np.nan
            mid_ang_opt[i] = np.linalg.norm(h.midpoint_ang_vel_opt_world) if h.midpoint_ang_vel_opt_world is not None else np.nan

            # 2D 中点：OpenPose 拇指尖（4）和食指尖（8）的平均值
            if (
                h.hand_keypoints_2d is not None
                and len(h.hand_keypoints_2d) > INDEX_TIP_INDEX
            ):
                thumb_tip_2d = h.hand_keypoints_2d[THUMB_TIP_INDEX]
                index_tip_2d = h.hand_keypoints_2d[INDEX_TIP_INDEX]
                if not (
                    np.allclose(thumb_tip_2d, 0)
                    or np.allclose(index_tip_2d, 0)
                ):
                    mid2d_raw[i, :] = (thumb_tip_2d + index_tip_2d) / 2.0

            # 从中点旋转矩阵中提取方向轴列（x，y，z）
            for pose_obj, ax_container in [(mid_pose_raw[i], mid_ax_raw), (mid_pose_opt[i], mid_ax_opt)]:
                if pose_obj is not None:
                    R_mat = pose_obj[:3, :3]
                    for j in range(3): ax_container[i, j, :] = R_mat[:, j]

        # 后处理：插值 NaN 间隙并应用框平滑以获得更清晰的绘图曲线
        dist_opt = HandsOps._interpolate_and_smooth(dist_raw)
        mid2d_opt = np.stack([HandsOps._interpolate_and_smooth(mid2d_raw[:, d]) for d in range(2)], axis=1)

        return locals()


    @staticmethod
    def _interpolate_and_smooth(v: np.ndarray, size: int = 5) -> np.ndarray:
        """
        通过线性插值填充 NaN 间隙，然后应用均匀（盒）滤波器进行平滑。

        用于在缺少检测的分析图中生成视觉上清晰的曲线
        （NaN 帧）否则会产生不连续性。

        参数：
            v    (np.ndarray): 一维输入数组可能包含 NaN 值，形状 (N,)。
            size (int)       : uniform_filter1d 的内核大小（框平滑窗口）。
                               默认值为 5。

        返回：
            np.ndarray：相同形状 (N,) 的平滑一维浮点数组。
                        如果存在少于 2 个有效（非 NaN）样本，则返回不变。
        """
        v = v.copy()
        nm = ~np.isnan(v)
        if np.sum(nm) >= 2:
            v[~nm] = np.interp(np.flatnonzero(~nm), np.flatnonzero(nm), v[nm])
            return uniform_filter1d(v, size=size)
        return v


    @staticmethod
    def _compute_pose_deviation(
        raw_list: List[Optional[np.ndarray]],
        opt_list: List[Optional[np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算每帧位置误差（米）和旋转误差（度）
        原始和优化的位姿矩阵。

        对于每个有效帧对：
          - 位置误差 = 平移向量之间的欧几里德距离 (||t_raw - t_opt||)。
          - 旋转误差 = 相对旋转 R_raw.T @ R_opt 的旋转角度 (deg)。

        参数：
            raw_list (List[Optional[np.ndarray]])：原始 4×4 姿态矩阵的 N 长度列表
                                                   （或者 None 对于丢失的帧）。
            opt_list (List[Optional[np.ndarray]])：优化的 4×4 位姿矩阵的 N 长度列表
                                                   （或者 None 对于丢失的帧）。

        返回：
            元组[np.ndarray，np.ndarray]：
                - pos_err (np.ndarray, shape N)：每帧位置偏差（以米为单位）。
                - rot_err (np.ndarray, shape N)：每帧旋转偏差（以度为单位）。
                  对于缺少任一位姿的帧，两个数组都包含 NaN。
        """
        N = len(raw_list)
        pos_err = np.full(N, np.nan)
        rot_err = np.full(N, np.nan)

        for i in range(N):
            if raw_list[i] is None or opt_list[i] is None:
                continue
            pr = raw_list[i][:3, 3]
            po = opt_list[i][:3, 3]
            pos_err[i] = np.linalg.norm(pr - po)

            rr = raw_list[i][:3, :3]
            ro = opt_list[i][:3, :3]
            r_rel = rr.T @ ro
            ang = np.linalg.norm(R.from_matrix(r_rel).as_rotvec())
            rot_err[i] = np.degrees(ang)

        return pos_err, rot_err


    @staticmethod
    def _compute_acc_from_vel_mag(vel_mag: np.ndarray, dt: float) -> np.ndarray:
        """
        通过速度的一阶有限差分估计标量加速度大小。

        对于每个连续的有效对：acc[i] = (vel_mag[i] - vel_mag[i-1]) / dt。
        用于从速度数组导出加速度时间序列以进行分析图。

        参数：
            vel_mag (np.ndarray)：一维速度幅度数组（m/s 或 rad/s），形状 (N,)。
                                  对于缺少数据的帧可能包含 NaN。
            dt      (float)     : 帧之间的时间步长，以秒为单位 (1 / fps)。

        返回：
            np.ndarray：一维加速度幅度数组，形状 (N,)，其中 NaN
                        当前或之前的速度样本丢失。
        """
        v = vel_mag.copy()
        acc = np.full_like(v, np.nan)
        for i in range(1, len(v)):
            if np.isnan(v[i]) or np.isnan(v[i-1]):
                continue
            acc[i] = (v[i] - v[i-1]) / dt
        return acc


    @staticmethod
    def save_hands_analysis_plots_two(
        aria_hands: Hands,
        save_dir: str,
        dt: float,
        cfg: Any
    ) -> None:
        """
        生成并保存两个全面的运动学分析图形（右手和左手）。

        每个图都是一个 7×4 的子图网格，涵盖：
          第 0 行：置信度 |抓握 + 拇指索引距离 |中点 X |中点 Y
          第 1 行：中点 Z |中位偏差 |中腐偏差 |中加速偏差
          第 2 行：中 3D 轨迹 |中二维轨迹 |手腕 X |手腕 Y
          第 3 行：手腕 Z |手腕位置偏差 |手腕腐烂偏差|手腕加速度偏差
          第 4 行：手腕 3D 轨迹 |手腕 2D 代理 |中↔手腕距离 |手腕|v|
          第 5 行：手腕 |ω| |中点|v| |中点|ω| | （空的）
          第 6 行：中点 X 轴 |中点 Y 轴 |中点Z轴|文字报告

        每个子图都覆盖原始（褪色）与优化（实心）轨迹以进行比较。

        参数：
            aria_hands (Hands)：全序列手部跟踪数据。
            save_dir (str) ：输出目录。如果不存在则创建。
            dt         (float)    : 每帧时间步长（以秒为单位）(1 / fps)。
            cfg        (Any)      : 具有属性的配置物体：
                                      cfg.grasp.fallback_distance_m (float)：捏距阈值 (m)。
                                      cfg.analysis.linear_velocity_limit_mps (float)：线速度限制 (m/s)。
                                      cfg.analysis.angular_velocity_limit_rad_s (float)：角速度限制 (rad/s)。

        返回：
            没有任何。保存文件：
                {save_dir}/aria_hands_analysis_r.png
                {save_dir}/aria_hands_analysis_l.png
        """
        os.makedirs(save_dir, exist_ok=True)
        warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

        def _safe_legend(ax: Any, fontsize: int = 8, ncol: int = 1) -> None:
            """仅当存在标记的艺术家时才绘制图例，以抑制 matplotlib 警告。"""
            handles, labels = ax.get_legend_handles_labels()
            if handles: ax.legend(loc="upper right", fontsize=fontsize, ncol=ncol)

        def _plot_axis_vec(ax: Any, v_raw: np.ndarray, v_opt: np.ndarray, title: str) -> None:
            """
            随时间绘制单个方向轴的 X/Y/Z 世界坐标系分量。

            参数：
                ax    (Axes)      : 目标子图轴。
                v_raw (np.ndarray)：原始轴分量向量，形状 (N, 3)。
                v_opt (np.ndarray)：优化的轴分量向量，形状 (N, 3)。
                title (str)       : 子图标题字符串。
            """
            ax.grid(True, alpha=0.35)

            labels = ["Xcomp", "Ycomp", "Zcomp"]
            for j in range(3):
                ax.plot(frames, v_raw[:, j], alpha=ANALYSIS_ALPHA_RAW, linewidth=ANALYSIS_LINEWIDTH_AUX,
                        label=f"Raw {labels[j]}")
                ax.plot(frames, v_opt[:, j], alpha=ANALYSIS_ALPHA_OPT, linewidth=ANALYSIS_LINEWIDTH_MAIN,
                        label=f"Opt {labels[j]}")

            ax.set_title(title, fontsize=ANALYSIS_SUBTITLE_FONTSIZE, fontweight='bold')
            ax.set_ylabel("axis component (unit)")
            ax.set_xlabel("Frame\nTime (s)")
            ax.set_ylim(-1.05, 1.05)
            HandsOps._set_frame_time_ticks(ax, frames, dt)
            ax.legend(loc="upper right", fontsize=8, ncol=2)

        def _plot_ts(ax: Any, y_raw: Optional[np.ndarray] = None, y_opt: Optional[np.ndarray] = None,
                        label_raw: str = "Raw", label_opt: str = "Optimized",
                        ylabel: str = "", title: str = "",
                        show_limit: Optional[float] = None, limit_label: str = "Limit") -> None:
            """
            通用时间序列子图：绘制原始（褪色）和优化（实心）曲线
            带有可选的水平虚线限制线。

            参数：
                ax          (Axes)           : 目标子图轴。
                y_raw (np.ndarray|None)：原始值，形状 (N,)。如果没有则跳过。
                y_opt (np.ndarray|None)：优化值，形状 (N,)。如果没有则跳过。
                label_raw (str) ：原始曲线的图例标签。
                label_opt (str) ：优化曲线的图例标签。
                ylabel      (str)            : Y 轴标签字符串。
                title       (str)            : 子图标题字符串。
                show_limit (float|None) ：如果给定，则绘制水平虚线限制线。
                limit_label (str) ：限制线的图例标签。
            """
            ax.grid(True, alpha=0.35)
            if y_raw is not None:
                ax.plot(frames, y_raw, color=C_RAW, alpha=ANALYSIS_ALPHA_RAW, linewidth=ANALYSIS_LINEWIDTH_AUX, label=label_raw)
            if y_opt is not None:
                ax.plot(frames, y_opt, color=C_OPT, alpha=ANALYSIS_ALPHA_OPT, linewidth=ANALYSIS_LINEWIDTH_MAIN, label=label_opt)
            if show_limit is not None:
                ax.axhline(show_limit, linestyle="--", linewidth=1.2, color="#444444", alpha=0.9, label=limit_label)
            ax.set_title(title, fontsize=ANALYSIS_SUBTITLE_FONTSIZE, fontweight='bold')
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Frame\nTime (s)")
            HandsOps._set_frame_time_ticks(ax, frames, dt)
            ax.legend(loc="upper right", fontsize=9)

        for is_right in [True, False]:
            side_name = "r" if is_right else "l"
            save_path = os.path.join(save_dir, f"aria_hands_analysis_{side_name}.png")

            data = HandsOps._collect_series_single_hand(aria_hands, is_right=is_right)

            frames = data["frames"]
            title_side = "RIGHT HAND" if is_right else "LEFT HAND"

            plt.style.use('seaborn-v0_8-whitegrid')
            fig = plt.figure(figsize=ANALYSIS_FIGSIZE)
            gs = fig.add_gridspec(ANALYSIS_GRID_ROWS, ANALYSIS_GRID_COLS, hspace=0.35, wspace=0.22)

            # 统一配色（专业、高对比度）
            C_CONF = "#2E86C1"
            C_GRASP = "#E74C3C"
            C_RAW = "#E74C3C"
            C_OPT = "#2ECC71"
            C_AUX = "#F1C40F"
            C_DEV = "#8E44AD"
            C_ROT = "#8B4513"

            fig.suptitle(
                f"ARIA HAND ADVANCED ANALYSIS REPORT - {title_side}",
                fontsize=ANALYSIS_TITLE_FONTSIZE,
                fontweight='bold'
            )

            # -------------------------
            # 1）信心
            # -------------------------
            ax = fig.add_subplot(gs[0, 0])
            ax.grid(True, alpha=0.35)
            ax.plot(frames, data["conf"], color=C_CONF, linewidth=ANALYSIS_LINEWIDTH_MAIN, alpha=0.95, label="Confidence")
            ax.set_title("Hand Tracking Confidence over Time", fontsize=ANALYSIS_SUBTITLE_FONTSIZE, fontweight='bold')
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Confidence")
            ax.set_xlabel("Frame\nTime (s)")
            HandsOps._set_frame_time_ticks(ax, frames, dt)
            ax.legend(loc="upper right", fontsize=9)

            # -------------------------
            # 2) 抓取动作 + 拇指食指距离 (raw/opt) + 阈值
            # -------------------------
            ax = fig.add_subplot(gs[0, 1])
            ax.grid(True, alpha=0.35)

            # 抓取状态 (0/1) 作为填充的步骤区域
            ax.fill_between(frames, data["grasp"], color=C_GRASP, alpha=0.18, step="pre", label="Grasp State (Closed=1)")

            # 在辅助 y 轴上叠加拇指索引距离
            ax2 = ax.twinx()
            ax2.plot(frames, data["dist_raw"], color=C_RAW, alpha=ANALYSIS_ALPHA_RAW, linewidth=ANALYSIS_LINEWIDTH_AUX, label="Thumb-Index Dist (Raw)")
            ax2.plot(frames, data["dist_opt"], color=C_OPT, alpha=ANALYSIS_ALPHA_OPT, linewidth=ANALYSIS_LINEWIDTH_MAIN, label="Thumb-Index Dist (Opt)")
            ax2.axhline(cfg.grasp.fallback_distance_m, linestyle="--", linewidth=1.2, color="#444444", alpha=0.9, label="grasp threshold")

            ax.set_title("Grasp Action + Thumb-Index Distance (Raw/Opt)", fontsize=ANALYSIS_SUBTITLE_FONTSIZE, fontweight='bold')
            ax.set_ylabel("Grasp State")
            ax.set_yticks([0, 1])
            ax.set_xlabel("Frame\nTime (s)")
            HandsOps._set_frame_time_ticks(ax, frames, dt)

            # 将两个轴的图例合并到辅助轴中
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
            ax2.set_ylabel("Distance (m)")

            # -------------------------
            # 3/4/5) 中点 X/Y/Z (raw/opt)
            # -------------------------
            ax = fig.add_subplot(gs[0, 2])
            _plot_ts(ax, data["mid_raw_xyz"][:, 0], data["mid_opt_xyz"][:, 0],
                     ylabel="m", title="Midpoint X in World (Raw/Opt)",
                     label_raw="Mid X Raw", label_opt="Mid X Opt")

            ax = fig.add_subplot(gs[0, 3])
            _plot_ts(ax, data["mid_raw_xyz"][:, 1], data["mid_opt_xyz"][:, 1],
                     ylabel="m", title="Midpoint Y in World (Raw/Opt)",
                     label_raw="Mid Y Raw", label_opt="Mid Y Opt")

            ax = fig.add_subplot(gs[1, 0])
            _plot_ts(ax, data["mid_raw_xyz"][:, 2], data["mid_opt_xyz"][:, 2],
                     ylabel="m", title="Midpoint Z in World (Raw/Opt)",
                     label_raw="Mid Z Raw", label_opt="Mid Z Opt")

            # -------------------------
            # 6/7) 中点 Position/Rotation 偏差（优化与原始）
            # -------------------------
            mid_pos_err, mid_rot_err = HandsOps._compute_pose_deviation(data["mid_pose_raw"], data["mid_pose_opt"])

            ax = fig.add_subplot(gs[1, 1])
            ax.grid(True, alpha=0.35)
            ax.fill_between(frames, mid_pos_err, color=C_DEV, alpha=ANALYSIS_ALPHA_FILL)
            ax.plot(frames, mid_pos_err, color=C_DEV, linewidth=ANALYSIS_LINEWIDTH_MAIN, label="Pos Dev (m)")
            ax.set_title("Midpoint Position Deviation (Opt vs Raw)", fontsize=ANALYSIS_SUBTITLE_FONTSIZE, fontweight='bold')
            ax.set_ylabel("m")
            ax.set_xlabel("Frame\nTime (s)")
            HandsOps._set_frame_time_ticks(ax, frames, dt)
            ax.legend(loc="upper right", fontsize=9)

            ax = fig.add_subplot(gs[1, 2])
            ax.grid(True, alpha=0.35)
            ax.fill_between(frames, mid_rot_err, color=C_ROT, alpha=ANALYSIS_ALPHA_FILL)
            ax.plot(frames, mid_rot_err, color=C_ROT, linewidth=ANALYSIS_LINEWIDTH_MAIN, label="Rot Dev (deg)")
            ax.set_title("Midpoint Rotation Deviation (Opt vs Raw)", fontsize=ANALYSIS_SUBTITLE_FONTSIZE, fontweight='bold')
            ax.set_ylabel("deg")
            ax.set_xlabel("Frame\nTime (s)")
            HandsOps._set_frame_time_ticks(ax, frames, dt)
            ax.legend(loc="upper right", fontsize=9)

            # -------------------------
            # 8) 中点加速度偏差（原始与优化，来自 lin vel mag）
            # -------------------------
            mid_acc_raw = HandsOps._compute_acc_from_vel_mag(data["mid_lin_raw"], dt)
            mid_acc_opt = HandsOps._compute_acc_from_vel_mag(data["mid_lin_opt"], dt)
            mid_acc_dev = np.abs(mid_acc_opt - mid_acc_raw)

            ax = fig.add_subplot(gs[1, 3])
            ax.grid(True, alpha=0.35)
            ax.fill_between(frames, mid_acc_dev, color="#34495E", alpha=ANALYSIS_ALPHA_FILL)
            ax.plot(frames, mid_acc_dev, color="#34495E", linewidth=ANALYSIS_LINEWIDTH_MAIN, label="|Acc Opt - Acc Raw|")
            ax.set_title("Midpoint Acceleration Deviation (Opt vs Raw)", fontsize=ANALYSIS_SUBTITLE_FONTSIZE, fontweight='bold')
            ax.set_ylabel("m/s²")
            ax.set_xlabel("Frame\nTime (s)")
            HandsOps._set_frame_time_ticks(ax, frames, dt)
            ax.legend(loc="upper right", fontsize=9)

            # -------------------------
            # 9) 中点 3D 轨迹 (raw/opt)
            # -------------------------
            ax = fig.add_subplot(gs[2, 0], projection='3d')
            ax.grid(True, alpha=0.3)
            mr = data["mid_raw_xyz"]
            mo = data["mid_opt_xyz"]
            if np.any(~np.isnan(mr[:, 0])):
                ax.plot(mr[:, 0], mr[:, 1], mr[:, 2], color=C_RAW, alpha=0.35, linewidth=1.2, label="Mid Raw")
            if np.any(~np.isnan(mo[:, 0])):
                ax.plot(mo[:, 0], mo[:, 1], mo[:, 2], color=C_OPT, alpha=0.95, linewidth=2.2, label="Mid Opt")
            ax.set_title("Midpoint 3D Trajectory (World)", fontsize=ANALYSIS_SUBTITLE_FONTSIZE, fontweight='bold')
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.legend(loc="upper right", fontsize=8)

            # -------------------------
            # 10) 中点 2D 轨迹 (raw/opt)
            # -------------------------
            ax = fig.add_subplot(gs[2, 1])
            ax.grid(True, alpha=0.35)
            m2r = data["mid2d_raw"]
            m2o = data["mid2d_opt"]
            if np.any(~np.isnan(m2r[:, 0])):
                ax.plot(m2r[:, 0], m2r[:, 1], color=C_RAW, alpha=0.35, linewidth=1.2, label="Mid 2D Raw")
            if np.any(~np.isnan(m2o[:, 0])):
                ax.plot(m2o[:, 0], m2o[:, 1], color=C_OPT, alpha=0.95, linewidth=2.0, label="Mid 2D Opt")
            ax.set_title("Midpoint 2D Trajectory (Image)", fontsize=ANALYSIS_SUBTITLE_FONTSIZE, fontweight='bold')
            ax.set_xlabel("u (px)")
            ax.set_ylabel("v (px)")
            ax.legend(loc="upper right", fontsize=9)

            # -------------------------
            # 11/12/13) 手腕 X/Y/Z (raw/opt)
            # -------------------------
            ax = fig.add_subplot(gs[2, 2])
            _plot_ts(ax, data["wrist_raw_xyz"][:, 0], data["wrist_opt_xyz"][:, 0],
                     ylabel="m", title="Wrist X in World (Raw/Opt)",
                     label_raw="Wrist X Raw", label_opt="Wrist X Opt")

            ax = fig.add_subplot(gs[2, 3])
            _plot_ts(ax, data["wrist_raw_xyz"][:, 1], data["wrist_opt_xyz"][:, 1],
                     ylabel="m", title="Wrist Y in World (Raw/Opt)",
                     label_raw="Wrist Y Raw", label_opt="Wrist Y Opt")

            ax = fig.add_subplot(gs[3, 0])
            _plot_ts(ax, data["wrist_raw_xyz"][:, 2], data["wrist_opt_xyz"][:, 2],
                     ylabel="m", title="Wrist Z in World (Raw/Opt)",
                     label_raw="Wrist Z Raw", label_opt="Wrist Z Opt")

            # -------------------------
            # 14/15) 手腕Position/Rotation偏差
            # -------------------------
            wrist_pos_err, wrist_rot_err = HandsOps._compute_pose_deviation(data["wrist_pose_raw"], data["wrist_pose_opt"])

            ax = fig.add_subplot(gs[3, 1])
            ax.grid(True, alpha=0.35)
            ax.fill_between(frames, wrist_pos_err, color=C_DEV, alpha=ANALYSIS_ALPHA_FILL)
            ax.plot(frames, wrist_pos_err, color=C_DEV, linewidth=ANALYSIS_LINEWIDTH_MAIN, label="Pos Dev (m)")
            ax.set_title("Wrist Position Deviation (Opt vs Raw)", fontsize=ANALYSIS_SUBTITLE_FONTSIZE, fontweight='bold')
            ax.set_ylabel("m")
            ax.set_xlabel("Frame\nTime (s)")
            HandsOps._set_frame_time_ticks(ax, frames, dt)
            ax.legend(loc="upper right", fontsize=9)

            ax = fig.add_subplot(gs[3, 2])
            ax.grid(True, alpha=0.35)
            ax.fill_between(frames, wrist_rot_err, color=C_ROT, alpha=ANALYSIS_ALPHA_FILL)
            ax.plot(frames, wrist_rot_err, color=C_ROT, linewidth=ANALYSIS_LINEWIDTH_MAIN, label="Rot Dev (deg)")
            ax.set_title("Wrist Rotation Deviation (Opt vs Raw)", fontsize=ANALYSIS_SUBTITLE_FONTSIZE, fontweight='bold')
            ax.set_ylabel("deg")
            ax.set_xlabel("Frame\nTime (s)")
            HandsOps._set_frame_time_ticks(ax, frames, dt)
            ax.legend(loc="upper right", fontsize=9)

            # -------------------------
            # 16) 手腕加速度偏差（原始与优化，来自 lin vel mag）
            # -------------------------
            wrist_acc_raw = HandsOps._compute_acc_from_vel_mag(data["wrist_lin_raw"], dt)
            wrist_acc_opt = HandsOps._compute_acc_from_vel_mag(data["wrist_lin_opt"], dt)
            wrist_acc_dev = np.abs(wrist_acc_opt - wrist_acc_raw)

            ax = fig.add_subplot(gs[3, 3])
            ax.grid(True, alpha=0.35)
            ax.fill_between(frames, wrist_acc_dev, color="#34495E", alpha=ANALYSIS_ALPHA_FILL)
            ax.plot(frames, wrist_acc_dev, color="#34495E", linewidth=ANALYSIS_LINEWIDTH_MAIN, label="|Acc Opt - Acc Raw|")
            ax.set_title("Wrist Acceleration Deviation (Opt vs Raw)", fontsize=ANALYSIS_SUBTITLE_FONTSIZE, fontweight='bold')
            ax.set_ylabel("m/s²")
            ax.set_xlabel("Frame\nTime (s)")
            HandsOps._set_frame_time_ticks(ax, frames, dt)
            ax.legend(loc="upper right", fontsize=9)

            # -------------------------
            # 17) 手腕 3D 轨迹 (raw/opt)
            # -------------------------
            ax = fig.add_subplot(gs[4, 0], projection='3d')
            ax.grid(True, alpha=0.3)
            wr = data["wrist_raw_xyz"]
            wo = data["wrist_opt_xyz"]
            if np.any(~np.isnan(wr[:, 0])):
                ax.plot(wr[:, 0], wr[:, 1], wr[:, 2], color=C_RAW, alpha=0.35, linewidth=1.2, label="Wrist Raw")
            if np.any(~np.isnan(wo[:, 0])):
                ax.plot(wo[:, 0], wo[:, 1], wo[:, 2], color=C_OPT, alpha=0.95, linewidth=2.2, label="Wrist Opt")
            ax.set_title("Wrist 3D Trajectory (World)", fontsize=ANALYSIS_SUBTITLE_FONTSIZE, fontweight='bold')
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.legend(loc="upper right", fontsize=8)

            # -------------------------
            # 18) 手腕2D轨迹代理
            # 注意：这里没有相机参数，无法直接投影手腕3D；
            #       拇指索引中点二维轨迹用作结构代理。
            # -------------------------
            ax = fig.add_subplot(gs[4, 1])
            ax.grid(True, alpha=0.35)
            if np.any(~np.isnan(m2r[:, 0])):
                ax.plot(m2r[:, 0], m2r[:, 1], color=C_RAW, alpha=0.35, linewidth=1.2, label="2D Raw (Mid)")
            if np.any(~np.isnan(m2o[:, 0])):
                ax.plot(m2o[:, 0], m2o[:, 1], color=C_OPT, alpha=0.95, linewidth=2.0, label="2D Opt (Mid)")
            ax.set_title("Wrist 2D Trajectory (Image, Proxy)", fontsize=ANALYSIS_SUBTITLE_FONTSIZE, fontweight='bold')
            ax.set_xlabel("u (px)")
            ax.set_ylabel("v (px)")
            ax.legend(loc="upper right", fontsize=9)

            # -------------------------
            # 19) 中点 ↔ 手腕距离 (raw/opt)
            # -------------------------
            ax = fig.add_subplot(gs[4, 2])
            ax.grid(True, alpha=0.35)
            ax.plot(frames, data["d_mid2w_raw"], color=C_RAW, alpha=ANALYSIS_ALPHA_RAW, linewidth=ANALYSIS_LINEWIDTH_AUX, label="Mid2Wrist Raw")
            ax.plot(frames, data["d_mid2w_opt"], color=C_OPT, alpha=ANALYSIS_ALPHA_OPT, linewidth=ANALYSIS_LINEWIDTH_MAIN, label="Mid2Wrist Opt")
            ax.set_title("Distance: Midpoint ↔ Wrist (Raw/Opt)", fontsize=ANALYSIS_SUBTITLE_FONTSIZE, fontweight='bold')
            ax.set_ylabel("m")
            ax.set_xlabel("Frame\nTime (s)")
            HandsOps._set_frame_time_ticks(ax, frames, dt)
            ax.legend(loc="upper right", fontsize=9)

            # -------------------------
            # 20/21/22/23) Wrist/Midpoint 线速度和角速度 (raw/opt) + 限制线
            # -------------------------
            ax = fig.add_subplot(gs[4, 3])
            _plot_ts(ax, data["wrist_lin_raw"], data["wrist_lin_opt"],
                     ylabel="m/s", title="Wrist Linear Velocity |v| (Raw/Opt)",
                     label_raw="Wrist |v| Raw", label_opt="Wrist |v| Opt",
                     show_limit=cfg.analysis.linear_velocity_limit_mps, limit_label=f"v_limit={cfg.analysis.linear_velocity_limit_mps:.1f}m/s")

            ax = fig.add_subplot(gs[5, 0])
            _plot_ts(ax, data["wrist_ang_raw"], data["wrist_ang_opt"],
                     ylabel="rad/s", title="Wrist Angular Velocity |ω| (Raw/Opt)",
                     label_raw="Wrist |ω| Raw", label_opt="Wrist |ω| Opt",
                     show_limit=cfg.analysis.angular_velocity_limit_rad_s, limit_label=f"ω_limit={cfg.analysis.angular_velocity_limit_rad_s:.1f}rad/s")

            ax = fig.add_subplot(gs[5, 1])
            _plot_ts(ax, data["mid_lin_raw"], data["mid_lin_opt"],
                     ylabel="m/s", title="Midpoint Linear Velocity |v| (Raw/Opt)",
                     label_raw="Mid |v| Raw", label_opt="Mid |v| Opt",
                     show_limit=cfg.analysis.linear_velocity_limit_mps, limit_label=f"v_limit={cfg.analysis.linear_velocity_limit_mps:.1f}m/s")

            ax = fig.add_subplot(gs[5, 2])
            _plot_ts(ax, data["mid_ang_raw"], data["mid_ang_opt"],
                     ylabel="rad/s", title="Midpoint Angular Velocity |ω| (Raw/Opt)",
                     label_raw="Mid |ω| Raw", label_opt="Mid |ω| Opt",
                     show_limit=cfg.analysis.angular_velocity_limit_rad_s, limit_label=f"ω_limit={cfg.analysis.angular_velocity_limit_rad_s:.1f}rad/s")

            # -------------------------
            # 中点方向轴 (Raw/Opt)：世界坐标系中的 x 轴/y 轴/z 轴
            # -------------------------
            ax = fig.add_subplot(gs[6, 0])
            _plot_axis_vec(
                ax,
                data["mid_ax_raw"][:, 0, :],   # 世界中 x 轴的中点
                data["mid_ax_opt"][:, 0, :],
                title="Midpoint X-axis in World (Raw/Opt)"
            )

            ax = fig.add_subplot(gs[6, 1])
            _plot_axis_vec(
                ax,
                data["mid_ax_raw"][:, 1, :],   # 世界 y 轴中点
                data["mid_ax_opt"][:, 1, :],
                title="Midpoint Y-axis in World (Raw/Opt)"
            )

            ax = fig.add_subplot(gs[6, 2])
            _plot_axis_vec(
                ax,
                data["mid_ax_raw"][:, 2, :],   # 世界 z 轴中点
                data["mid_ax_opt"][:, 2, :],
                title="Midpoint Z-axis in World (Raw/Opt)"
            )

            # -------------------------
            # 文本报告面板：核心统计摘要呈现为等宽文本框
            # -------------------------
            ax = fig.add_subplot(gs[6, 3])
            ax.axis("off")

            def count_grasps_from_seq(seq: np.ndarray) -> int:
                """计算二进制状态序列中的抓取开始事件（0→1 转换）。"""
                return int(np.sum((seq[1:] == 1) & (seq[:-1] == 0)))

            valid_frames = int(np.sum(data["conf"] > 0))
            grasps_cnt = count_grasps_from_seq(data["grasp"])

            # 速度统计：选择作为主要指标
            peak_wrist_v = np.nanmax(data["wrist_lin_opt"]) if np.any(~np.isnan(data["wrist_lin_opt"])) else 0.0
            mean_wrist_v = np.nanmean(data["wrist_lin_opt"]) if np.any(~np.isnan(data["wrist_lin_opt"])) else 0.0
            peak_wrist_w = np.nanmax(data["wrist_ang_opt"]) if np.any(~np.isnan(data["wrist_ang_opt"])) else 0.0
            mean_wrist_w = np.nanmean(data["wrist_ang_opt"]) if np.any(~np.isnan(data["wrist_ang_opt"])) else 0.0

            peak_mid_v = np.nanmax(data["mid_lin_opt"]) if np.any(~np.isnan(data["mid_lin_opt"])) else 0.0
            mean_mid_v = np.nanmean(data["mid_lin_opt"]) if np.any(~np.isnan(data["mid_lin_opt"])) else 0.0
            peak_mid_w = np.nanmax(data["mid_ang_opt"]) if np.any(~np.isnan(data["mid_ang_opt"])) else 0.0
            mean_mid_w = np.nanmean(data["mid_ang_opt"]) if np.any(~np.isnan(data["mid_ang_opt"])) else 0.0

            # 原始轨迹和优化轨迹之间 position/rotation 偏差的 RMSE
            pos_rmse_mid = np.sqrt(np.nanmean(mid_pos_err**2)) if np.any(~np.isnan(mid_pos_err)) else 0.0
            rot_rmse_mid = np.sqrt(np.nanmean(mid_rot_err**2)) if np.any(~np.isnan(mid_rot_err)) else 0.0
            pos_rmse_wrist = np.sqrt(np.nanmean(wrist_pos_err**2)) if np.any(~np.isnan(wrist_pos_err)) else 0.0
            rot_rmse_wrist = np.sqrt(np.nanmean(wrist_rot_err**2)) if np.any(~np.isnan(wrist_rot_err)) else 0.0

            report = []
            report.append(f"ARIA HAND REPORT ({title_side})")
            report.append("-" * 34)
            report.append(f"Frames: {len(frames)}")
            report.append(f"Valid Frames: {valid_frames}")
            report.append(f"Grasp Events: {grasps_cnt}")
            report.append("")
            report.append("[OPT LIMITS]")
            report.append(f"v_limit: {cfg.analysis.linear_velocity_limit_mps:.2f} m/s")
            report.append(f"w_limit: {cfg.analysis.angular_velocity_limit_rad_s:.2f} rad/s")
            report.append("")
            report.append("[Wrist (Opt)]")
            report.append(f"Peak |v|: {peak_wrist_v:.3f} m/s")
            report.append(f"Mean |v|: {mean_wrist_v:.3f} m/s")
            report.append(f"Peak |ω|: {peak_wrist_w:.3f} rad/s")
            report.append(f"Mean |ω|: {mean_wrist_w:.3f} rad/s")
            report.append(f"RMSE Pos: {pos_rmse_wrist*100:.2f} cm")
            report.append(f"RMSE Rot: {rot_rmse_wrist:.2f} deg")
            report.append("")
            report.append("[Midpoint (Opt)]")
            report.append(f"Peak |v|: {peak_mid_v:.3f} m/s")
            report.append(f"Mean |v|: {mean_mid_v:.3f} m/s")
            report.append(f"Peak |ω|: {peak_mid_w:.3f} rad/s")
            report.append(f"Mean |ω|: {mean_mid_w:.3f} rad/s")
            report.append(f"RMSE Pos: {pos_rmse_mid*100:.2f} cm")
            report.append(f"RMSE Rot: {rot_rmse_mid:.2f} deg")

            ax.text(
                0.02, 0.98, "\n".join(report),
                va="top", ha="left",
                fontsize=10,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", fc="#111111", ec="#444444", alpha=0.95),
                color="white"
            )
            plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05)
            try:
                plt.tight_layout(rect=[0, 0, 1, 0.965])
            except:
                pass
            plt.savefig(save_path, dpi=ANALYSIS_FIG_DPI)
            plt.close()
            print(f"[***] Hand analysis plot saved to: {save_path}")


    @staticmethod
    def print_summary_and_eval(aria_hands: Hands) -> None:
        """
        将格式化的手部交互和运动学摘要报告打印到标准输出。

        Reports: 总帧数、每侧手部存在计数、抓取事件计数、
        peak/mean 手腕线速度和角速度（来自优化轨迹），
        以及基于手部存在覆盖范围的总体数据质量评级。

        参数：
            aria_hands (Hands)：全序列手部跟踪数据。

        返回：
            没有任何。将边框框报告输出到标准输出。
        """
        total_frames = len(aria_hands)
        r_frames = sum(1 for h in aria_hands.hands if h.hand_r)
        l_frames = sum(1 for h in aria_hands.hands if h.hand_l)

        lin_vels = []
        ang_vels = []
        for h_group in aria_hands.hands:
            for hand in [h_group.hand_r, h_group.hand_l]:
                if hand and hand.wrist_pose is not None:
                    lin_vels.append(np.linalg.norm(hand.wrist_lin_vel_opt_world))
                    ang_vels.append(np.linalg.norm(hand.wrist_ang_vel_opt_world))

        max_lin_v = max(lin_vels) if lin_vels else 0.0
        avg_lin_v = np.mean(lin_vels) if lin_vels else 0.0
        max_ang_v = max(ang_vels) if ang_vels else 0.0
        avg_ang_v = np.mean(ang_vels) if ang_vels else 0.0

        def count_grasps(hand_side: str) -> int:
            """计算一侧的抓握起始事件（0→1 转换）。"""
            states = [getattr(h, hand_side).grasp_state if getattr(h, hand_side) else 0 for h in aria_hands.hands]
            return sum(1 for i in range(1, len(states)) if states[i] == 1 and states[i-1] == 0)

        r_grasps = count_grasps('hand_r')
        l_grasps = count_grasps('hand_l')

        print("\n" + "╔" + "═" * 60 + "╗")
        print(f"║{'ARIA HAND INTERACTION & KINEMATICS REPORT':^60}║")
        print("╠" + "═" * 60 + "╣")
        print(f"║  - Total Frames       : {total_frames:<38} ║")
        print(f"║  - Hand Presence (R/L): {f'{r_frames} / {l_frames}':<38} ║")
        print(f"║  - Grasp Events (R/L) : {f'{r_grasps} / {l_grasps}':<38} ║")
        print(f"║ {'':<58} ║")
        print(f"║ [SMOOTHED KINEMATICS - OPTIMIZED] {'':<25} ║")
        print(f"║  - Peak Linear Speed  : {max_lin_v:>8.3f} m/s {'':<28} ║")
        print(f"║  - Mean Linear Speed  : {avg_lin_v:>8.3f} m/s {'':<28} ║")
        print(f"║  - Peak Angular Speed : {max_ang_v:>8.3f} rad/s {'':<26} ║")
        print(f"║  - Mean Angular Speed : {avg_ang_v:>8.3f} rad/s {'':<26} ║")

        # 质量得分：双手的平均手部存在比例
        quality_score = (r_frames + l_frames) / (2 * total_frames + 1e-6)
        quality = "EXCELLENT" if quality_score > 0.8 else "GOOD" if quality_score > 0.4 else "PARTIAL"
        print(f"║  - Overall Data Qual  : {quality:<38} ║")
        print("╚" + "═" * 60 + "╝\n")


    @staticmethod
    def _draw_skeleton(img: np.ndarray, hand: Any) -> np.ndarray:
        """
        使用 HaMeR/OpenPose 21 个关键点布局在图像上绘制 2D 手部骨骼叠加层。

        使用 HAND_KEYPOINT_CONNECTIONS 进行骨骼连接。关节点颜色
        是每个手指并在张开手和抓握状态之间变化。在抓取状态下，
        每个关节上都渲染了一个额外的外发光环，以增强视觉效果。

        关键点索引约定 (HaMeR/OpenPose)：
            0=手腕；1-4=拇指 CMC/MCP/IP/Tip；5-8=食指 MCP/PIP/DIP/Tip；
            9-12=中指；13-16=无名指；17-20=小指。

        参数：
            img  (np.ndarray): 要绘制的 BGR 输入图像，形状（高、宽、3）。
            hand (Any)       : HandData 物体具有：
                                 hand_keypoints_2d（np.ndarray，形状21×2）：2D关键点（px）。
                                 grasp_state（整数）：0=打开，1=closed/grasp.

        返回：
            np.ndarray：就地绘制骨架的图像。如果返回不变
                        手部为无、关键点缺失或少于 21 点。
        """
        if hand is None or hand.hand_keypoints_2d is None:
            return img

        pts_float = np.asarray(hand.hand_keypoints_2d, dtype=np.float64)
        if len(pts_float) < 21 or not np.isfinite(pts_float).all():
            return img
        safe_limit = float(max(img.shape[:2]) * 4)
        if np.any(np.abs(pts_float) > safe_limit):
            return img
        pts = np.rint(pts_float).astype(np.int32)
        is_grasp = (hand.grasp_state == 1)

        # BGR 中的每个手指颜色映射：[拇指、食指、中指、无名指、小指]
        if not is_grasp:
            # 开放状态的柔和配色方案
            finger_colors = [
                (160, 160, 255), # 拇指 - 珊瑚粉（微红）
                (160, 255, 160), # 索引 - 鲜绿（绿色）
                (255, 210, 160), # 中-天蓝色（偏蓝）
                (150, 250, 250), # 戒指-亮黄色（淡黄色）
                (250, 160, 250)  # Pinky - 薰衣草紫（偏紫）
            ]
        else:
            # 握持状态：每根手指都变成纯红色，所以握紧的手
            # 事件一目了然。
            finger_colors = [(0, 0, 255)] * 5

        # 分辨率自适应尺寸（参考：480 像素高图像）。
        sc = img.shape[0] / 480.0
        bone_lw = max(1, int(round(1 * sc)))
        dot_r   = max(1, int(round(2 * sc)))
        glow_r  = max(2, int(round(3 * sc)))

        # 使用 HaMeR/OpenPose 关节拓扑绘制骨骼连接（细浅灰色）
        for idx1, idx2 in HAND_KEYPOINT_CONNECTIONS:
            p1, p2 = tuple(pts[idx1]), tuple(pts[idx2])
            if p1 != (0,0) and p2 != (0,0):
                cv2.line(img, p1, p2, (210, 210, 210), bone_lw, cv2.LINE_AA)

        for i, pt in enumerate(pts):
            p = tuple(pt)
            if p == (0, 0): continue

            if is_grasp:
                # 抓取状态：所有 21 个关键点（包括 wrist）均为红色。
                color = (0, 0, 255)
            else:
                # 张开手状态：保持每根手指的配色，手腕单独显示为白色。
                if i == WRIST_INDEX:
                    color = (255, 255, 255)
                else:
                    f_idx = FINGER_INDEX_BY_KEYPOINT[i]
                    color = finger_colors[f_idx]

            cv2.circle(img, p, dot_r, color, -1, cv2.LINE_AA)

            # 抓握状态下关节周围的发光环可增强视觉效果
            if is_grasp:
                cv2.circle(img, p, glow_r, color, bone_lw, cv2.LINE_AA)

            # 仅在张开状态下为白色手腕点添加灰色轮廓。
            if not is_grasp and i == WRIST_INDEX:
                cv2.circle(img, p, dot_r, (150, 150, 150), bone_lw, cv2.LINE_AA)

        return img


    @staticmethod
    def _draw_axis(img: np.ndarray, pose: Optional[np.ndarray], k: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        绘制风格化的 3D 坐标轴小控件（X/Y/Z 箭头 + 底座圆盘 + 激光笔）
        投影到图像上由给定位姿定义的位置。

        小发明元素：
          - 三个箭头轴：X=红色、Y=绿色、Z=蓝色（OpenCV BGR 约定）。
          - XZ 平面上的装饰基盘，用于空间接地。
          - 沿X轴方向延伸的激光指针线。
          - 半透明混合（0.7 覆盖 + 0.3 原始）以获得干净的外观。

        参数：
            img  (np.ndarray)         : BGR 输入图像，形状（高、宽、3）。
            pose (np.ndarray or None) : 4×4 相机空间姿态矩阵 (T_cam_hand)。
                                        如果没有，则图像原样返回。
            k    (np.ndarray)         : 相机固有矩阵，形状 (3, 3)。
            d    (np.ndarray)         : 失真系数，形状 (4,) 或 (5,)。

        返回：
            np.ndarray：混合了轴小控件的图像。
                        如果位姿为“无”或原点超出框架，则返回原始图像。
        """
        if pose is None:
            return img
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            return img
        if pose[2, 3] <= 1e-4:
            return img

        # 1.提取旋转向量和平移向量
        r_vec, _ = cv2.Rodrigues(pose[:3, :3])
        t_vec = pose[:3, 3]

        # 2. 在 3D 空间中定义点
        axis_len_x = 0.06
        axis_len_y = 0.10
        axis_len_z = 0.06
        pointer_len = 0.10 # 指针线长12cm
        base_r = 0.025     # 基盘半径2.5cm

        # Define: X、Y、Z轴端点、原点、指针端点
        pts_3d = np.float32([
            [axis_len_x, 0, 0],          # X
            [0, axis_len_y, 0],          # Y（更长）
            [0, 0, axis_len_z],          # Z
            [0, 0, 0],                   # origin
            [pointer_len, 0, 0],         # 沿 X 的指针
        ])

        # 基盘：XZ平面16点圆
        num_circle_pts = 16
        circle_pts_3d = []
        for i in range(num_circle_pts):
            angle = 2 * np.pi * i / num_circle_pts
            # circle_pts_3d.append([base_r * np.cos(angle), base_r * np.sin(angle), 0]) # XY 平面
            circle_pts_3d.append([base_r * np.cos(angle), 0, base_r * np.sin(angle)])# XZ平面
        pts_3d = np.append(pts_3d, np.float32(circle_pts_3d), axis=0)

        # 3.将所有3D点投影到2D像素坐标
        try:
            img_pts, _ = cv2.projectPoints(pts_3d, r_vec, t_vec, k, d)
            img_pts = img_pts.reshape(-1, 2)
        except (cv2.error, ValueError, OverflowError):
            return img
        if not np.isfinite(img_pts).all():
            return img

        h, w = img.shape[:2]
        safe_limit = float(max(h, w) * 4)
        if np.any(np.abs(img_pts) > safe_limit):
            return img
        img_pts = [tuple(int(round(value)) for value in point) for point in img_pts]

        origin = img_pts[3]
        if not (0 <= origin[0] < w and 0 <= origin[1] < h):
            return img

        # --- 绘图开始 ---
        overlay = img.copy()

        # A.绘制基盘（XZ平面半透明环）
        for i in range(num_circle_pts):
            p1 = img_pts[5 + i]
            p2 = img_pts[5 + (i + 1) % num_circle_pts]
            cv2.line(overlay, p1, p2, (200, 200, 200), 1, cv2.LINE_AA)

        # B. 绘制激光笔（X方向长伸，青黄色光）
        laser_end = img_pts[4]
        cv2.line(overlay, origin, laser_end, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(overlay, laser_end, 2, (255, 255, 0), -1, cv2.LINE_AA)

        # C. 用箭头绘制标准 XYZ 轴（BGR：X=红色，Y=绿色，Z=蓝色）
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        labels = ["X", "Y", "Z"]
        for i in range(3):
            cv2.arrowedLine(overlay, origin, img_pts[i], colors[i], 3, tipLength=0.3, line_type=cv2.LINE_AA)
            label_pos = (img_pts[i][0] + 8, img_pts[i][1] + 8)
            cv2.putText(overlay, labels[i], label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, colors[i], 2, cv2.LINE_AA)

        # D. 原点处的小白点标记枢轴点
        cv2.circle(overlay, origin, 3, (255, 255, 255), -1, cv2.LINE_AA)

        # 将叠加层与原始图像混合以获得半透明效果
        return cv2.addWeighted(overlay, 0.7, img, 0.3, 0)


    @staticmethod
    def _draw_aria_hand_panel(
        img: np.ndarray,
        idx: int,
        hand: Any,
        is_right: bool,
        opt_v_limit: float,
    ) -> np.ndarray:
        """
        在图像的底角绘制每只手的信息面板（玻璃式 UI 卡）。

        面板显示：
          - 手侧标签和抓握状态（打开/关闭），抓握上有闪烁的红色边框。
          - 跟踪置信值。
          - 每个手指有五个弯曲和外展条（标准化为 [0, 1] 范围）。
          - 带有当前速度读数的手腕线速度计条。

        面板锚固：右侧→右下角；左侧 → 左下角。
        所有尺寸均与图像宽度成比例缩放，以实现与分辨率无关。

        参数：
            img      (np.ndarray): 要绘制的 BGR 图像，形状（高、宽、3）。
            idx      (int)       : 当前帧索引（用于眨眼动画计时）。
            hand     (Any)       : HandData 物体，如果未检测到手，则为 None。
            is_right (bool) : True = 右手（右下）； False = 左（左下）。
            opt_v_limit（浮动）

        返回：
            np.ndarray：就地绘制手部信息面板的图像。
        """
        is_blink = (idx // 5) % 2 == 0

        # 缩放助手：S() 表示像素大小，F() 表示字体比例，T() 表示线条粗细
        # 1280x960 时参考宽度 840 → sc=1.52。匹配 SLAM 面板的
        # 布局参考，以便底部手面板不会与
        # 顶部 SLAM 面板。
        img_h, img_w = img.shape[:2]
        sc = img_w / 840.0

        def S(val): return int(val * sc)
        def F(val): return max(0.3, val * sc)
        def T(val): return max(1, int(val * sc))

        w, h = S(200), S(180)
        margin = S(10)

        # 锚点位置：右侧面板位于右下角，左侧面板位于左下角
        y = img_h - h - margin
        x = img_w - w - margin if is_right else margin

        # 关节角度键映射：UI标签→（屈曲键列表、外展键）
        mapping = {
            "Thu": (["Thumb_CMC_Flex", "Thumb_MCP_Flex", "Thumb_IP_Flex"], "Thumb_CMC_Abd"),
            "Ind": (["Index_MCP_Flex", "Index_PIP_Flex", "Index_DIP_Flex"], "Index_MCP_Abd"),
            "Mid": (["Middle_MCP_Flex", "Middle_PIP_Flex", "Middle_DIP_Flex"], "Middle_MCP_Abd"),
            "Rin": (["Ring_MCP_Flex", "Ring_PIP_Flex", "Ring_DIP_Flex"], "Ring_MCP_Abd"),
            "Pin": (["Pinky_MCP_Flex", "Pinky_PIP_Flex", "Pinky_DIP_Flex"], "Pinky_MCP_Abd")
        }

        # --- 2. 状态逻辑 ---
        if hand is not None:
            is_grasp = (hand.grasp_state == 1)
            conf = hand.confidence
            state_str = "CLOSED" if is_grasp else "OPEN"
            col = (0, 0, 255) if is_grasp else (0, 191, 255)
            angle_data = hand.joint_angles.data if (hand.joint_angles and hand.joint_angles.data) else {}
        else:
            is_grasp = False; conf = 0.0; state_str = "N/A"
            col = (120, 120, 120); angle_data = {}

        # --- 3.绘制磨砂玻璃背景板---
        img = draw_glass_rect(img, (x, y), (x + w, y + h))
        if is_grasp and is_blink:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), T(2), cv2.LINE_AA)
                # 左上角的红点作为抓取指示器
                cv2.circle(img, (x + S(15), y + S(18)), S(4), (0, 0, 255), -1, cv2.LINE_AA)

        # --- 4.标题行：手边居中+抓握状态标签 ---
        title = f"{'RIGHT' if is_right else 'LEFT'} HAND: {state_str}"
        font_scale = F(0.45)
        (t_w, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, font_scale, T(1))
        t_x = x + (w - t_w) // 2
        cv2.putText(img, title, (t_x, y + S(25)), cv2.FONT_HERSHEY_DUPLEX, font_scale, col, T(1), cv2.LINE_AA)

        # --- 5. 置信行 ---
        conf_txt = f"CONFIDENCE: {conf:.2f}"
        # stat_txt = f"STATUS: {state_str}"
        cv2.putText(img, conf_txt, (x + S(10), y + S(45)), cv2.FONT_HERSHEY_DUPLEX, F(0.32), (200, 200, 200), T(1), cv2.LINE_AA)
        # cv2.putText(img, stat_txt, (x + S(105), y + S(45)), cv2.FONT_HERSHEY_DUPLEX, F(0.32), col, T(1), cv2.LINE_AA)

        # --- 6. 五指杆：顶部杆=屈曲（标准为180°），底部=外展（标准为45°）---
        finger_names = ["Thu", "Ind", "Mid", "Rin", "Pin"]
        max_bar_w = S(125)

        for i, ui_name in enumerate(finger_names):
            y_row = y + S(65 + i * 18)
            cv2.putText(img, ui_name, (x + S(10), y_row + S(8)), cv2.FONT_HERSHEY_DUPLEX, F(0.35), (255, 255, 255), T(1), cv2.LINE_AA)

            flex_keys, abd_key = mapping[ui_name]
            flex_sum = sum([angle_data.get(k, 0) for k in flex_keys])
            abd_val = angle_data.get(abd_key, 0)

            f_col = (0, 0, 255) if is_grasp else (200, 130, 60)
            f_w = int(min(flex_sum / 180.0, 1.0) * max_bar_w)
            a_w = int(min(abd_val / 45.0, 1.0) * max_bar_w)

            # 用于屈曲（顶部）和外展（底部）杆的深色背景槽
            cv2.rectangle(img, (x + S(55), y_row + S(1)), (x + S(55) + max_bar_w, y_row + S(4)), (50, 50, 50), -1)
            cv2.rectangle(img, (x + S(55), y_row + S(7)), (x + S(55) + max_bar_w, y_row + S(10)), (50, 50, 50), -1)

            if f_w > 0:
                cv2.rectangle(img, (x + S(55), y_row + S(1)), (x + S(55) + f_w, y_row + S(4)), f_col, -1, cv2.LINE_AA)
            if a_w > 0:
                cv2.rectangle(img, (x + S(55), y_row + S(7)), (x + S(55) + a_w, y_row + S(10)), (100, 200, 255), -1, cv2.LINE_AA)

        # --- 7. 速度计：显示优化的手腕线速度，固定在 opt_v_limit ---
        if hand and hand.wrist_lin_vel_opt_world is not None:
            velocity_mag = np.linalg.norm(hand.wrist_lin_vel_opt_world)
        else:
            velocity_mag = 0.0

        v_y_row = y + S(158)
        cv2.putText(img, "VEL", (x + S(10), v_y_row + S(10)), cv2.FONT_HERSHEY_DUPLEX, F(0.35), (255, 255, 255), T(1), cv2.LINE_AA)

        max_vel_limit = opt_v_limit
        vel_ratio = min(velocity_mag / max_vel_limit, 1.0)
        vel_bar_w = int(vel_ratio * max_bar_w)

        # 速度计的深灰色背景槽
        cv2.rectangle(img, (x + S(55), v_y_row + S(2)), (x + S(55) + max_bar_w, v_y_row + S(10)), (40, 40, 40), -1)

        # 青色低于 1 m/s，黄绿色高于（颜色变化表示高速）
        v_color = (255, 255, 0) if velocity_mag < 1.0 else (0, 255, 255)

        if vel_bar_w > 0:
            cv2.rectangle(img, (x + S(55), v_y_row + S(2)), (x + S(55) + vel_bar_w, v_y_row + S(10)), v_color, -1, cv2.LINE_AA)

        vel_text = f"{velocity_mag:.2f} m/s"
        cv2.putText(img, vel_text, (x + S(125), v_y_row - S(2)), cv2.FONT_HERSHEY_DUPLEX, F(0.3), v_color, T(1), cv2.LINE_AA)

        return img


    @staticmethod
    def _draw_opt_wrist_thumb_index_only(
        img: np.ndarray,
        aria_hand_data: Any,
        k: np.ndarray,
        d: np.ndarray,
        c2w: np.ndarray,
        draw_lines: bool = True,
        radius: int = 4,
        thickness: int = -1,
    ) -> np.ndarray:
        """
        绘制 5 个优化的关键节点及其骨架连接，从
        世界坐标到相机图像上。

        绘制的节点：手腕（W）、拇指基部、食指基部、拇指尖（T）、食指尖（I）。
        骨骼连接（当 draw_lines=True 时）：
            手腕 → 拇指根部 → 拇指尖
            手腕 → 分度盘底座 → 分度盘尖端
                        回退：如果基指关节坐标不可用，则直接连接手腕 → 尖端。

        节点颜色编码（BGR）：
            W（手腕）= 白色 | T（拇指尖）= 红色 | I（索引提示）= 绿色 |碱=青色。

        参数：
            img           (np.ndarray): 要绘制的 BGR 图像，形状（高、宽、3）。
            aria_hand_data （任意）：具有优化的世界空间字段的 HandData：
                                          wrist_pose_opt_world、thumb/index_translation_opt_world、
                                          thumb/index_base_opt_world.
            k             (np.ndarray): 相机固有矩阵，形状 (3, 3)。
            d             (np.ndarray): 失真系数。
            c2w           (np.ndarray): 相机到世界的 4×4 变换矩阵。
            draw_lines (bool) ：是否绘制骨骼连接线。默认为真。
            radius        (int)       : 关键点圆半径（以像素为单位）。默认 4。
            thickness     (int)       : 圆形填充厚度（-1 = 填充）。默认-1。

        返回：
            np.ndarray：具有优化关键点和就地绘制的骨骼的图像。
        """
        if aria_hand_data is None:
            return img

        T_w2c = np.linalg.inv(c2w)

        def _project_world_point(p_w: np.ndarray) -> Optional[Tuple[int, int]]:
            """将单个世界空间 3D 点投影到图像像素坐标 (u, v)。"""
            if p_w is None:
                return None
            p_w = np.asarray(p_w, dtype=np.float64).reshape(3)
            if not np.isfinite(p_w).all() or not np.isfinite(c2w).all():
                return None
            # 变换世界 → 相机坐标系
            p_c = (T_w2c[:3, :3] @ p_w) + T_w2c[:3, 3]
            if not np.isfinite(p_c).all() or p_c[2] <= 1e-4:
                return None
            rvec = np.zeros((3, 1), dtype=np.float64)
            tvec = np.zeros((3, 1), dtype=np.float64)
            try:
                uv, _ = cv2.projectPoints(
                    p_c.reshape(1, 3), rvec, tvec, k, d
                )
            except (cv2.error, ValueError, OverflowError):
                return None
            u, v = uv.reshape(2)
            if not np.isfinite([u, v]).all():
                return None
            h, w = img.shape[:2]
            if not (0 <= u < w and 0 <= v < h):
                return None
            return int(round(u)), int(round(v))

        def _draw_one_hand(hand: Any) -> None:
            if hand is None:
                return

            # 获取优化的世界空间关键点位置
            p_wrist_w = None
            if hand.wrist_pose_opt_world is not None:
                p_wrist_w = hand.wrist_pose_opt_world[:3, 3]

            p_t_tip_w = hand.thumb_translation_opt_world
            p_i_tip_w = hand.index_translation_opt_world

            # 基指关节位置（旧数据中可能不存在）
            p_t_base_w = getattr(hand, "thumb_base_opt_world", None)
            p_i_base_w = getattr(hand, "index_base_opt_world", None)

            # 将所有关键点投影到 2D 像素坐标
            uv_wrist = _project_world_point(p_wrist_w)
            uv_t_tip = _project_world_point(p_t_tip_w)
            uv_i_tip = _project_world_point(p_i_tip_w)
            uv_t_base = _project_world_point(p_t_base_w)
            uv_i_base = _project_world_point(p_i_base_w)

            if uv_wrist is None and uv_t_tip is None and uv_i_tip is None:
                return

            C_WRIST = (255, 255, 255)  # White
            C_T_TIP = (0, 0, 255)      # Red
            C_I_TIP = (0, 255, 0)      # Green
            C_BASE  = (255, 255, 0)    # 青色（指关节）
            C_LINE  = (255, 255, 0)    # 青色（用于骨骼）

            if draw_lines:
                if uv_wrist and uv_t_base:
                    cv2.line(img, uv_wrist, uv_t_base, C_LINE, 1, cv2.LINE_AA)
                if uv_wrist and uv_i_base:
                    cv2.line(img, uv_wrist, uv_i_base, C_LINE, 1, cv2.LINE_AA)
                if uv_t_base and uv_t_tip:
                    cv2.line(img, uv_t_base, uv_t_tip, C_LINE, 1, cv2.LINE_AA)
                if uv_i_base and uv_i_tip:
                    cv2.line(img, uv_i_base, uv_i_tip, C_LINE, 1, cv2.LINE_AA)

                # 回退：如果未提取底座（旧数据），则直接连接手腕 -> 尖端
                if uv_t_base is None and uv_wrist and uv_t_tip:
                    cv2.line(img, uv_wrist, uv_t_tip, C_LINE, 1, cv2.LINE_AA)
                if uv_i_base is None and uv_wrist and uv_i_tip:
                    cv2.line(img, uv_wrist, uv_i_tip, C_LINE, 1, cv2.LINE_AA)

            # 绘制带有轮廓和标签的手腕节点
            if uv_wrist is not None:
                cv2.circle(img, uv_wrist, radius + 1, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.circle(img, uv_wrist, radius, C_WRIST, thickness, cv2.LINE_AA)
                cv2.putText(img, "W", (uv_wrist[0] + 6, uv_wrist[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_WRIST, 1, cv2.LINE_AA)

            # 画拇指指关节底座（青色，带白色核心点）
            if uv_t_base is not None:
                cv2.circle(img, uv_t_base, radius, C_BASE, thickness, cv2.LINE_AA)
                cv2.circle(img, uv_t_base, 2, (255, 255, 255), -1, cv2.LINE_AA)

            # 绘制食指底座（青色，带白色芯点）
            if uv_i_base is not None:
                cv2.circle(img, uv_i_base, radius, C_BASE, thickness, cv2.LINE_AA)
                cv2.circle(img, uv_i_base, 2, (255, 255, 255), -1, cv2.LINE_AA)

            # 画拇指尖（红色，T标签）
            if uv_t_tip is not None:
                cv2.circle(img, uv_t_tip, radius, C_T_TIP, thickness, cv2.LINE_AA)
                cv2.putText(img, "T", (uv_t_tip[0] + 6, uv_t_tip[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_T_TIP, 1, cv2.LINE_AA)

            # 绘制索引尖端（绿色，I 标签）
            if uv_i_tip is not None:
                cv2.circle(img, uv_i_tip, radius, C_I_TIP, thickness, cv2.LINE_AA)
                cv2.putText(img, "I", (uv_i_tip[0] + 6, uv_i_tip[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_I_TIP, 1, cv2.LINE_AA)

        _draw_one_hand(aria_hand_data)

        return img


    @staticmethod
    def draw_aria_hands_panel(
        img: np.ndarray,
        idx: int,
        aria_hands_data: HandsData,
        opt_v_limit: float,
    ) -> np.ndarray:
        """
        在图像上绘制双手（左右手）的 UI 信息面板。

        每一方代表 _draw_aria_hand_panel()。右侧面板是
        放置在右下角；左下角的左侧面板。

        参数：
            img             (np.ndarray)   : 要绘制的 BGR 图像，形状（高、宽、3）。
            idx             (int)          : 当前帧索引（用于闪烁动画）。
            aria_hands_data (HandsData)：具有 hand_r 和 hand_l 字段的容器。
            opt_v_limit（浮动）

        返回：
            np.ndarray：绘制了双手信息面板的图像。
        """
        img = HandsOps._draw_aria_hand_panel(img, idx, aria_hands_data.hand_r, True, opt_v_limit)
        img = HandsOps._draw_aria_hand_panel(img, idx, aria_hands_data.hand_l, False, opt_v_limit)
        return img


    @staticmethod
    def _draw_advanced_hud(
        img: np.ndarray,
        hand: Any,
        k: np.ndarray,
        d: np.ndarray,
        c2w: np.ndarray,
        grasp_threshold: float,
    ) -> np.ndarray:
        """
        绘制高级 HUD 叠加层，显示捏距、臂长和抓握状态
        作为带注释的连接线和浮动标签框。

        所有显示的距离和连接线均仅根据
        平滑（Opt）世界空间坐标以实现时间稳定性。

        叠加元素：
          - 虚线：拇指尖↔食指尖（颜色随抓握状态变化）。
          - 实线：中点→手腕。
          - 中点附近的浮动标签："DIST(OPT): Xm" + 抓取阈值。
          - 手腕附近的浮动标签："ARM(OPT): Xm"（中点到手腕的距离）。
          - 将标签连接到其锚点的引导线。

        参数：
            img  (np.ndarray): 要绘制的 BGR 图像，形状（高、宽、3）。
            hand (Any)       : HandData 具有优化的世界字段：
                                 wrist_pose_opt_world、thumb/index_translation_opt_world、
                                 midpoint_translation_opt_world、grasp_state。
            k    (np.ndarray): 相机固有矩阵，形状 (3, 3)。
            d    (np.ndarray): 失真系数。
            c2w  (np.ndarray): 相机到世界的 4×4 变换矩阵。
            grasp_threshold（浮动）

        返回：
            np.ndarray：带有就地绘制的 HUD 元素的图像。
                        如果 hand 为 None，则返回原始图像不变，
                        wrist_pose_opt_world 为“无”，或者任何关键点都在视图之外。
        """
        def _project_world_to_2d(
            p_w: Optional[np.ndarray],
            c2w: np.ndarray,
            k: np.ndarray,
            d: np.ndarray
        ) -> Optional[np.ndarray]:
            """将世界空间 3D 点投影到 2D 像素坐标作为 int32 数组。"""
            if p_w is None:
                return None
            p_w = np.asarray(p_w, dtype=np.float64).reshape(3)
            if not np.isfinite(p_w).all() or not np.isfinite(c2w).all():
                return None
            try:
                T_w2c = np.linalg.inv(c2w)
            except np.linalg.LinAlgError:
                return None
            p_c = (T_w2c[:3, :3] @ p_w) + T_w2c[:3, 3]
            if not np.isfinite(p_c).all() or p_c[2] <= 1e-4:
                return None
            # rvec 和 tvec 为零，因为 p_c 已经在相机帧中
            try:
                uv, _ = cv2.projectPoints(
                    p_c.reshape(1, 3),
                    np.zeros(3),
                    np.zeros(3),
                    k,
                    d,
                )
            except (cv2.error, ValueError, OverflowError):
                return None
            u, v = uv.reshape(2)
            if not np.isfinite([u, v]).all():
                return None
            h, w = img.shape[:2]
            if not (0 <= u < w and 0 <= v < h):
                return None
            return np.array(
                [int(round(u)), int(round(v))],
                dtype=np.int32,
            )

        def _draw_dotted_line(
            img: np.ndarray,
            p1: np.ndarray,
            p2: np.ndarray,
            color: Tuple[int, int, int],
            thickness: int = 1,
            gap: int = 4
        ) -> None:
            """通过交替 drawn/skipped 线段在两个 2D 点之间绘制虚线。"""
            dist = np.linalg.norm(p1 - p2)
            if dist < 1e-3: return
            for i in range(0, int(dist), gap * 2):
                start = tuple((p1 + (p2 - p1) * (i / dist)).astype(np.int32))
                end = tuple((p1 + (p2 - p1) * ((i + gap) / dist)).astype(np.int32))
                cv2.line(img, start, end, color, thickness, cv2.LINE_AA)

        # 1.安全检查：确保优化数据存在
        if hand is None or hand.wrist_pose_opt_world is None:
            return img

        # 2.提取优化的世界坐标3D点
        p_wrist_w = hand.wrist_pose_opt_world[:3, 3]
        p_thumb_w = hand.thumb_translation_opt_world
        p_index_w = hand.index_translation_opt_world
        p_mid_w   = hand.midpoint_translation_opt_world

        required_points = (p_wrist_w, p_thumb_w, p_index_w, p_mid_w)
        if any(
            point is None
            or not np.isfinite(np.asarray(point, dtype=np.float64)).all()
            for point in required_points
        ):
            return img

        # 3.计算优化的物理距离（用于文本显示）
        grasp_dist_3d = np.linalg.norm(p_thumb_w - p_index_w)
        wrist_dist_3d = np.linalg.norm(p_mid_w - p_wrist_w)

        # 4. 将 3D 点投影到 2D 像素坐标
        uv_wrist = _project_world_to_2d(p_wrist_w, c2w, k, d)
        uv_thumb = _project_world_to_2d(p_thumb_w, c2w, k, d)
        uv_index = _project_world_to_2d(p_index_w, c2w, k, d)
        uv_mid   = _project_world_to_2d(p_mid_w, c2w, k, d)

        # 如果关键点无法投影（例如，在视图之外），请跳过 HUD 绘制
        if uv_wrist is None or uv_thumb is None or uv_index is None or uv_mid is None:
            return img

        # 5.UI颜色和样式设置
        is_active = hand.grasp_state == 1
        COLOR_GRASP = (0, 215, 255) if not is_active else (0, 69, 255) # 金色 vs 橙红色
        COLOR_WRIST = (255, 255, 0)   # Cyan
        COLOR_TEXT  = (255, 255, 255) # White
        BG_COLOR    = (20, 20, 20)    # 标签盒的深色背景

        # 6. 绘制拇指食指虚线和手腕中点实线
        _draw_dotted_line(img, uv_thumb, uv_index, COLOR_GRASP, 1, 4)
        cv2.line(img, tuple(uv_mid), tuple(uv_wrist), COLOR_WRIST, 1, cv2.LINE_AA)

        # 7.确定标签偏移方向，避免遮挡手中心
        h_img, w_img = img.shape[:2]
        offset_direction = -1 if uv_mid[0] > w_img // 2 else 1

        # --- 标签 1：抓握距离（捏距） --- [已注释掉 — 视觉上太忙了]
        # tag1_pos = uv_mid + np.array([offset_direction * 70, -50])
        # cv2.line(img, 元组(uv_mid), 元组(tag1_pos), (150, 150, 150), 1, cv2.LINE_AA)
        #
        # txt_dist = f"DIST(OPT): {grasp_dist_3d:.3f}m"
        # txt_limit = f"THRS: {grasp_threshold:.3f}m"
        #
        # (tw1, th1), _ = cv2.getTextSize(txt_dist, cv2.FONT_HERSHEY_DUPLEX, 0.32, 1)
        # (tw2, th2), _ = cv2.getTextSize(txt_limit, cv2.FONT_HERSHEY_DUPLEX, 0.3, 1)
        # box_w = max(tw1, tw2) + 12
        # box_h = th1 + th2 + 15
        #
        # rect_start = (tag1_pos[0] - (box_w if offset_direction == -1 else 0), tag1_pos[1] - box_h)
        #
        # cv2.rectangle(img, rect_start, (rect_start[0] + box_w, rect_start[1] + box_h), BG_COLOR, -1)
        # if is_active:
        #     cv2.rectangle(img, rect_start, (rect_start[0] + box_w, rect_start[1] + box_h), COLOR_GRASP, 1, cv2.LINE_AA)
        #
        # cv2.putText(img, txt_dist, (rect_start[0] + 6, rect_start[1] + th1 + 5),
        #             cv2.FONT_HERSHEY_DUPLEX、0.32、COLOR_TEXT、1、cv2.LINE_AA)
        # cv2.putText(img, txt_limit, (rect_start[0] + 6, rect_start[1] + th1 + th2 + 10),
        #             cv2.FONT_HERSHEY_DUPLEX、0.3、COLOR_GRASP、1、cv2.LINE_AA)

        # --- 标签 2：中点到手腕的臂长 ---
        tag2_pos = uv_wrist + np.array([offset_direction * 50, 40])
        cv2.line(img, tuple(uv_wrist), tuple(tag2_pos), (150, 150, 150), 1, cv2.LINE_AA)

        txt_arm = f"ARM(OPT): {wrist_dist_3d:.3f}m"
        (tw3, th3), _ = cv2.getTextSize(txt_arm, cv2.FONT_HERSHEY_DUPLEX, 0.32, 1)
        rect_start2 = (tag2_pos[0] - (tw3 + 12 if offset_direction == -1 else 0), tag2_pos[1] - th3 - 8)

        cv2.rectangle(img, rect_start2, (rect_start2[0] + tw3 + 12, rect_start2[1] + th3 + 12), BG_COLOR, -1)
        cv2.putText(img, txt_arm, (rect_start2[0] + 6, rect_start2[1] + th3 + 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.32, COLOR_WRIST, 1, cv2.LINE_AA)

        return img


    @staticmethod
    def draw_aria_hands_skeleton(
        img: np.ndarray,
        aria_hands_data: HandsData,
        k: np.ndarray,
        d: np.ndarray,
        c2w: np.ndarray,
        grasp_threshold: float,
        full_skeleton: bool = False,
    ) -> np.ndarray:
        """
        完整的每帧可视化入口点：渲染完整的手部覆盖
        双手包括先进的 HUD、中点位姿轴小控件和优化的关键点。

        对于每只检测到的手（右 and/or 左），按顺序应用：
          1. _draw_advanced_hud() — pinch/arm 距离标签和连接线。
          2. _draw_axis() — 中点帧处的 XYZ 姿态轴小控件。
          3. _draw_opt_wrist_thumb_index_only() — 优化的 wrist/thumb/index 节点叠加。

        参数：
            img             (np.ndarray)   : 要绘制的 BGR 图像，形状（高、宽、3）。
            aria_hands_data (HandsData)：具有 hand_r 和 hand_l 字段的容器。
            k               (np.ndarray)   : 相机固有矩阵，形状 (3, 3)。
            d               (np.ndarray)   : 失真系数。
            c2w             (np.ndarray)   : 相机到世界的 4×4 变换矩阵。
            grasp_threshold（浮动）

        返回：
            np.ndarray：为双手渲染所有可视化层的图像。
        """
        for hand in [aria_hands_data.hand_r, aria_hands_data.hand_l]:
            if hand is not None:
                img = HandsOps._draw_advanced_hud(img, hand, k, d, c2w, grasp_threshold)
                if hand.midpoint_pose_opt_world is not None:
                    midpoint_pose_cam = np.linalg.inv(c2w) @ hand.midpoint_pose_opt_world
                    img = HandsOps._draw_axis(img, midpoint_pose_cam, k, d)
                if full_skeleton:
                    # 渲染完整的 21 个关键点手部骨骼（所有手指 + 骨骼）
                    img = HandsOps._draw_skeleton(img, hand)
                else:
                    # 仅渲染简化的平行下颌视图（手腕 + 拇指 + 食指）
                    img = HandsOps._draw_opt_wrist_thumb_index_only(img, hand, k, d, c2w)
        return img
