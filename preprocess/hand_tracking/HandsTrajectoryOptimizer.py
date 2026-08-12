
import numpy as np
from typing import Optional, List, Tuple, Any
from scipy.signal import savgol_filter

from data_types.HandsTypes import MidpointFrameBuilder, HandData, Hands


class SimpleSmoother:
    """
    用于轨迹级平滑的实用程序类。
    处理 1D 和 3D 信号过滤和间隙管理。
    """

    def __init__(
        self,
        dt: float,
        sg_window: int,
        sg_polyorder: int,
        min_valid_frames: int,
        fill_max_gap: int,
    ):
        """
        使用过滤参数初始化平滑器。

        参数：
            dt (float): 帧之间的时间步长(1/FPS)。
            sg_window (int)：Savitzky-Golay 窗口大小（必须为奇数）。
            sg_polyorder (int)：Savitzky-Golay 多项式阶数。
            min_valid_frames (int)：触发优化所需的最小帧数。
            fill_max_gap (int)：要插值的最大连续 NaN。
        """
        self.dt = float(dt)
        self.sg_window = sg_window
        self.sg_polyorder = sg_polyorder
        self.min_valid_frames = min_valid_frames
        self.fill_max_gap = fill_max_gap


    def _interp_nans_1d(self, x: np.ndarray) -> np.ndarray:
        """
        执行线性插值以填充一维数组中的所有 NaN。

        参数：
            x (np.ndarray): 具有可能 NaN 的一维数组。
        返回：
            np.ndarray：插值一维数组。
        """
        x = x.copy()
        nan = np.isnan(x)
        # 若有效值小于两个，则不进行插值
        if np.sum(~nan) < 2:
            return x
        idx = np.arange(len(x))
        x[nan] = np.interp(idx[nan], idx[~nan], x[~nan])
        return x


    def _fill_gaps_xyz(self, xyz: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """
        使用线性插值填充 3D 轨迹中的短时间间隙。

        参数：
            xyz (np.ndarray): (N, 3) 轨迹数组。
            valid (np.ndarray): (N,) 有效帧的布尔掩码。
        返回：
            np.ndarray：间隙填充轨迹。
        """
        out = xyz.copy()
        out[~valid] = np.nan
        idx_valid = np.where(valid)[0]
        if len(idx_valid) < 2:
            return out

        for a, b in zip(idx_valid[:-1], idx_valid[1:]):
            gap = b - a - 1
            if 0 < gap <= self.fill_max_gap:
                for d in range(3):
                    out[a+1:b, d] = np.linspace(out[a, d], out[b, d], gap + 2)[1:-1]
        return out


    def optimize_positions(self, pos: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """
        使用 Savitzky-Golay 过滤进行位置平滑的主条目。

        参数：
            pos (np.ndarray): (N, 3) 原始位置数组。
            valid (np.ndarray): (N,) 布尔掩码。
        返回：
            np.ndarray：平滑的 (N, 3) 位置数组。
        """
        if len(pos) < self.min_valid_frames or np.sum(valid) < self.min_valid_frames:
            return pos.copy()

        # 1.过滤前填充小间隙
        p = self._fill_gaps_xyz(pos, valid)
        out = p.copy()

        # 2.窗口大小调整（必须为奇数且<=长度）
        w = self.sg_window
        if w % 2 == 0: w += 1
        w = min(w, len(p) if len(p) % 2 == 1 else len(p) - 1)

        if w < 5:
            return np.nan_to_num(out, nan=0.0)

        # 3. 逐轴应用 Savitzky-Golay
        for d in range(3):
            xd = self._interp_nans_1d(out[:, d])
            try:
                out[:, d] = savgol_filter(xd, window_length=w, polyorder=min(self.sg_polyorder, w-2), mode="interp")
            except:
                out[:, d] = xd
        return out



class HandsTrajectoryOptimizer:
    """
    协调手部运动学优化过程，同时处理
    双手跨过整个序列。
    """

    def __init__(self, cfg: Any, dt: float):
        """
        参数：
            cfg (ConfigBox/Namespace): hand_tracking.trajectory 配置
            dt (float): 时间步长(1/FPS)。
        """
        self.cfg = cfg
        self.dt = dt
        self.smoother = SimpleSmoother(dt,
                                       self.cfg.sg_window,
                                       self.cfg.sg_polyorder,
                                       self.cfg.min_valid_frames,
                                       self.cfg.fill_max_gap_frames
                                       )
        self.mid_builder = MidpointFrameBuilder()


    def run(self, hands: Hands) -> None:
        """
        执行完整的手部运动学优化流程。

        参数：
            aria_hands (AriaHands)：要修改的序列级手容器。
        """
        self._optimize_all_hands(hands)
        print(f"[***] Smoothing pipeline finished using AriaHandsOptimizer.")


    def _optimize_all_hands(self, aria_hands: Hands) -> None:
        """
        迭代双手并应用基于分段的平滑。
        """
        for hand_attr in ["hand_r", "hand_l"]:
            presence = np.array([getattr(f, hand_attr) is not None for f in aria_hands.hands], dtype=bool)
            if np.sum(presence) < self.cfg.min_valid_frames:
                continue

            # 提取连续检测片段
            segments = self._extract_segments(presence)

            for (s, e) in segments:
                seg_len = e - s
                if seg_len < self.cfg.min_valid_frames:
                    continue

                # 准备段数据
                frames = aria_hands.hands[s:e]
                hands = [getattr(fr, hand_attr) for fr in frames]
                valid_mask = np.array([h is not None for h in hands], dtype=bool)

                # --- 步骤 1：位置平滑 (Savitzky-Golay) ---
                wrist_pos_raw = self._get_raw_pos_array(hands, "wrist_pose_raw_world")
                thumb_pos_raw = self._get_raw_pos_array(hands, "thumb_translation_raw_world")
                index_pos_raw = self._get_raw_pos_array(hands, "index_translation_raw_world")
                thumb_base_raw = self._get_raw_pos_array(hands, "thumb_base_raw_world")
                index_base_raw = self._get_raw_pos_array(hands, "index_base_raw_world")

                wrist_pos_opt = self.smoother.optimize_positions(wrist_pos_raw, valid_mask)
                thumb_pos_opt = self.smoother.optimize_positions(thumb_pos_raw, valid_mask)
                index_pos_opt = self.smoother.optimize_positions(index_pos_raw, valid_mask)
                thumb_base_opt = self.smoother.optimize_positions(thumb_base_raw, valid_mask)
                index_base_opt = self.smoother.optimize_positions(index_base_raw, valid_mask)

                mid_pos_opt = 0.5 * (thumb_pos_opt + index_pos_opt)

                # --- 步骤 2：方向平滑 (EMA + 基础重新正交) ---
                # 手腕和中点 EMA 缓存
                wrist_x_ema, wrist_y_ema = None, None
                mid_x_ema, mid_y_ema = None, None
                mid_prev_R = None

                for k in range(seg_len):
                    h = hands[k]
                    if h is None: continue

                    # A. 更新手腕位姿（平滑位置 + EMA 方向）
                    h.wrist_pose_opt_world = np.eye(4)
                    h.wrist_pose_opt_world[:3, 3] = wrist_pos_opt[k]

                    if h.wrist_pose_raw_world is not None:
                        wr_raw_R = h.wrist_pose_raw_world[:3, :3]
                        wr_x, wrist_x_ema = self._ema_unit_vec(wr_raw_R[:, 0], wrist_x_ema, alpha=self.cfg.ema_alpha_x)
                        wr_y, wrist_y_ema = self._ema_unit_vec(wr_raw_R[:, 1], wrist_y_ema, alpha=self.cfg.ema_alpha_y)

                        # 手腕的 Gram-Schmidt 正交化
                        wr_z = np.cross(wr_x, wr_y)
                        wr_z /= (np.linalg.norm(wr_z) + 1e-6)
                        wr_y = np.cross(wr_z, wr_x)
                        h.wrist_pose_opt_world[:3, :3] = np.column_stack([wr_x, wr_y, wr_z])

                    # B. 更新平滑的指尖和 MCP 底座
                    h.thumb_translation_opt_world = thumb_pos_opt[k]
                    h.index_translation_opt_world = index_pos_opt[k]
                    h.thumb_base_opt_world = thumb_base_opt[k]
                    h.index_base_opt_world = index_base_opt[k]

                    # C. 更新中点位姿（平滑位置 + 夹爪坐标系重建）
                    # 使用平滑后的刚性 MCP 底座重建夹爪坐标系。
                    mid_R_rebuild = self.mid_builder.build(
                        thumb_w=thumb_pos_opt[k],
                        index_w=index_pos_opt[k],
                        thumb_base_w=thumb_base_opt[k],
                        index_base_w=index_base_opt[k],
                        wrist_w=wrist_pos_opt[k],
                        midpoint_w=mid_pos_opt[k],
                        prev_R=mid_prev_R
                    )

                    # 如果构造失败，则退回到平滑的手腕方向
                    if mid_R_rebuild is None:
                        mid_R_rebuild = mid_prev_R if mid_prev_R is not None else h.wrist_pose_opt_world[:3, :3]

                    # EMA 中点基向量的平滑
                    mid_x, mid_x_ema = self._ema_unit_vec(mid_R_rebuild[:, 0], mid_x_ema, alpha=self.cfg.ema_alpha_x)
                    mid_y, mid_y_ema = self._ema_unit_vec(mid_R_rebuild[:, 1], mid_y_ema, alpha=self.cfg.ema_alpha_y)

                    # 中点的 Gram-Schmidt 正交化
                    mid_z = np.cross(mid_x, mid_y)
                    mid_z /= (np.linalg.norm(mid_z) + 1e-6)
                    mid_y = np.cross(mid_z, mid_x)
                    mid_R_opt = np.column_stack([mid_x, mid_y, mid_z])

                    h.midpoint_translation_opt_world = mid_pos_opt[k]
                    h.midpoint_pose_opt_world = np.eye(4)
                    h.midpoint_pose_opt_world[:3, :3] = mid_R_opt
                    h.midpoint_pose_opt_world[:3, 3] = mid_pos_opt[k]
                    h.midpoint_orientation_opt_world = mid_R_opt.flatten()
                    mid_prev_R = mid_R_opt

                # --- 步骤 3 和 4：速度计算（有限差分）---
                self._assign_linear_vel_from_pos(hands, self.dt, key="wrist")
                self._assign_linear_vel_from_pos(hands, self.dt, key="midpoint")
                self._assign_angular_vel_from_rot(hands, self.dt, key="wrist")
                self._assign_angular_vel_from_rot(hands, self.dt, key="midpoint")


    @staticmethod
    def _get_raw_pos_array(hands: List[Optional[HandData]], attr_name: str) -> np.ndarray:
        """用于将位置向量或位姿翻译提取到 numpy 数组中的实用程序。"""
        res = []
        for h in hands:
            val = getattr(h, attr_name) if h else None
            if val is not None and val.shape == (4, 4):
                val = val[:3, 3]
            res.append(val if val is not None else np.zeros(3))
        return np.array(res)


    @staticmethod
    def _extract_segments(presence: np.ndarray) -> List[Tuple[int, int]]:
        """标识有效手部检测的连续 start/end 索引。"""
        segs = []
        T, i = len(presence), 0
        while i < T:
            if not presence[i]:
                i += 1
                continue
            j = i + 1
            while j < T and presence[j]:
                j += 1
            segs.append((i, j))
            i = j
        return segs


    def _ema_unit_vec(self, v: np.ndarray, v_ema: Optional[np.ndarray], alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        将指数移动平均线应用于具有符号一致性的单位向量。
        """
        v = np.asarray(v, dtype=np.float64)
        v /= (np.linalg.norm(v) + 1e-6)

        if v_ema is None:
            return v, v.copy()

        # 确保符号一致性，防止平滑过程中出现 180 度跳跃
        if float(np.dot(v, v_ema)) < 0.0:
            v = -v

        v_new = (1.0 - float(alpha)) * v_ema + float(alpha) * v
        v_new /= (np.linalg.norm(v_new) + 1e-6)
        return v_new, v_new.copy()


    @staticmethod
    def _assign_linear_vel_from_pos(hands: List[HandData], dt: float, key: str = "wrist") -> None:
        """计算线速度 v = (p_curr - p_prev) / dt。"""
        prev_p = None
        for h in hands:
            if h is None:
                prev_p = None
                continue
            p = h.wrist_pose_opt_world[:3, 3] if key == "wrist" else h.midpoint_translation_opt_world
            if p is None:
                prev_p = None
                continue
            vel = (p - prev_p) / dt if prev_p is not None else np.zeros(3)
            if key == "wrist": h.wrist_lin_vel_opt_world = vel
            else: h.midpoint_lin_vel_opt_world = vel
            prev_p = p.copy()


    @staticmethod
    def _assign_angular_vel_from_rot(hands: List[HandData], dt: float, key: str = "wrist") -> None:
        """
        使用旋转对数图计算角速度：w = log(R_prev.T @ R_curr) / dt。
        """
        prev_R = None
        for h in hands:
            if h is None:
                prev_R = None
                continue

            curr_pose = h.wrist_pose_opt_world if key == "wrist" else h.midpoint_pose_opt_world
            if curr_pose is None:
                prev_R = None
                continue

            curr_R = curr_pose[:3, :3]

            if prev_R is None:
                ang_vel = np.zeros(3)
            else:
                try:
                    # 计算相对旋转并映射到旋转向量（轴角空间）
                    from scipy.spatial.transform import Rotation as R_lib
                    rel_rot_mat = prev_R.T @ curr_R
                    ang_vel = R_lib.from_matrix(rel_rot_mat).as_rotvec() / dt
                except Exception:
                    ang_vel = np.zeros(3)

            if key == "wrist": h.wrist_ang_vel_opt_world = ang_vel
            else: h.midpoint_ang_vel_opt_world = ang_vel

            prev_R = curr_R.copy()
