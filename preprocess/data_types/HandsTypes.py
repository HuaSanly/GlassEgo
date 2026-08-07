import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any,Literal
from numpy.typing import NDArray


class MidpointFrameBuilder:
    """
    在世界坐标系中的手部中点构造正交的夹爪式方向。
    逻辑：
        x = normalize(index_base - thumb_base)
        y_proj = 将 (midpoint - wrist) 投影到与 x 正交的平面
        z = x cross y
    """

    def __init__(
        self,
        eps_norm: float = 1e-6,
        eps_arm: float = 1e-5,
        eps_y: float = 1e-5,
        use_sign_consistency: bool = True
    ):
        """
        使用鲁棒性阈值初始化构建器。
        """
        self.eps_norm = float(eps_norm)
        self.eps_arm = float(eps_arm)
        self.eps_y = float(eps_y)
        self.use_sign_consistency = bool(use_sign_consistency)


    def _safe_normalize(self, v: np.ndarray) -> Optional[np.ndarray]:
        """使用 epsilon 检查标准化向量以避免被零除。"""
        n = float(np.linalg.norm(v))
        if n < self.eps_norm:
            return None
        return v / n


    @staticmethod
    def _make_pose(Rm: np.ndarray, t: np.ndarray) -> np.ndarray:
        """组装 4x4 变换矩阵。"""
        T = np.eye(4)
        T[:3, :3] = Rm
        T[:3, 3] = t
        return T


    def build(
        self,
        thumb_w: np.ndarray,
        index_w: np.ndarray,
        thumb_base_w: np.ndarray,   # Thumb MCP (index 2)
        index_base_w: np.ndarray,   # Index MCP (index 5)
        wrist_w: np.ndarray,
        midpoint_w: np.ndarray,
        prev_R: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        为拇指索引中点构造一个稳定的旋转矩阵。
        Uses Thumb MCP (2) and Index MCP (5) to keep a stable X axis.
        """
        # x = thumb_base -> index_base（避免指尖接触时出现奇异性）
        x_raw = index_base_w - thumb_base_w
        x = self._safe_normalize(x_raw)
        if x is None:
            return prev_R

        # y 轴使用 base_midpoint 进行刚体假设
        base_midpoint_w = (thumb_base_w + index_base_w) / 2.0
        arm = base_midpoint_w - wrist_w
        if float(np.linalg.norm(arm)) < self.eps_arm:
            return prev_R

        y_raw = arm
        # 格拉姆-施密特投影
        y_proj = y_raw - float(np.dot(y_raw, x)) * x
        y = self._safe_normalize(y_proj)
        if y is None:
            return prev_R

        z = self._safe_normalize(np.cross(x, y))
        if z is None:
            return prev_R

        y = self._safe_normalize(np.cross(z, x))
        if y is None:
            return prev_R

        # 标志一致性：防止180°翻转
        if self.use_sign_consistency and prev_R is not None:
            if float(np.dot(prev_R[:, 0], x)) < 0.0:
                x, y = -x, -y
                z = np.cross(x, y)

        return np.column_stack([x, y, z])

@dataclass
class HandsJointAngles:
    """
    根据 21 点 Aria 骨架计算 20 个关节角度（以度为单位）。
    基于屈曲和外展的 emg2pose 定义。
    """
    data: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_keypoints_3d(cls, kpts: np.ndarray):
        """根据骨骼矢量和手掌平面投影计算角度。"""
        if kpts is None or len(kpts) < 21:
            return cls(data={})

        angles = {}

        def get_angle(v1, v2):
            v1_n = v1 / (np.linalg.norm(v1) + 1e-6)
            v2_n = v2 / (np.linalg.norm(v2) + 1e-6)
            return np.degrees(np.arccos(np.clip(np.dot(v1_n, v2_n), -1.0, 1.0)))

        def get_abduction(bone_vec, ref_vec, plane_normal):
            def project(v): return v - np.dot(v, plane_normal) * plane_normal
            return get_angle(project(bone_vec), project(ref_vec))

        # Palm plane: Wrist(0), Index MCP(5), Middle MCP(9).
        v_w_m = kpts[9] - kpts[0]
        v_w_i = kpts[5] - kpts[0]
        palm_normal = np.cross(v_w_i, v_w_m)
        palm_normal /= (np.linalg.norm(palm_normal) + 1e-6)
        v_mid_prox_ref = kpts[10] - kpts[9]

        fingers_map = {
            'Index':  [5, 6, 7, 8],
            'Middle': [9, 10, 11, 12],
            'Ring':   [13, 14, 15, 16],
            'Pinky':  [17, 18, 19, 20],
        }

        for name, idxs in fingers_map.items():
            mcp, pip, dip, tip = idxs
            v_metacarpal = kpts[mcp] - kpts[0]
            v_prox, v_inter, v_dist = kpts[pip]-kpts[mcp], kpts[dip]-kpts[pip], kpts[tip]-kpts[dip]
            angles[f'{name}_MCP_Flex'] = get_angle(v_metacarpal, v_prox)
            angles[f'{name}_PIP_Flex'] = get_angle(v_prox, v_inter)
            angles[f'{name}_DIP_Flex'] = get_angle(v_inter, v_dist)
            angles[f'{name}_MCP_Abd'] = 0.0 if name == 'Middle' else get_abduction(v_prox, v_mid_prox_ref, palm_normal)

        v_thu_cmc = kpts[1] - kpts[0]
        v_thu_metacarpal = kpts[2] - kpts[1]
        v_thu_prox = kpts[3] - kpts[2]
        v_thu_dist = kpts[4] - kpts[3]
        angles['Thumb_CMC_Flex'] = get_angle(v_thu_cmc, v_thu_metacarpal)
        angles['Thumb_CMC_Abd']  = get_abduction(v_thu_metacarpal, v_mid_prox_ref, palm_normal)
        angles['Thumb_MCP_Flex'] = get_angle(v_thu_metacarpal, v_thu_prox)
        angles['Thumb_IP_Flex']  = get_angle(v_thu_prox, v_thu_dist)

        return cls(data=angles)



@dataclass
class HandData:
    """数据类存储单手的本地跟踪数据和世界空间运动学。"""
    d2c: Optional[np.ndarray] = None
    c2w: Optional[np.ndarray] = None
    is_right: bool = None
    confidence: float = None
    wrist_pose: Optional[np.ndarray] = None
    palm_pose: Optional[np.ndarray] = None
    hand_keypoints_3d: Optional[np.ndarray] = None
    hand_keypoints_2d: Optional[np.ndarray] = None
    grasp_state: int = 0
    joint_angles: Optional[HandsJointAngles] = None

    # 手腕运动学
    wrist_pose_raw_world: Optional[np.ndarray] = None
    wrist_pose_opt_world: Optional[np.ndarray] = None
    wrist_lin_vel_raw_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    wrist_ang_vel_raw_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    wrist_lin_vel_opt_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    wrist_ang_vel_opt_world: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # 提示运动学
    index_translation_raw_world: Optional[np.ndarray] = None
    index_translation_opt_world: Optional[np.ndarray] = None
    thumb_translation_raw_world: Optional[np.ndarray] = None
    thumb_translation_opt_world: Optional[np.ndarray] = None

    # 中点运动学
    midpoint_translation_raw_world: Optional[np.ndarray] = None
    midpoint_orientation_raw_world: Optional[np.ndarray] = None
    midpoint_translation_opt_world: Optional[np.ndarray] = None
    midpoint_orientation_opt_world: Optional[np.ndarray] = None
    midpoint_pose_raw_world: Optional[np.ndarray] = None
    midpoint_pose_opt_world: Optional[np.ndarray] = None
    midpoint_lin_vel_raw_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    midpoint_ang_vel_raw_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    midpoint_lin_vel_opt_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    midpoint_ang_vel_opt_world: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # 刚性底座
    thumb_base_raw_world: Optional[np.ndarray] = None
    index_base_raw_world: Optional[np.ndarray] = None
    thumb_base_opt_world: Optional[np.ndarray] = None
    index_base_opt_world: Optional[np.ndarray] = None


    @property
    def distance_midpoint2wrist_raw_world(self) -> Optional[float]:
        if self.wrist_pose_raw_world is None or self.midpoint_translation_raw_world is None: return None
        return float(np.linalg.norm(self.wrist_pose_raw_world[:3, 3] - self.midpoint_translation_raw_world))


    @property
    def distance_midpoint2wrist_opt_world(self) -> Optional[float]:
        if self.wrist_pose_opt_world is None or self.midpoint_pose_opt_world is None: return None
        return float(np.linalg.norm(self.wrist_pose_opt_world[:3, 3] - self.midpoint_pose_opt_world[:3, 3]))


@dataclass
class HandsData:
    idx: int = 0
    ts: int = 0
    hand_r: Optional[HandData] = None
    hand_l: Optional[HandData] = None


@dataclass
class Hands:
    tss: List[int] = field(default_factory=list)
    hands: List[HandsData] = field(default_factory=list)
    data_path: str = None


    def __len__(self) -> int:
        return len(self.tss)


    @staticmethod
    def _safe_list(arr: Any) -> Any:
        if isinstance(arr, np.ndarray):
            return arr.tolist()
        if isinstance(arr, (np.floating, np.integer)):
            return float(arr) if isinstance(arr, np.floating) else int(arr)
        return arr

    
class HandDetection:
    bbox: NDArray[np.float32]             # (4,)，[x1, y1, x2, y2]
    label: Literal["Left", "Right"]
    confidence: float
    landmarks_2d: NDArray[np.float32]     # (21, 2)，像素坐标 [x, y]
    world_landmarks: NDArray[np.float32]  # (21, 3)，世界坐标 [x, y, z]

    @property
    def is_right_int(self) -> int:
        return int(self.label == "Right")
