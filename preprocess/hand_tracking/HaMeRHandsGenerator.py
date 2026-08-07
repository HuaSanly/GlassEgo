"""Generate hand data using the HaMeR/OpenPose 21-keypoint convention.

Keypoint order:
    0 Wrist
    1-4 Thumb: CMC, MCP, IP, Tip
    5-8 Index: MCP, PIP, DIP, Tip
    9-12 Middle: MCP, PIP, DIP, Tip
    13-16 Ring: MCP, PIP, DIP, Tip
    17-20 Pinky: MCP, PIP, DIP, Tip
"""

import numpy as np
from tqdm import tqdm

from scipy.spatial.transform import Rotation as R
from typing import Optional, Tuple
from data_types.HandsTypes import Hands, HandsData, HandData, HandsJointAngles, MidpointFrameBuilder
from data_types.CamTypes import Cam


from hand_tracking.MediaPipeHandDetector import MediaPipeHandDetector
from hand_tracking.VitPoseHandDetector import VitPoseHandDetector
from hand_tracking.HaMeRModel import HaMeRModel


class HaMeRHandsGenerator:
    
    HAND_SIZE_WRIST_TO_MIDDLE_MCP_M = 0.085  # 以米为单位的成年人平均手腕到中间 MCP 的距离（用于深度估计）

    def __init__(self,data_path:str,cfg_path:str,cam:Cam):
        self.data_path = data_path
        self.cfg_path = cfg_path
        self.cam = cam


        #初始化手部检测器
        try:
            self.detector = VitPoseHandDetector(device="cuda")
            self._detector_name = "VitPose"
        except (ImportError, FileNotFoundError, Exception) as e:
            print(f"[HaMeR] ViTPose not available ({e}), falling back to MediaPipe detector")
            self.detector = MediaPipeHandDetector()
            self._detector_name = "MediaPipe"
        #初始化HaMeR
        self.hamer_model = HaMeRModel()

        if not self.hamer_model.is_available:
            print(
                "[HaMeR] WARNING: HaMeR model not available. "
                "Falling back to MediaPipe-only 3D recovery."
            )

        # 用于速度计算的缓存
        self.prev_r_cache = None
        self.prev_l_cache = None
        self.prev_r_mid_cache = None
        self.prev_l_mid_cache = None
        self.prev_r_mid_R = None
        self.prev_l_mid_R = None
        self.mid_frame_builder = MidpointFrameBuilder()

    def _recover_absolute_3d_from_hamer(
        self,
        kpts_3d_hamer: np.ndarray,    # (21, 3) HaMeR 相机空间关节
        kpts_2d_mp: np.ndarray,        # (21, 2) MediaPipe 用于深度估计的 2D 检测
        k: np.ndarray,                 # (3, 3) 相机内在
        h_img: int, w_img: int,
    )->Optional[np.ndarray]:
        """"使用针孔模型重新估算HaMeR 3D关节的绝对深度"""
        wrist_2d = kpts_2d_mp[0]
        middle_mcp_2d = kpts_2d_mp[9]

        # 与 HaMeR 3D 关节的物理距离
        physical_dist = float(np.linalg.norm(kpts_3d_hamer[9] - kpts_3d_hamer[0]))
        if physical_dist < 0.01:
            physical_dist = self.HAND_SIZE_WRIST_TO_MIDDLE_MCP_M

        # 2D像素距离
        pixel_dist = float(np.linalg.norm(middle_mcp_2d - wrist_2d))
        if pixel_dist < 5.0:
            return None

        fx = k[0, 0]
        fy = k[1, 1]
        focal = (fx + fy) / 2.0
        z_wrist = focal * physical_dist / pixel_dist

        if z_wrist < 0.05 or z_wrist > 3.0:
            return None

        # 反投影手腕 2D -> 3D 相机坐标系
        cx, cy = k[0, 2], k[1, 2]
        x_wrist = (wrist_2d[0] - cx) * z_wrist / fx
        y_wrist = (wrist_2d[1] - cy) * z_wrist / fy
        wrist_cam = np.array([x_wrist, y_wrist, z_wrist], dtype=np.float32)

        # 使用 HaMeR 相对于手腕的结构偏移
        offsets = kpts_3d_hamer - kpts_3d_hamer[0:1]
        kpts_cam = wrist_cam[np.newaxis, :] + offsets

        if np.any(kpts_cam[:, 2] < 0.01):
            kpts_cam[:, 2] = np.clip(kpts_cam[:, 2], 0.01, None)

        return kpts_cam.astype(np.float32)
    def _build_hand_data(
        self,
        kpts_cam: np.ndarray,   # (21, 3) HaMeR/OpenPose order, camera coordinates
        kpts_2d: np.ndarray,    # (21, 2) HaMeR/OpenPose order, pixel coordinates
        confidence: float,
        c2w: np.ndarray,
        k: np.ndarray,
        h_img: int, w_img: int,
        is_right: bool,
    ) -> HandData:
        """从相机帧 21 个关键点构建 AriaHandData。"""

        # 相机坐标系中的手腕位姿（简单：使用手腕位置+手掌方向）
        wrist_pos_cam = kpts_cam[0]
        palm_center_cam = np.mean(kpts_cam[[5, 9, 13, 17]], axis=0)
        index_mcp_cam = kpts_cam[5]
        middle_mcp_cam = kpts_cam[9]

        # 构建手腕坐标系：Z = 手掌法向，Y = 手腕 -> 手掌方向
        v_wrist_palm = palm_center_cam - wrist_pos_cam
        v_wrist_palm_norm = np.linalg.norm(v_wrist_palm)
        if v_wrist_palm_norm < 1e-6:
            wrist_pose = None
        else:
            y_axis = v_wrist_palm / v_wrist_palm_norm
            v_lateral = index_mcp_cam - middle_mcp_cam
            x_axis = np.cross(y_axis, v_lateral)
            x_norm = np.linalg.norm(x_axis)
            if x_norm < 1e-6:
                wrist_pose = None
            else:
                x_axis /= x_norm
                z_axis = np.cross(x_axis, y_axis)
                z_axis /= (np.linalg.norm(z_axis) + 1e-6)
                y_axis = np.cross(z_axis, x_axis)

                wrist_pose = np.eye(4, dtype=np.float64)
                wrist_pose[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
                wrist_pose[:3, 3] = wrist_pos_cam

        # 抓取检测：基于比率（尺度不变）
        # 拇指尖 (Aria 0) 与食指尖 (Aria 1)，按手掌大小标准化
        thumb_tip = kpts_cam[4]
        index_tip = kpts_cam[8]
        wrist = kpts_cam[0]
        mid_mcp = kpts_cam[9]
        distance = float(np.linalg.norm(thumb_tip - index_tip))
        palm_size = float(np.linalg.norm(mid_mcp - wrist))
        if palm_size > 0.01:
            grasp_ratio = distance / palm_size
            grasp_state = 1 if grasp_ratio < 1.0 else 0
        else:
            grasp_threshold = 0.105
            grasp_state = 1 if distance < grasp_threshold else 0

        # 关节角度
        joint_angles = HandsJointAngles.from_keypoints_3d(kpts_cam)

        # 对 d2c 使用身份，因为我们没有用于基于图像的方法的设备->相机
        d2c = np.eye(4, dtype=np.float64)

        return HandData(
            d2c=d2c,
            c2w=c2w,
            is_right=is_right,
            confidence=confidence,
            wrist_pose=wrist_pose,
            palm_pose=wrist_pose,  # 近似：与手腕相同
            hand_keypoints_3d=kpts_cam,
            hand_keypoints_2d=kpts_2d,
            grasp_state=grasp_state,
            joint_angles=joint_angles,
        )
    def _compute_and_assign_vel(self, hands_data: HandsData,
                                c2w: np.ndarray, dt: float) -> None:
        """计算世界坐标系中的位姿、速度和中点夹爪坐标系。"""
        def robust_rot(matrix):
            try:
                return R.from_matrix(matrix)
            except ValueError:
                U, S, Vt = np.linalg.svd(matrix)
                d = np.linalg.det(U @ Vt)
                if d < 0: U[:, -1] *= -1
                return R.from_matrix(U @ Vt)

        for is_right in [True, False]:
            h_data = hands_data.hand_r if is_right else hands_data.hand_l
            prev_cache = self.prev_r_cache if is_right else self.prev_l_cache
            prev_mid_cache = self.prev_r_mid_cache if is_right else self.prev_l_mid_cache
            prev_R = self.prev_r_mid_R if is_right else self.prev_l_mid_R

            if h_data and h_data.wrist_pose is not None:
                # 手腕 -> 世界
                p_cam = h_data.wrist_pose[:3, 3]
                r_cam = h_data.wrist_pose[:3, :3]
                p_world = (c2w[:3, :3] @ p_cam) + c2w[:3, 3]
                r_world = c2w[:3, :3] @ r_cam

                h_data.wrist_pose_raw_world = np.eye(4)
                h_data.wrist_pose_raw_world[:3, :3] = r_world
                h_data.wrist_pose_raw_world[:3, 3] = p_world

                if prev_cache is not None:
                    h_data.wrist_lin_vel_raw_world = (p_world - prev_cache['pos']) / dt
                    rel = prev_cache['rot'].T @ r_world
                    h_data.wrist_ang_vel_raw_world = robust_rot(rel).as_rotvec() / dt

                cache_val = {'pos': p_world, 'rot': r_world}
                if is_right: self.prev_r_cache = cache_val
                else: self.prev_l_cache = cache_val

                # 中点夹爪坐标系
                if h_data.hand_keypoints_3d is not None and len(h_data.hand_keypoints_3d) >= 21:
                    thumb_w = (c2w[:3, :3] @ h_data.hand_keypoints_3d[4]) + c2w[:3, 3]
                    index_w = (c2w[:3, :3] @ h_data.hand_keypoints_3d[8]) + c2w[:3, 3]
                    thumb_base_w = (c2w[:3, :3] @ h_data.hand_keypoints_3d[2]) + c2w[:3, 3]
                    index_base_w = (c2w[:3, :3] @ h_data.hand_keypoints_3d[5]) + c2w[:3, 3]

                    h_data.thumb_translation_raw_world = thumb_w
                    h_data.index_translation_raw_world = index_w
                    h_data.thumb_base_raw_world = thumb_base_w
                    h_data.index_base_raw_world = index_base_w

                    midpoint_w = (thumb_w + index_w) / 2.0
                    h_data.midpoint_translation_raw_world = midpoint_w

                    R_mid = self.mid_frame_builder.build(
                        thumb_w=thumb_w, index_w=index_w,
                        thumb_base_w=thumb_base_w, index_base_w=index_base_w,
                        wrist_w=p_world, midpoint_w=midpoint_w, prev_R=prev_R,
                    )
                    if R_mid is None:
                        R_mid = prev_R if prev_R is not None else r_world.copy()

                    h_data.midpoint_pose_raw_world = np.eye(4)
                    h_data.midpoint_pose_raw_world[:3, :3] = R_mid
                    h_data.midpoint_pose_raw_world[:3, 3] = midpoint_w
                    h_data.midpoint_orientation_raw_world = R_mid.flatten()

                    if prev_mid_cache is not None:
                        h_data.midpoint_lin_vel_raw_world = (midpoint_w - prev_mid_cache['pos']) / dt
                        rel = prev_mid_cache['rot'].T @ R_mid
                        h_data.midpoint_ang_vel_raw_world = robust_rot(rel).as_rotvec() / dt

                    cache_mid = {'pos': midpoint_w, 'rot': R_mid}
                    if is_right:
                        self.prev_r_mid_cache = cache_mid
                        self.prev_r_mid_R = R_mid
                    else:
                        self.prev_l_mid_cache = cache_mid
                        self.prev_l_mid_R = R_mid
    def get_hands_data(self)->Hands:
        """完整对外pipeline"""
        hands = Hands(data_path=self.data_path)
        dt = 1.0 / self.cam.fps

        for i,cam_data in enumerate(tqdm(self.cam.cam,desc="Hands")):
            #图像获取
            img = cam_data.img  #rgb
            h_img, w_img = img.shape[:2]
            k = cam_data.k
            c2w = cam_data.c2w

            #手部检测
            if self._detector_name == "MediaPipe":
                timestamp_ms = int(i * 1000.0 / self.cam.fps)
                detections = self.detector.detect(img, timestamp_ms)
            else:
                detections = self.detector.detect(img)

            hand_r = None
            hand_l = None

            fx = k[0, 0]
            fy = k[1, 1]
            focal = (fx + fy) / 2.0  #焦距

            for hand in detections:
                label = hand['label']
                det_confidence = hand['confidence']
                # 第 2 阶段：HaMeR 从裁剪中恢复 3D 网格
                hamer_result = self.hamer_model.predict_from_crop(
                    img, hand['bbox'],
                    is_right=hand['is_right_int'],
                    focal_length=focal,
                )

                if hamer_result is not None:
                    kpts_cam = hamer_result['joints_3d']   # (21, 3) 在相机空间
                    kpts_2d = hamer_result['joints_2d']    # (21, 2) 像素坐标
                    hamer_confidence = hamer_result['confidence']

                    final_confidence = det_confidence * hamer_confidence #ego模式下置信度可能较低

                    wrist_z = kpts_cam[0,2]  # (4, 4) 相机空间
                    if wrist_z < 0.05 or wrist_z > 3.0:
                        # HaMeR 的深度不可靠（可能是由于焦距
                        # 不匹配 — HaMeR 假设 f≈5000，但 Aria 的 f≈320）。
                        # 从像素大小+真实焦点重新估计绝对深度。
                        kpts_cam = self._recover_absolute_3d_from_hamer(kpts_cam, hand['landmarks_2d'], k, h_img, w_img,)
                        if kpts_cam is None:
                            continue
                        # 现在深度已修正，重新计算置信度 
                        final_confidence = max(float(det_confidence), 0.50)
                    h_data = self._build_hand_data(
                        kpts_cam, kpts_2d, final_confidence,
                        c2w, k, h_img, w_img,
                        is_right=(label == "Right"),
                    )
                    if label == "Right":
                        if hand_r is None or final_confidence > hand_r.confidence:
                            hand_r = h_data
                    else:
                        if hand_l is None or final_confidence > hand_l.confidence:
                            hand_l = h_data

            frame_data = HandsData(cam_data.idx, cam_data.ts, hand_r, hand_l)

            # 计算速度和中点坐标系
            self._compute_and_assign_vel(frame_data, c2w, dt)

            hands.hands.append(frame_data)
            hands.tss.append(cam_data.ts)
        return hands
