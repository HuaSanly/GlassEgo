import os
import numpy as np

import mediapipe as mp


MEDIAPIPE_HF_REPO = "Leo-TX/mediapipe-hand"

class MediaPipeHandDetector:

    def __init__(self):
        self._mp = mp

        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "mediapipe", "hand_landmarker.task"
        )

        if not os.path.isfile(model_path):
            try:
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download(
                    repo_id=MEDIAPIPE_HF_REPO,
                    filename="hand_landmarker.task",
                )
                print(f"[MediaPipe] Downloaded hand_landmarker.task from HuggingFace Hub")
            except Exception:
                raise FileNotFoundError(
                    f"MediaPipe hand_landmarker.task not found at {model_path}. "
                    "Download from: https://storage.googleapis.com/mediapipe-models/"
                    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
                )

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=model_path
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = (
            mp.tasks.vision.HandLandmarker.create_from_options(options)
        )
    def detect(self, img_rgb: np.ndarray,timestamp_ms) -> list:
        """
        使用MediaPipe检测手部关键点。
        参数：
            img_rgb: RGB 图像（高、宽、3），uint8。
        返回：
            带键的字典列表：
                'bbox': np.array([x1, y1, x2, y2])
                'label'："Left" 或 "Right"
                'is_right_int'：1 为右，0 为左（对于 ViTDetDataset）
                'confidence'：浮动
                'landmarks_2d': (21, 2) 像素坐标（用于后备）
                'world_landmarks': (21, 3) 以手为中心的米（用于后备）
        """
        h_img, w_img = img_rgb.shape[:2] # 图像宽高

        mp_img = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=img_rgb
        )
        results = self.landmarker.detect_for_video(mp_img,timestamp_ms)

        detections = [] #用于存储最终结果

        if results.hand_landmarks and results.handedness:     #手部关键点和左右手分类结果
            for hand_landmarks, hand_world_lms, handedness_list in zip(   
                results.hand_landmarks,
                results.hand_world_landmarks,
                results.handedness,
            ):
                label = handedness_list[0].category_name # "Left" or "Right"
                confidence = handedness_list[0].score #置信度
                kpts_2d = np.array([[lm.x * w_img, lm.y * h_img] for lm in hand_landmarks], dtype=np.float32) #lm是mediapipe返回的归一化坐标
                kpts_world = np.array([[lm.x, lm.y, lm.z] for lm in hand_world_lms], dtype=np.float32) #以手为中心的世界坐标

                # 计算带填充的边界框
                x_min, y_min = kpts_2d.min(axis=0)
                x_max, y_max = kpts_2d.max(axis=0)
                pad_x = (x_max - x_min) * 0.3
                pad_y = (y_max - y_min) * 0.3
                bbox = np.array([
                    max(0, x_min - pad_x),
                    max(0, y_min - pad_y),
                    min(w_img, x_max + pad_x),
                    min(h_img, y_max + pad_y),
                ], dtype=np.float32)

                detections.append({
                    'bbox': bbox,
                    'label': label,
                    'is_right_int': 1 if label == "Right" else 0,
                    'confidence': confidence,
                    'landmarks_2d': kpts_2d,
                    'world_landmarks': kpts_world,
                })
        return detections