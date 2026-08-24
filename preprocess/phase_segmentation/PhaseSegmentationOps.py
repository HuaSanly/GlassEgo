import numpy as np


class PhaseSegmentationOps:
    """阶段评分和时序平滑的纯函数。"""

    @staticmethod
    def median_filter_values(values: np.ndarray, window: int) -> np.ndarray:
        """使用边界复制的滑动中值过滤连续特征。"""
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(values) == 0 or window <= 1:
            return values.copy()
        if window % 2 == 0:
            window += 1
        pad = window // 2
        padded = np.pad(values, (pad, pad), mode="edge")
        return np.asarray(
            [np.nanmedian(padded[index : index + window]) for index in range(len(values))],
            dtype=np.float64,
        )

    @staticmethod
    def sigmoid_score(values: np.ndarray, reference: float, scale: float) -> np.ndarray:
        """把连续运动特征映射为 0 到 1 的启发式证据分数。"""
        if scale <= 0.0:
            raise ValueError("Score scale must be positive")
        values = np.asarray(values, dtype=np.float64)
        logits = np.clip((values - float(reference)) / float(scale), -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-logits))

    @staticmethod
    def combine_non_operation_score(
        camera_motion_score: np.ndarray,
        hand_presence_score: np.ndarray,
        hand_motion_score: np.ndarray,
        grasp_score: np.ndarray,
        hand_available: np.ndarray,
        hand_presence_weight: float,
        hand_motion_weight: float,
        grasp_weight: float,
        hand_operation_weight: float,
        no_hand_weight: float,
    ) -> np.ndarray:
        """融合相机非操作证据和手部操作证据。"""
        camera_motion_score = np.asarray(camera_motion_score, dtype=np.float64)
        presence = np.asarray(hand_presence_score, dtype=np.float64)
        motion = np.asarray(hand_motion_score, dtype=np.float64)
        grasp = np.asarray(grasp_score, dtype=np.float64)
        hand_available = np.asarray(hand_available, dtype=bool)
        lengths = {
            len(camera_motion_score),
            len(presence),
            len(motion),
            len(grasp),
            len(hand_available),
        }
        if len(lengths) != 1:
            raise ValueError("Phase evidence arrays must have the same length")
        if not np.all(np.isfinite(camera_motion_score)):
            raise ValueError("Camera motion score must be finite")
        if not 0.0 <= hand_operation_weight <= 1.0:
            raise ValueError("hand_operation_weight must be between 0 and 1")
        if not 0.0 <= no_hand_weight <= 1.0:
            raise ValueError("no_hand_weight must be between 0 and 1")
        weights = np.asarray(
            [hand_presence_weight, hand_motion_weight, grasp_weight],
            dtype=np.float64,
        )
        if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
            raise ValueError("Hand evidence weights must be non-negative and non-zero")
        weights /= weights.sum()
        hand_support = (
            weights[0] * np.nan_to_num(presence, nan=0.0)
            + weights[1] * np.nan_to_num(motion, nan=0.0)
            + weights[2] * np.nan_to_num(grasp, nan=0.0)
        )
        hand_support = np.where(hand_available, hand_support, 0.0)
        persistent_no_hand = np.where(
            hand_available,
            1.0 - np.nan_to_num(presence, nan=0.0),
            0.0,
        )
        score = (
            camera_motion_score
            - float(hand_operation_weight) * hand_support
            + float(no_hand_weight) * persistent_no_hand
        )
        return np.clip(score, 0.0, 1.0)

    @staticmethod
    def classify_binary_phases(
        non_operation_score: np.ndarray,
        enter_threshold: float,
        exit_threshold: float,
        enter_frames: int,
        exit_frames: int,
        min_non_operation_frames: int,
    ) -> np.ndarray:
        """用滞回和最短持续时间输出二值阶段。"""
        scores = np.asarray(non_operation_score, dtype=np.float64).reshape(-1)
        if not 0.0 <= exit_threshold < enter_threshold <= 1.0:
            raise ValueError("Phase thresholds must satisfy 0 <= exit < enter <= 1")
        if enter_frames < 1 or exit_frames < 1 or min_non_operation_frames < 1:
            raise ValueError("Phase dwell lengths must be positive")

        non_operation = np.zeros(len(scores), dtype=bool)
        state = False
        high_count = 0
        low_count = 0
        for index, score in enumerate(scores):
            if not state:
                high_count = high_count + 1 if score >= enter_threshold else 0
                if high_count >= enter_frames:
                    state = True
                    non_operation[index - enter_frames + 1 : index + 1] = True
                continue

            non_operation[index] = True
            low_count = low_count + 1 if score <= exit_threshold else 0
            if low_count >= exit_frames:
                state = False
                non_operation[index - exit_frames + 1 : index + 1] = False
                low_count = 0

        non_operation = PhaseSegmentationOps.remove_short_true_runs(
            non_operation,
            min_non_operation_frames,
        )
        return non_operation.astype(np.int32)

    @staticmethod
    def find_segments(values: np.ndarray) -> list[tuple[int, int, int]]:
        values = np.asarray(values, dtype=np.int32).reshape(-1)
        if len(values) == 0:
            return []
        segments = []
        start = 0
        value = int(values[0])
        for index in range(1, len(values)):
            if values[index] != value:
                segments.append((start, index - 1, value))
                start = index
                value = int(values[index])
        segments.append((start, len(values) - 1, value))
        return segments

    @staticmethod
    def remove_short_true_runs(mask: np.ndarray, min_length: int) -> np.ndarray:
        result = np.asarray(mask, dtype=bool).copy()
        for start, end, value in PhaseSegmentationOps.find_segments(result):
            if value and end - start + 1 < min_length:
                result[start : end + 1] = False
        return result
