import numpy as np


class PhaseSegmentationOps:
    """HumanEgo 阶段切割使用的时序启发式操作。"""

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
    def fill_short_false_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
        result = np.asarray(mask, dtype=bool).copy()
        if max_gap <= 0:
            return result
        for start, end, value in PhaseSegmentationOps.find_segments(result):
            if value:
                continue
            if (
                start > 0
                and end < len(result) - 1
                and end - start + 1 <= max_gap
                and result[start - 1]
                and result[end + 1]
            ):
                result[start : end + 1] = True
        return result

    @staticmethod
    def remove_short_true_runs(mask: np.ndarray, min_length: int) -> np.ndarray:
        result = np.asarray(mask, dtype=bool).copy()
        for start, end, value in PhaseSegmentationOps.find_segments(result):
            if value and end - start + 1 < min_length:
                result[start : end + 1] = False
        return result

    @staticmethod
    def median_filter_modes(mode: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return np.asarray(mode, dtype=np.int32).copy()
        if window % 2 == 0:
            window += 1
        values = np.asarray(mode, dtype=np.int32)
        pad = window // 2
        padded = np.pad(values, (pad, pad), mode="edge")
        return np.asarray(
            [np.median(padded[index : index + window]) for index in range(len(values))],
            dtype=np.int32,
        )

    @staticmethod
    def merge_short_mode_runs(mode: np.ndarray, min_length: int) -> np.ndarray:
        result = np.asarray(mode, dtype=np.int32).copy()
        if min_length <= 1:
            return result
        for start, end, value in PhaseSegmentationOps.find_segments(result):
            if end - start + 1 >= min_length:
                continue
            replacement = (
                result[end + 1]
                if end + 1 < len(result)
                else result[start - 1]
                if start > 0
                else value
            )
            result[start : end + 1] = replacement
        return result

    @staticmethod
    def compute_stop_mask(
        linear_speed: np.ndarray,
        angular_speed: np.ndarray,
        v_threshold: float,
        w_threshold: float,
        hold_frames: int,
        debounce_frames: int,
    ) -> np.ndarray:
        below_threshold = (
            (np.asarray(linear_speed) < v_threshold)
            & (np.asarray(angular_speed) < w_threshold)
        )
        result = PhaseSegmentationOps.fill_short_false_gaps(
            below_threshold,
            debounce_frames,
        )
        return PhaseSegmentationOps.remove_short_true_runs(result, hold_frames)

    @staticmethod
    def apply_yaw_veto(
        stop: np.ndarray,
        yaw_unwrapped_deg: np.ndarray,
        min_frames: int,
        veto_deg: float,
    ) -> np.ndarray:
        result = PhaseSegmentationOps.remove_short_true_runs(stop, min_frames)
        for start, end, value in PhaseSegmentationOps.find_segments(result):
            if value and abs(yaw_unwrapped_deg[end] - yaw_unwrapped_deg[start]) >= veto_deg:
                result[start : end + 1] = False
        return result

    @staticmethod
    def delay_stop_start(stop: np.ndarray, offset_frames: int) -> np.ndarray:
        if offset_frames <= 0:
            return np.asarray(stop, dtype=bool).copy()
        result = np.zeros(len(stop), dtype=bool)
        for start, end, value in PhaseSegmentationOps.find_segments(stop):
            if value and start + offset_frames <= end:
                result[start + offset_frames : end + 1] = True
        return result

    @staticmethod
    def compute_modes(
        stop: np.ndarray,
        linear_speed: np.ndarray,
        angular_speed: np.ndarray,
        w_rot_threshold: float,
        v_rot_max: float,
        median_window: int,
        min_run_frames: int,
        transition_frames: int,
    ) -> np.ndarray:
        mode = np.where(
            stop,
            0,
            np.where(
                (angular_speed >= w_rot_threshold)
                & (linear_speed <= 2.0 * v_rot_max),
                2,
                1,
            ),
        ).astype(np.int32)
        mode = PhaseSegmentationOps.median_filter_modes(mode, median_window)
        mode = PhaseSegmentationOps.merge_short_mode_runs(mode, min_run_frames)

        if transition_frames > 0:
            base_mode = mode.copy()
            for _, end, _ in PhaseSegmentationOps.find_segments(base_mode)[:-1]:
                next_start = end + 1
                next_end = min(len(mode), next_start + transition_frames)
                mode[next_start:next_end] = 3
        return mode

    @staticmethod
    def refine_stop_with_hand_speed(
        mode: np.ndarray,
        hand_speed: np.ndarray,
        velocity_threshold: float,
        stable_frames: int,
        boundary_frames: int,
        min_valid_fraction: float,
    ) -> np.ndarray:
        result = np.asarray(mode, dtype=np.int32).copy()
        for start, end, value in PhaseSegmentationOps.find_segments(result):
            if value != 0:
                continue
            speeds = hand_speed[start : end + 1]
            valid = np.isfinite(speeds)
            if not len(speeds) or valid.mean() < min_valid_fraction:
                continue

            stable = np.zeros(len(speeds), dtype=bool)
            for index in range(len(speeds) - stable_frames + 1):
                window = speeds[index : index + stable_frames]
                if np.all(np.isfinite(window)) and np.mean(window) < velocity_threshold:
                    stable[index : index + stable_frames] = True

            stable_indices = np.flatnonzero(stable)
            if len(stable_indices) == 0:
                result[start : end + 1] = 3
                continue
            keep_start = max(start, start + int(stable_indices[0]) - boundary_frames)
            keep_end = min(end, start + int(stable_indices[-1]) + boundary_frames)
            result[start:keep_start] = 3
            result[keep_end + 1 : end + 1] = 3
        return result

    @staticmethod
    def inject_finished(mode: np.ndarray, finished_frames: int) -> np.ndarray:
        result = np.asarray(mode, dtype=np.int32).copy()
        if finished_frames > 0 and len(result):
            result[max(0, len(result) - finished_frames) :] = 4
        return result
