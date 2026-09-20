import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

try:
    from preprocess.data_types.HandsTypes import HandData, Hands, MidpointFrameBuilder
except ModuleNotFoundError:
    from data_types.HandsTypes import HandData, Hands, MidpointFrameBuilder


class SimpleSmoother:
    """Apply Savitzky-Golay smoothing to one complete finite segment."""

    def __init__(self, sg_window: int, sg_polyorder: int, min_valid_frames: int):
        self.sg_window = int(sg_window)
        self.sg_polyorder = int(sg_polyorder)
        self.min_valid_frames = int(min_valid_frames)

    def optimize_positions(self, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("Position smoothing requires an Nx3 array")
        if not np.isfinite(positions).all():
            raise ValueError("Position smoothing requires finite values")
        if len(positions) < self.min_valid_frames:
            return positions.copy()
        window = self.sg_window + self.sg_window % 2
        window = min(window, len(positions) if len(positions) % 2 else len(positions) - 1)
        if window < 5:
            return positions.copy()
        return savgol_filter(
            positions,
            window_length=window,
            polyorder=min(self.sg_polyorder, window - 2),
            axis=0,
            mode="interp",
        )


class HandsTrajectoryOptimizer:
    """Smooth complete hand segments and derive timestamp-aware velocities."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.smoother = SimpleSmoother(cfg.sg_window, cfg.sg_polyorder, cfg.min_valid_frames)
        self.mid_builder = MidpointFrameBuilder()

    def run(self, hands: Hands) -> None:
        self._discard_incomplete_hands(hands)
        self._optimize_all_hands(hands)
        self.assign_velocities(hands)

    @staticmethod
    def remove_excessive_speed_hands(hands: Hands, max_speed_mps: float, pose_attr: str) -> int:
        """Remove the first frame after a world-space wrist speed violation."""
        max_speed_mps = float(max_speed_mps)
        if not np.isfinite(max_speed_mps) or max_speed_mps <= 0.0:
            raise ValueError("max_speed_mps must be finite and positive")
        removed = 0
        for hand_attr in ("hand_r", "hand_l"):
            previous_position = None
            previous_ts = None
            for frame in hands.hands:
                hand = getattr(frame, hand_attr)
                pose = getattr(hand, pose_attr, None) if hand is not None else None
                if not HandsTrajectoryOptimizer._valid_pose(pose):
                    previous_position = None
                    previous_ts = None
                    continue
                position = np.asarray(pose[:3, 3], dtype=np.float64)
                if previous_position is not None:
                    delta_t = (int(frame.ts) - previous_ts) / 1e9
                    if delta_t > 0.0 and np.linalg.norm(position - previous_position) / delta_t > max_speed_mps:
                        setattr(frame, hand_attr, None)
                        removed += 1
                        continue
                previous_position = position.copy()
                previous_ts = int(frame.ts)
        return removed

    def _optimize_all_hands(self, hands: Hands) -> None:
        for hand_attr in ("hand_r", "hand_l"):
            presence = np.array([getattr(frame, hand_attr) is not None for frame in hands.hands], dtype=bool)
            for start, end in self._extract_segments(presence):
                segment = [getattr(frame, hand_attr) for frame in hands.hands[start:end]]
                position_fields = {
                    name: self.smoother.optimize_positions(self._positions(segment, name))
                    for name in (
                        "wrist_pose_raw_world",
                        "thumb_translation_raw_world",
                        "index_translation_raw_world",
                        "thumb_base_raw_world",
                        "index_base_raw_world",
                    )
                }
                midpoint_positions = 0.5 * (
                    position_fields["thumb_translation_raw_world"]
                    + position_fields["index_translation_raw_world"]
                )
                wrist_rotation = None
                midpoint_rotation = None
                alpha = float(self.cfg.ema_alpha_rotation)
                for offset, hand in enumerate(segment):
                    wrist_rotation = self._ema_rotation(
                        hand.wrist_pose_raw_world[:3, :3], wrist_rotation, alpha
                    )
                    wrist_position = position_fields["wrist_pose_raw_world"][offset]
                    hand.wrist_pose_opt_world = self._pose(wrist_rotation, wrist_position)
                    for name in (
                        "thumb_translation_raw_world",
                        "index_translation_raw_world",
                        "thumb_base_raw_world",
                        "index_base_raw_world",
                    ):
                        setattr(hand, name.replace("_raw_", "_opt_"), position_fields[name][offset])

                    midpoint_raw = self.mid_builder.build(
                        hand.thumb_translation_opt_world,
                        hand.index_translation_opt_world,
                        hand.thumb_base_opt_world,
                        hand.index_base_opt_world,
                        wrist_position,
                        midpoint_positions[offset],
                    )
                    if midpoint_raw is None:
                        hand.midpoint_pose_opt_world = None
                        hand.midpoint_translation_opt_world = None
                        hand.midpoint_orientation_opt_world = None
                        midpoint_rotation = None
                        continue
                    midpoint_rotation = self._ema_rotation(midpoint_raw, midpoint_rotation, alpha)
                    hand.midpoint_translation_opt_world = midpoint_positions[offset]
                    hand.midpoint_pose_opt_world = self._pose(midpoint_rotation, midpoint_positions[offset])
                    hand.midpoint_orientation_opt_world = midpoint_rotation.flatten()

    def assign_velocities(self, hands: Hands) -> None:
        """Recompute raw and optimized velocities using each frame's timestamp."""
        for hand_attr in ("hand_r", "hand_l"):
            previous = {"raw": None, "opt": None}
            for frame in hands.hands:
                hand = getattr(frame, hand_attr)
                if hand is None:
                    previous = {"raw": None, "opt": None}
                    continue
                for kind in ("raw", "opt"):
                    wrist_pose = getattr(hand, f"wrist_pose_{kind}_world", None)
                    midpoint_pose = getattr(hand, f"midpoint_pose_{kind}_world", None)
                    midpoint_position = getattr(hand, f"midpoint_translation_{kind}_world", None)
                    if not self._valid_pose(wrist_pose) or not self._valid_pose(midpoint_pose) or not self._valid_point(midpoint_position):
                        previous[kind] = None
                        continue
                    current = (int(frame.ts), wrist_pose, midpoint_pose, np.asarray(midpoint_position, dtype=float))
                    prior = previous[kind]
                    dt = (current[0] - prior[0]) / 1e9 if prior is not None else None
                    wrist_linear = (current[1][:3, 3] - prior[1][:3, 3]) / dt if dt and dt > 0 else np.zeros(3)
                    midpoint_linear = (current[3] - prior[3]) / dt if dt and dt > 0 else np.zeros(3)
                    wrist_angular = self._angular_velocity(prior[1][:3, :3] if prior else None, current[1][:3, :3], dt)
                    midpoint_angular = self._angular_velocity(prior[2][:3, :3] if prior else None, current[2][:3, :3], dt)
                    setattr(hand, f"wrist_lin_vel_{kind}_world", wrist_linear)
                    setattr(hand, f"wrist_ang_vel_{kind}_world", wrist_angular)
                    setattr(hand, f"midpoint_lin_vel_{kind}_world", midpoint_linear)
                    setattr(hand, f"midpoint_ang_vel_{kind}_world", midpoint_angular)
                    previous[kind] = current

    @classmethod
    def _discard_incomplete_hands(cls, hands: Hands) -> None:
        for frame in hands.hands:
            for hand_attr in ("hand_r", "hand_l"):
                hand = getattr(frame, hand_attr)
                if hand is not None and not cls.has_complete_raw_geometry(hand):
                    setattr(frame, hand_attr, None)

    @classmethod
    def has_complete_raw_geometry(cls, hand: HandData) -> bool:
        return all(cls._valid_pose(getattr(hand, name, None)) for name in ("wrist_pose_raw_world", "midpoint_pose_raw_world")) and all(
            cls._valid_point(getattr(hand, name, None))
            for name in ("thumb_translation_raw_world", "index_translation_raw_world", "thumb_base_raw_world", "index_base_raw_world", "midpoint_translation_raw_world")
        )

    @staticmethod
    def _positions(segment, attribute: str) -> np.ndarray:
        positions = []
        for hand in segment:
            value = getattr(hand, attribute, None)
            if isinstance(value, np.ndarray) and value.shape == (4, 4):
                value = value[:3, 3]
            if not HandsTrajectoryOptimizer._valid_point(value):
                raise ValueError(f"Incomplete hand trajectory field: {attribute}")
            positions.append(value)
        return np.asarray(positions, dtype=np.float64)

    @staticmethod
    def _valid_point(value) -> bool:
        value = np.asarray(value) if value is not None else None
        return value is not None and value.shape == (3,) and bool(np.isfinite(value).all())

    @staticmethod
    def _valid_pose(value) -> bool:
        if value is None:
            return False
        pose = np.asarray(value, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            return False
        rotation = pose[:3, :3]
        return np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)

    @staticmethod
    def _pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = Rotation.from_matrix(rotation).as_matrix()
        pose[:3, 3] = translation
        return pose

    @staticmethod
    def _ema_rotation(current: np.ndarray, previous: np.ndarray | None, alpha: float) -> np.ndarray:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("ema_alpha_rotation must be in (0, 1]")
        current = Rotation.from_matrix(current).as_matrix()
        if previous is None:
            return current
        delta = Rotation.from_matrix(previous.T @ current).as_rotvec()
        return previous @ Rotation.from_rotvec(alpha * delta).as_matrix()

    @staticmethod
    def _angular_velocity(previous, current, dt) -> np.ndarray:
        if previous is None or current is None or dt is None or dt <= 0:
            return np.zeros(3)
        return Rotation.from_matrix(previous.T @ current).as_rotvec() / dt

    @staticmethod
    def _extract_segments(presence: np.ndarray):
        segments = []
        index = 0
        while index < len(presence):
            if not presence[index]:
                index += 1
                continue
            end = index + 1
            while end < len(presence) and presence[end]:
                end += 1
            segments.append((index, end))
            index = end
        return segments
