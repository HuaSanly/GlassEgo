from dataclasses import dataclass

from preprocess.data_types.VIOTypes import (
    ARIA_MPS_INITIAL_HEADING,
    ARIA_MPS_WORLD_FRAME,
    ARIA_MPS_WORLD_ORIGIN,
    ARIA_MPS_YAW_CONVENTION,
)


OPERATION_MODE = 0
NON_OPERATION_MODE = 1
FINISHED_MODE = 2
PHASE_NAMES = {
    OPERATION_MODE: "OPERATION",
    NON_OPERATION_MODE: "NON_OPERATION",
    FINISHED_MODE: "FINISHED",
}
PHASE_SCHEMA_VERSION = 4

FORCED_NON_OPERATION_PREFIX_FRAMES = 120
MIN_CLASSIFIED_MIDDLE_FRAMES = 50
FINISHED_TAIL_FRAMES = 10

# 120 fixed non-operation frames + at least 50 classified frames + 10 finished frames.
MIN_UNIT_FRAME_COUNT = (
    FORCED_NON_OPERATION_PREFIX_FRAMES
    + MIN_CLASSIFIED_MIDDLE_FRAMES
    + FINISHED_TAIL_FRAMES
)


@dataclass(frozen=True)
class PhaseFrame:
    """单帧操作阶段、运动学和手部证据。"""

    frame_idx: int
    timestamp_ns: int
    mode: int
    linear_speed_mps: float
    angular_speed_rad_s: float
    yaw_unwrapped_deg: float
    operation_confidence: float = 0.0
    non_operation_confidence: float = 0.0
    camera_motion_score: float = 0.0
    hand_presence_score: float | None = None
    hand_motion_score: float | None = None
    grasp_score: float | None = None
    hand_evidence_available: bool = False

    @property
    def is_operation(self) -> bool:
        return self.mode == OPERATION_MODE

    @property
    def is_finished(self) -> bool:
        return self.mode == FINISHED_MODE

    @property
    def mode_name(self) -> str:
        return PHASE_NAMES.get(self.mode, "UNKNOWN")

    def to_dict(self) -> dict:
        return {
            "frame_idx": int(self.frame_idx),
            "timestamp_ns": int(self.timestamp_ns),
            "mode": int(self.mode),
            "mode_name": self.mode_name,
            "phase": self.mode_name,
            "is_operation": self.is_operation,
            "is_finished": self.is_finished,
            "linear_speed_mps": float(self.linear_speed_mps),
            "angular_speed_rad_s": float(self.angular_speed_rad_s),
            "yaw_unwrapped_deg": float(self.yaw_unwrapped_deg),
            "operation_confidence": float(self.operation_confidence),
            "non_operation_confidence": float(self.non_operation_confidence),
            "camera_motion_score": float(self.camera_motion_score),
            "hand_presence_score": (
                None
                if self.hand_presence_score is None
                else float(self.hand_presence_score)
            ),
            "hand_motion_score": (
                None
                if self.hand_motion_score is None
                else float(self.hand_motion_score)
            ),
            "grasp_score": (
                None if self.grasp_score is None else float(self.grasp_score)
            ),
            "hand_evidence_available": bool(self.hand_evidence_available),
        }


@dataclass(frozen=True)
class CandidateSegment:
    """一个连续操作时间窗口。"""

    start_frame_idx: int
    end_frame_idx: int
    start_timestamp_ns: int
    end_timestamp_ns: int

    def to_dict(self) -> dict:
        return {
            "start_frame_idx": int(self.start_frame_idx),
            "end_frame_idx": int(self.end_frame_idx),
            "start_timestamp_ns": int(self.start_timestamp_ns),
            "end_timestamp_ns": int(self.end_timestamp_ns),
            "duration_s": (
                int(self.end_timestamp_ns) - int(self.start_timestamp_ns)
            ) / 1_000_000_000.0,
        }


@dataclass(frozen=True)
class PhaseSequence:
    """一个数据单元的完整阶段切割结果。"""

    frames: tuple[PhaseFrame, ...]
    candidate_segments: tuple[CandidateSegment, ...]
    summary: dict

    def to_dict(self) -> dict:
        return {
            "schema_version": PHASE_SCHEMA_VERSION,
            "world_frame": ARIA_MPS_WORLD_FRAME,
            "world_origin": ARIA_MPS_WORLD_ORIGIN,
            "initial_heading": ARIA_MPS_INITIAL_HEADING,
            "yaw_convention": ARIA_MPS_YAW_CONVENTION,
            "phase_names": PHASE_NAMES,
            "frames": [frame.to_dict() for frame in self.frames],
            "candidate_segments": [
                segment.to_dict() for segment in self.candidate_segments
            ],
            "summary": self.summary,
        }
