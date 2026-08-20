from dataclasses import dataclass


PHASE_NAMES = {
    0: "STOP",
    1: "FORWARD",
    2: "ROTATE",
    3: "TRANSITION",
    4: "FINISHED",
}


@dataclass(frozen=True)
class PhaseFrame:
    """单帧相机运动阶段及其运动学指标。"""

    frame_idx: int
    timestamp_ns: int
    mode: int
    stop: bool
    linear_speed_mps: float
    angular_speed_rad_s: float
    yaw_unwrapped_deg: float

    @property
    def mode_name(self) -> str:
        return PHASE_NAMES.get(self.mode, "UNKNOWN")

    def to_dict(self) -> dict:
        return {
            "frame_idx": int(self.frame_idx),
            "timestamp_ns": int(self.timestamp_ns),
            "mode": int(self.mode),
            "mode_name": self.mode_name,
            "stop": bool(self.stop),
            "linear_speed_mps": float(self.linear_speed_mps),
            "angular_speed_rad_s": float(self.angular_speed_rad_s),
            "yaw_unwrapped_deg": float(self.yaw_unwrapped_deg),
        }


@dataclass(frozen=True)
class CandidateSegment:
    """一个候选静止/操作时间窗口。"""

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
            "schema_version": 1,
            "phase_names": PHASE_NAMES,
            "frames": [frame.to_dict() for frame in self.frames],
            "candidate_segments": [
                segment.to_dict() for segment in self.candidate_segments
            ],
            "summary": self.summary,
        }
