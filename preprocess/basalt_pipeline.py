import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation, Slerp

PREPROCESS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PREPROCESS_ROOT.parent
DEFAULT_CONFIG_ROOT = PREPROCESS_ROOT / "config"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PREPROCESS_ROOT) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_ROOT))

from vio.BasaltVIOGenerator import BasaltVIOGenerator
from utils.utils_math import time_it


class BasaltPipeline:
    """Basalt VIO 独立诊断入口。"""

    def __init__(self, config_root: str | Path = DEFAULT_CONFIG_ROOT):
        config_path = Path(config_root) / "vio.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing preprocess config: {config_path}")
        self.cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(self.cfg)

    @time_it
    def run(
        self,
        unit_dir: str | Path,
        ground_truth_path: str | Path | None = None,
        force: bool = False,
    ) -> dict:
        generator = BasaltVIOGenerator(unit_dir=unit_dir, cfg=self.cfg)
        result = generator.get_camera_poses(force=force)
        report = dict(result.report)
        report["metrics"] = self._evaluate_trajectory(
            result.trajectory_path,
            ground_truth_path,
        )
        return report

    @classmethod
    def _evaluate_trajectory(
        cls,
        trajectory_path: Path,
        ground_truth_path: str | Path | None,
    ) -> dict | None:
        if ground_truth_path is None:
            return None
        pose_timestamps_ns, T_world_imu = cls._load_trajectory(trajectory_path)
        gt_timestamps_ns, gt_T_world_imu = cls._load_trajectory(
            Path(ground_truth_path).expanduser().resolve()
        )

        estimate = []
        ground_truth = []
        for timestamp_ns, transform in zip(pose_timestamps_ns, T_world_imu):
            gt_transform = cls._interpolate_pose(
                int(timestamp_ns),
                gt_timestamps_ns,
                gt_T_world_imu,
            )
            if gt_transform is not None:
                estimate.append(transform)
                ground_truth.append(gt_transform)
        if len(estimate) < 3:
            raise ValueError(
                "Ground truth has fewer than three timestamp associations"
            )

        estimate = np.stack(estimate)
        ground_truth = np.stack(ground_truth)
        alignment = cls._align_se3(
            estimate[:, :3, 3],
            ground_truth[:, :3, 3],
        )
        aligned_estimate = np.einsum("ij,njk->nik", alignment, estimate)
        translation_error = np.linalg.norm(
            aligned_estimate[:, :3, 3] - ground_truth[:, :3, 3],
            axis=1,
        )

        rpe_translation = []
        rpe_rotation_deg = []
        for index in range(1, len(aligned_estimate)):
            estimate_delta = (
                np.linalg.inv(aligned_estimate[index - 1])
                @ aligned_estimate[index]
            )
            ground_truth_delta = (
                np.linalg.inv(ground_truth[index - 1]) @ ground_truth[index]
            )
            delta_error = np.linalg.inv(ground_truth_delta) @ estimate_delta
            rpe_translation.append(np.linalg.norm(delta_error[:3, 3]))
            rpe_rotation_deg.append(
                np.degrees(
                    Rotation.from_matrix(delta_error[:3, :3]).magnitude()
                )
            )

        return {
            "associations": len(estimate),
            "ate_rmse_m": float(np.sqrt(np.mean(translation_error**2))),
            "rpe_translation_rmse_m": float(
                np.sqrt(np.mean(np.square(rpe_translation)))
            ),
            "rpe_rotation_rmse_deg": float(
                np.sqrt(np.mean(np.square(rpe_rotation_deg)))
            ),
        }

    @staticmethod
    def _load_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray]:
        if not path.is_file():
            raise FileNotFoundError(f"Trajectory not found: {path}")
        timestamps = []
        transforms = []
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            for row_index, row in enumerate(reader):
                if not row or row[0].lstrip().startswith("#"):
                    continue
                if len(row) < 8:
                    raise ValueError(f"Invalid trajectory row {row_index}: {row}")
                values = np.asarray(row[1:8], dtype=np.float64)
                quaternion_wxyz = values[3:]
                norm = np.linalg.norm(quaternion_wxyz)
                if not np.all(np.isfinite(values)) or norm < 1e-12:
                    raise ValueError(f"Invalid trajectory row {row_index}: {row}")
                quaternion_wxyz /= norm
                transform = np.eye(4, dtype=np.float64)
                transform[:3, :3] = Rotation.from_quat(
                    [
                        quaternion_wxyz[1],
                        quaternion_wxyz[2],
                        quaternion_wxyz[3],
                        quaternion_wxyz[0],
                    ]
                ).as_matrix()
                transform[:3, 3] = values[:3]
                timestamps.append(int(row[0]))
                transforms.append(transform)
        if not timestamps:
            raise ValueError(f"Trajectory contains no poses: {path}")
        timestamps = np.asarray(timestamps, dtype=np.int64)
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("Trajectory timestamps must be strictly increasing")
        return timestamps, np.stack(transforms)

    @staticmethod
    def _interpolate_pose(
        timestamp_ns: int,
        timestamps_ns: np.ndarray,
        transforms: np.ndarray,
    ) -> np.ndarray | None:
        right = int(np.searchsorted(timestamps_ns, timestamp_ns, side="left"))
        if right < len(timestamps_ns) and timestamps_ns[right] == timestamp_ns:
            return transforms[right].copy()
        if right == 0 or right >= len(timestamps_ns):
            return None

        left = right - 1
        interval_ns = int(timestamps_ns[right] - timestamps_ns[left])
        if interval_ns <= 0 or interval_ns > 100_000_000:
            return None
        ratio = (timestamp_ns - int(timestamps_ns[left])) / interval_ns
        result = np.eye(4, dtype=np.float64)
        result[:3, 3] = (
            (1.0 - ratio) * transforms[left, :3, 3]
            + ratio * transforms[right, :3, 3]
        )
        rotations = Rotation.from_matrix(
            np.stack([transforms[left, :3, :3], transforms[right, :3, :3]])
        )
        result[:3, :3] = Slerp([0.0, 1.0], rotations)([ratio]).as_matrix()[0]
        return result

    @staticmethod
    def _align_se3(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        source_mean = source.mean(axis=0)
        target_mean = target.mean(axis=0)
        covariance = (source - source_mean).T @ (target - target_mean)
        u, _, vt = np.linalg.svd(covariance)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = target_mean - rotation @ source_mean
        return transform


def _parse_args():
    parser = argparse.ArgumentParser(description="Validate monocular Basalt VIO")
    parser.add_argument("--unit", required=True, help="Path to one data unit")
    parser.add_argument(
        "--ground_truth",
        default=None,
        help="Optional EuroC-format IMU ground truth",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore a valid cached VIO result",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = BasaltPipeline().run(
        unit_dir=args.unit,
        ground_truth_path=args.ground_truth,
        force=args.force,
    )
    print(json.dumps(result, indent=2))
