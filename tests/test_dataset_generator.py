import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from preprocess.DatasetGenerator import DatasetGenerator
from utils.utils_artifact_store import FrameArtifactStore


class DatasetGeneratorTests(unittest.TestCase):
    def test_only_finished_frames_receive_finished_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            unit_dir = Path(temporary_dir)
            store = FrameArtifactStore(unit_dir)
            image = np.zeros((2, 2, 3), dtype=np.uint8)
            detections = []
            for frame_idx in (0, 1):
                for filename in (
                    "rgb_WoArm.png",
                    "rgb_WArmObjKpts.png",
                    "rgb_WoArm_WArmObjKpts.png",
                    "mask_arm.png",
                    "mask_arm_and_obj.png",
                ):
                    store.write_image(frame_idx, filename, image, True)
                frame_dir = store.frame_dir(frame_idx, True)
                detections.append(
                    SimpleNamespace(
                        frame_idx=frame_idx,
                        combined_mask_path=frame_dir / "mask_arm_and_obj.png",
                        objects=(),
                    )
                )

            identity = np.eye(4, dtype=np.float64)
            pose_document = {
                "anchor_key": "obj1",
                "cam0_c2w": identity.tolist(),
                "anchor_to_world": identity.tolist(),
                "world_to_anchor": identity.tolist(),
                "frames": [
                    {
                        "frame_idx": frame_idx,
                        "timestamp_ns": frame_idx,
                        "hands": {},
                        "objects": {"obj1": {"T_obj_to_world": identity.tolist()}},
                    }
                    for frame_idx in (0, 1)
                ],
            }
            triangulation_document = {
                "objects": {
                    "obj1": {
                        "points_3d_world": [[0.0, 0.0, 1.0]],
                        "object_to_world_matrix": identity.tolist(),
                    }
                }
            }
            vio_result = SimpleNamespace(
                calibration=SimpleNamespace(
                    intrinsics=np.asarray([1.0, 1.0, 0.0, 0.0]),
                    resolution=(2, 2),
                ),
                trajectory=SimpleNamespace(
                    frames=tuple(
                        SimpleNamespace(frame_idx=frame_idx, c2w=identity)
                        for frame_idx in (0, 1)
                    )
                ),
            )

            DatasetGenerator(unit_dir, store=store).run(
                pose_document,
                triangulation_document,
                detections,
                {0: image, 1: image},
                vio_result,
                30.0,
                {0, 1},
                {1},
            )

            metadata = []
            for frame_idx in (0, 1):
                path = store.frame_dir(frame_idx, True) / "training_data.json"
                with path.open("r", encoding="utf-8") as stream:
                    metadata.append(json.load(stream)["metadata"])
            self.assertEqual(metadata[0]["is_finished"], 0.0)
            self.assertEqual(metadata[1]["is_finished"], 1.0)


if __name__ == "__main__":
    unittest.main()
