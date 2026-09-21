import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from preprocess.data_types.HandsTypes import HandData, Hands, HandsData
from preprocess.data_types.ObjectTypes import ObjectFrameData, ObjectMaskData
from preprocess.data_types.VIOTypes import (
    ARIA_MPS_INITIAL_HEADING,
    ARIA_MPS_WORLD_FRAME,
    ARIA_MPS_WORLD_ORIGIN,
)
from preprocess.object_tracking.ObjectPosePropagator import ObjectPosePropagator
from preprocess.object_tracking.ObjectPoseQA import ObjectPoseQAExporter
from preprocess.object_tracking.ObjectTrackingGenerator import ObjectTrackingGenerator


class ObjectPosePropagatorTests(unittest.TestCase):
    def test_static_anchor_and_hand_latched_object_follow_humanego(self):
        poses = [
            self._pose([0.10, 0.0, 0.0]),
            self._pose([0.10, 0.0, 0.0]),
            self._pose([0.25, 0.0, 0.0]),
            self._pose([0.30, 0.0, 0.0]),
            self._pose([0.30, 0.0, 0.0]),
        ]
        grasps = [0, 1, 1, 0, 0]
        hands = Hands(
            tss=[100, 200, 300, 400],
            hands=[
                HandsData(
                    idx=index,
                    ts=(index + 1) * 100,
                    hand_r=HandData(
                        confidence=1.0,
                        grasp_state=grasp,
                        midpoint_pose_opt_world=pose,
                    ),
                )
                for index, (pose, grasp) in enumerate(zip(poses, grasps))
            ],
        )
        document = ObjectPosePropagator().propagate(
            self._triangulation(),
            [0, 1, 2, 3, 4],
            self._vio_result(5),
            hands,
        )

        anchor_poses = [
            np.asarray(frame["objects"]["obj1"]["T_obj_to_world"])
            for frame in document["frames"]
        ]
        for pose in anchor_poses:
            np.testing.assert_allclose(pose, np.eye(4))

        frames = document["frames"]
        self.assertFalse(frames[0]["objects"]["obj2"]["is_dynamic"])
        self.assertTrue(frames[1]["objects"]["obj2"]["is_dynamic"])
        self.assertTrue(frames[2]["objects"]["obj2"]["is_dynamic"])
        self.assertTrue(frames[3]["objects"]["obj2"]["is_dynamic"])
        self.assertFalse(frames[4]["objects"]["obj2"]["is_dynamic"])
        self.assertEqual(frames[4]["objects"]["obj2"]["pose_source"], "hand_last_pose")
        moved_pose = np.asarray(frames[2]["objects"]["obj2"]["T_obj_to_world"])
        held_pose = np.asarray(frames[3]["objects"]["obj2"]["T_obj_to_world"])
        released_pose = np.asarray(frames[4]["objects"]["obj2"]["T_obj_to_world"])
        np.testing.assert_allclose(moved_pose[:3, 3], [0.25, 0.0, 0.0])
        np.testing.assert_allclose(released_pose, held_pose)
        self.assertEqual(document["anchor_key"], "obj1")
        self.assertEqual(document["schema_version"], 2)
        self.assertEqual(document["dynamic_frame_count"], 3)

    def test_initial_grasp_can_latch_without_release_transition(self):
        hands = Hands(
            tss=[100, 200],
            hands=[
                HandsData(
                    idx=index,
                    ts=(index + 1) * 100,
                    hand_r=HandData(
                        confidence=1.0,
                        grasp_state=1,
                        midpoint_pose_opt_world=self._pose([0.10, 0.0, 0.0]),
                    ),
                )
                for index in range(2)
            ],
        )

        document = ObjectPosePropagator().propagate(
            self._triangulation(),
            [0, 1],
            self._vio_result(2),
            hands,
        )

        self.assertEqual(document["dynamic_frame_count"], 2)
        self.assertTrue(document["frames"][0]["objects"]["obj2"]["is_dynamic"])

    def test_large_object_uses_surface_proximity_not_center_distance(self):
        triangulation = self._triangulation()
        triangulation["objects"]["obj2"]["points_3d_world"] = [
            [0.1, -0.25, 0.0],
            [0.1, 0.25, 0.0],
        ]
        hands = Hands(
            tss=[100],
            hands=[
                HandsData(
                    idx=0,
                    ts=100,
                    hand_r=HandData(
                        confidence=1.0,
                        grasp_state=0.8,
                        midpoint_pose_opt_world=self._pose([0.34, 0.0, 0.0]),
                    ),
                )
            ],
        )
        document = ObjectPosePropagator().propagate(
            triangulation, [0], self._vio_result(1), hands
        )
        self.assertTrue(document["frames"][0]["objects"]["obj2"]["is_dynamic"])
        self.assertGreater(
            document["frames"][0]["hands"]["right"]["proximity_score"], 0.9
        )

    def test_object_keypoints_follow_dynamic_pose(self):
        triangulation = self._triangulation()
        local = ObjectTrackingGenerator._object_local_points(triangulation)
        frame_objects = {
            "obj1": {"T_obj_to_world": np.eye(4).tolist()},
            "obj2": {"T_obj_to_world": self._pose([0.30, 0.0, 0.0]).tolist()},
        }

        points = ObjectTrackingGenerator._transform_object_points(
            local,
            frame_objects,
        )

        np.testing.assert_allclose(points["obj2"]["world"][0], [0.30, 0.0, 0.0])

    def test_training_data_is_written_with_humanego_and_aria_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            unit_dir = Path(tmp)
            frame_dir = unit_dir / "preprocess" / "objects" / "all_data" / "00000"
            frame_dir.mkdir(parents=True)
            mask_path = frame_dir / "mask_obj1.png"
            combined_mask_path = frame_dir / "mask_arm_and_obj.png"
            mask_path.touch()
            combined_mask_path.touch()
            generator = ObjectTrackingGenerator.__new__(ObjectTrackingGenerator)
            generator.unit_dir = unit_dir
            generator.unit = SimpleNamespace(video_path=unit_dir / "video.mp4")
            generator.training_data_dir = unit_dir / "preprocess" / "all_data"
            generator.vio_result = SimpleNamespace(
                trajectory=SimpleNamespace(
                    frames=(
                        SimpleNamespace(
                            frame_idx=0,
                            timestamp_ns=100,
                            c2w=np.eye(4),
                        ),
                    ),
                ),
                calibration=SimpleNamespace(
                    intrinsics=np.asarray([100.0, 100.0, 4.0, 3.0]),
                    resolution=(8, 6),
                ),
            )
            frame_data = [
                ObjectFrameData(
                    frame_idx=0,
                    timestamp_ns=100,
                    objects=(
                        ObjectMaskData(
                            key="obj1",
                            prompt="anchor",
                            confidence=1.0,
                            boxes=np.zeros((1, 4)),
                            confidences=np.ones(1),
                            mask_path=mask_path,
                        ),
                    ),
                    combined_mask_path=combined_mask_path,
                    vis_path=None,
                )
            ]
            pose_document = {
                "anchor_key": "obj1",
                "cam0_c2w": np.eye(4).tolist(),
                "anchor_to_world": np.eye(4).tolist(),
                "world_to_anchor": np.eye(4).tolist(),
                "frame_count": 1,
                "frames": [
                    {
                        "frame_idx": 0,
                        "timestamp_ns": 100,
                        "hands": {
                            "left": {
                                "T_hand_to_world": None,
                                "grasp": 0.0,
                            }
                        },
                        "objects": {
                            "obj1": {
                                "T_obj_to_world": np.eye(4).tolist(),
                                "is_dynamic": False,
                            },
                            "obj2": {
                                "T_obj_to_world": self._pose(
                                    [0.10, 0.0, 0.0]
                                ).tolist(),
                                "is_dynamic": False,
                            },
                        },
                    }
                ],
            }

            generator._write_training_data(
                pose_document,
                self._triangulation(),
                frame_data,
                [np.zeros((6, 8, 3), dtype=np.uint8)],
                30.0,
            )

            output_path = (
                unit_dir
                / "preprocess"
                / "all_data"
                / "00000"
                / "training_data.json"
            )
            document = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(document["world_frame"], ARIA_MPS_WORLD_FRAME)
            self.assertEqual(document["metadata"]["anchor_key"], "obj1")
            self.assertIn("virtual_static_anchor", document["metadata"]["world_transforms"])
            self.assertEqual(document["entities"]["hands"], {})
            self.assertTrue(Path(document["obs"]["rgb_path"]).is_file())

    def test_object_centric_qa_exports_ply_and_png(self):
        pose_document = ObjectPosePropagator().propagate(
            self._triangulation(),
            [0],
            self._vio_result(1),
        )
        with tempfile.TemporaryDirectory() as tmp:
            report = ObjectPoseQAExporter(tmp).export(
                self._triangulation(),
                pose_document,
            )

            self.assertGreater(Path(report["ply"]).stat().st_size, 0)
            self.assertGreater(Path(report["png"]).stat().st_size, 0)

    @staticmethod
    def _pose(translation):
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = translation
        return pose

    @classmethod
    def _triangulation(cls):
        return {
            "schema_version": 3,
            "world_frame": ARIA_MPS_WORLD_FRAME,
            "world_origin": ARIA_MPS_WORLD_ORIGIN,
            "initial_heading": ARIA_MPS_INITIAL_HEADING,
            "cam0_c2w": np.eye(4).tolist(),
            "objects": {
                "obj1": {
                    "object_to_world_matrix": np.eye(4).tolist(),
                    "points_3d_world": [[0.0, 0.0, 0.0]],
                },
                "obj2": {
                    "object_to_world_matrix": cls._pose([0.10, 0.0, 0.0]).tolist(),
                    "points_3d_world": [[0.10, 0.0, 0.0]],
                },
            },
        }

    @staticmethod
    def _vio_result(frame_count):
        frames = tuple(
            SimpleNamespace(
                frame_idx=index,
                timestamp_ns=(index + 1) * 100,
                c2w=np.eye(4),
            )
            for index in range(frame_count)
        )
        return SimpleNamespace(
            trajectory=SimpleNamespace(
                world_frame=ARIA_MPS_WORLD_FRAME,
                frames=frames,
            )
        )


if __name__ == "__main__":
    unittest.main()
