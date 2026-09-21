import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from preprocess.data_types.VIOTypes import ARIA_MPS_WORLD_FRAME
from preprocess.object_tracking.ObjectTriangulator import ObjectTriangulator
from preprocess.object_tracking.OrientAnything import (
    angles_to_rot_matrix,
    estimate_frame_vlm,
    get_crop_from_2d_kpts,
)


class OrientAnythingTests(unittest.TestCase):
    def test_keypoint_crop_converts_bgr_to_rgb(self):
        image_bgr = np.zeros((8, 10, 3), dtype=np.uint8)
        image_bgr[2:6, 3:8] = [10, 20, 30]

        crop = get_crop_from_2d_kpts(
            image_bgr,
            np.array([[3.0, 2.0], [7.0, 5.0]]),
            pad=0,
        )

        self.assertEqual(crop.mode, "RGB")
        np.testing.assert_array_equal(np.asarray(crop)[0, 0], [30, 20, 10])

    def test_vlm_prediction_keeps_triangulated_translation(self):
        answer = {
            "ref_az_pred": 90.0,
            "ref_el_pred": 0.0,
            "ref_ro_pred": 0.0,
            "ref_alpha_pred": 2,
        }
        translation = np.array([0.2, -0.1, 1.5])

        with patch(
            "orient_anything.utils.app_utils.inf_single_case",
            return_value=answer,
        ):
            transform, info = estimate_frame_vlm(
                Image.new("RGB", (16, 16)),
                translation,
                do_rm_bkg=False,
                model=object(),
            )

        np.testing.assert_allclose(transform[:3, :3], angles_to_rot_matrix(90, 0, 0))
        np.testing.assert_allclose(transform[:3, 3], translation)
        self.assertEqual(info["vlm_symmetry_alpha"], 2)


class ObjectTriangulatorPoseMethodTests(unittest.TestCase):
    def _triangulator(self, pose_method):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        cfg = OmegaConf.create(
            {
                "pose_method": pose_method,
                "vlm_remove_background": True,
            }
        )
        return ObjectTriangulator(
            Path(temp_dir.name),
            cfg,
            vlm_model=object(),
        )

    def test_resolves_per_object_pose_method_with_default(self):
        triangulator = self._triangulator(
            {"obj_can": "vlm", "default": "pca2"}
        )

        self.assertEqual(triangulator._pose_method_for_object("obj_can"), "vlm")
        self.assertEqual(triangulator._pose_method_for_object("obj_plate"), "pca2")

    def test_vlm_pose_dispatches_to_orient_anything(self):
        triangulator = self._triangulator("vlm")
        expected = np.eye(4)
        expected[:3, 3] = [0.1, 0.2, 1.0]
        image = Image.new("RGB", (12, 12))

        with patch(
            "preprocess.object_tracking.ObjectTriangulator.estimate_frame_vlm",
            return_value=(expected, {"method": "vlm_anchor"}),
        ) as estimator:
            transform, info = triangulator._estimate_pose(
                np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0]]),
                is_anchor=True,
                anchor_center_cam=None,
                method="vlm",
                image=image,
            )

        np.testing.assert_array_equal(transform, expected)
        self.assertEqual(info["method"], "vlm_anchor")
        estimator.assert_called_once()
        call = estimator.call_args.kwargs
        self.assertIs(call["image"], image)
        np.testing.assert_allclose(
            call["t_cam"],
            [1.0 / 30.0, 1.0 / 30.0, 1.0],
        )
        self.assertTrue(call["is_anchor"])
        self.assertIsNone(call["anchor_center_cam"])
        self.assertTrue(call["do_rm_bkg"])
        self.assertIs(call["model"], triangulator.vlm_model)

    def test_triangulation_writes_vlm_pose_in_aria_world(self):
        cfg = OmegaConf.create(
            {
                "pose_method": {"obj_can": "vlm", "default": "pca2"},
                "vlm_crop_padding_px": 1,
                "vlm_remove_background": False,
                "step": 1,
                "smooth_window": 3,
                "smooth_polyorder": 1,
                "ba_f_scale": 3.0,
                "axes_len_m": 0.12,
                "point_radius_m": 0.005,
                "line_radius_m": 0.001,
            }
        )
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        triangulator = ObjectTriangulator(
            temp_dir.name,
            cfg,
            vlm_model=object(),
        )
        triangulated_points = [
            np.array([0.0, 0.0, 1.0]),
            np.array([0.1, 0.0, 1.0]),
            np.array([0.0, 0.1, 1.0]),
        ]
        triangulator.engine.triangulate_dlt = unittest.mock.Mock(
            side_effect=[(point, 1.0) for point in triangulated_points]
        )
        triangulator.engine.ba_refine = unittest.mock.Mock(
            side_effect=lambda point, *_: point
        )
        tracks_document = {
            "frames": [0, 1],
            "objects": {
                "obj_can": {
                    "tracks": [
                        [[5.0, 5.0], [8.0, 5.0], [5.0, 8.0]],
                        [[5.5, 5.0], [8.5, 5.0], [5.5, 8.0]],
                    ],
                    "visibility": [[1, 1, 1], [1, 1, 1]],
                }
            },
        }
        frames_bgr = [
            np.zeros((16, 16, 3), dtype=np.uint8),
            np.zeros((16, 16, 3), dtype=np.uint8),
        ]
        trajectory = SimpleNamespace(
            world_frame=ARIA_MPS_WORLD_FRAME,
            frames=(
                SimpleNamespace(frame_idx=0, c2w=np.eye(4)),
                SimpleNamespace(frame_idx=1, c2w=np.eye(4)),
            ),
        )
        vio_result = SimpleNamespace(
            calibration=SimpleNamespace(
                intrinsics=np.array([100.0, 100.0, 8.0, 8.0])
            ),
            trajectory=trajectory,
        )

        def fake_vlm(**kwargs):
            transform = np.eye(4)
            transform[:3, 3] = kwargs["t_cam"]
            return transform, {"method": "vlm_anchor (alpha:1)"}

        with (
            patch(
                "preprocess.object_tracking.ObjectTriangulator.estimate_frame_vlm",
                side_effect=fake_vlm,
            ),
            patch.object(
                triangulator,
                "_draw_qa",
                return_value=frames_bgr[-1],
            ),
        ):
            document, _ = triangulator.triangulate(
                tracks_document,
                frames_bgr,
                vio_result=vio_result,
            )

        self.assertEqual(document["schema_version"], 3)
        self.assertEqual(document["world_frame"], ARIA_MPS_WORLD_FRAME)
        self.assertEqual(
            document["pose_method"],
            {"obj_can": "vlm", "default": "pca2"},
        )
        result = document["objects"]["obj_can"]
        self.assertEqual(result["pose_info"]["pose_method"], "vlm")
        np.testing.assert_allclose(
            np.asarray(result["object_to_world_matrix"])[:3, 3],
            np.mean(triangulated_points, axis=0),
        )

    def test_object_tracking_result_schema_carries_world_metadata(self):
        from preprocess.data_types.ObjectTypes import ObjectTrackingResult

        result = ObjectTrackingResult(
            Path("/tmp/unit"),
            Path("/tmp/unit/video.mp4"),
            Path("/tmp/unit/preprocess/objects"),
            (),
            {},
        ).to_dict()

        self.assertEqual(result["schema_version"], 2)
        self.assertEqual(result["world_frame"], "aria_mps_x_right_y_up_z_backward")


if __name__ == "__main__":
    unittest.main()
