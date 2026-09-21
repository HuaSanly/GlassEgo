import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from preprocess.data_types.VIOTypes import (
    ARIA_MPS_INITIAL_HEADING,
    ARIA_MPS_WORLD_FRAME,
    ARIA_MPS_WORLD_ORIGIN,
    OPENCV_CAMERA_FRAME,
)
from preprocess.hand_tracking.HandCacheLoader import load_cached_hands


def _write_hand_frame(
    root: Path,
    frame_idx: int,
    timestamp_ns: int,
    schema=2,
    grasp_score=None,
    tracking_state="observed",
):
    frame_dir = root / "preprocess" / "all_data" / f"{frame_idx:05d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": schema,
        "camera_frame": OPENCV_CAMERA_FRAME,
        "world_frame": ARIA_MPS_WORLD_FRAME,
        "world_origin": ARIA_MPS_WORLD_ORIGIN,
        "initial_heading": ARIA_MPS_INITIAL_HEADING,
        "idx": frame_idx,
        "ts": timestamp_ns,
        "hand_r": {
            "tracking_state": tracking_state,
            "confidence": 0.8,
            "grasp_state": 1,
            "midpoint_lin_vel_opt_world": [0.1, 0.0, 0.0],
            "midpoint_ang_vel_opt_world": [0.0, 0.2, 0.0],
        },
        "hand_l": None,
    }
    if grasp_score is not None:
        document["hand_r"]["grasp_score"] = grasp_score
    (frame_dir / "hamer_hands.json").write_text(
        json.dumps(document),
        encoding="utf-8",
    )


class HandCacheLoaderTests(unittest.TestCase):
    def test_loads_aligned_cached_hands(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_hand_frame(root, 0, 100)
            _write_hand_frame(root, 1, 200)

            hands = load_cached_hands(root, [100, 200])

            self.assertEqual(len(hands.hands), 2)
            self.assertEqual(hands.hands[0].idx, 0)
            self.assertAlmostEqual(hands.hands[0].hand_r.confidence, 0.8)
            np.testing.assert_allclose(
                hands.hands[1].hand_r.midpoint_lin_vel_opt_world,
                [0.1, 0.0, 0.0],
            )

    def test_old_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_hand_frame(root, 0, 100, schema=1)

            with self.assertRaisesRegex(ValueError, "Unsupported hand cache schema"):
                load_cached_hands(root, [100])

    def test_continuous_grasp_score_precedes_legacy_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_hand_frame(root, 0, 100, grasp_score=0.35)

            hands = load_cached_hands(root, [100])

            self.assertAlmostEqual(hands.hands[0].hand_r.grasp_score, 0.35)
            self.assertAlmostEqual(hands.hands[0].hand_r.grasp_state, 0.35)

    def test_missing_frame_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                load_cached_hands(Path(temp_dir), [100])

    def test_tracking_state_survives_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_hand_frame(root, 0, 100, tracking_state="observed")
            _write_hand_frame(root, 1, 200, tracking_state="interpolated")
            hands = load_cached_hands(root, [100, 200])

            hands.save_hands_json(filename="round_trip.json")
            restored = load_cached_hands(root, [100, 200], filename="round_trip.json")

            self.assertEqual(
                [frame.hand_r.tracking_state for frame in restored.hands],
                ["observed", "interpolated"],
            )

    def test_missing_tracking_state_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_hand_frame(root, 0, 100, tracking_state=None)
            with self.assertRaisesRegex(ValueError, "tracking_state"):
                load_cached_hands(root, [100])

    def test_misaligned_timestamp_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_hand_frame(root, 0, 100)
            with self.assertRaisesRegex(ValueError, "timestamp is not aligned"):
                load_cached_hands(root, [200])

    def test_misaligned_frame_index_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_hand_frame(root, 0, 100)
            path = root / "preprocess" / "all_data" / "00000" / "hamer_hands.json"
            document = json.loads(path.read_text())
            document["idx"] = 1
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "index is not aligned"):
                load_cached_hands(root, [100])

    def test_extra_cached_frame_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_hand_frame(root, 0, 100)
            _write_hand_frame(root, 1, 200)
            with self.assertRaisesRegex(ValueError, "unexpected frames"):
                load_cached_hands(root, [100])


if __name__ == "__main__":
    unittest.main()
