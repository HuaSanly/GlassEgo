import unittest

import numpy as np
from omegaconf import OmegaConf

from preprocess.data_types.PhaseTypes import (
    FINISHED_MODE,
    FINISHED_TAIL_FRAMES,
    FORCED_NON_OPERATION_PREFIX_FRAMES,
    MIN_CLASSIFIED_MIDDLE_FRAMES,
    MIN_UNIT_FRAME_COUNT,
    NON_OPERATION_MODE,
    OPERATION_MODE,
    PHASE_SCHEMA_VERSION,
    PhaseFrame,
    PhaseSequence,
)
from preprocess.data_types.VIOTypes import ARIA_MPS_WORLD_FRAME
from preprocess.phase_segmentation.PhaseSegmentationGenerator import (
    PhaseSegmentationGenerator,
)
from preprocess.phase_segmentation.PhaseSegmentationOps import PhaseSegmentationOps


class PhaseSegmentationOpsTests(unittest.TestCase):
    def test_phase_sequence_serializes_schema_four(self):
        sequence = PhaseSequence(
            frames=(
                PhaseFrame(
                    0,
                    100,
                    0,
                    0.01,
                    0.02,
                    0.0,
                    operation_confidence=0.8,
                    non_operation_confidence=0.2,
                ),
            ),
            candidate_segments=(),
            summary={},
        )

        document = sequence.to_dict()

        self.assertEqual(document["schema_version"], PHASE_SCHEMA_VERSION)
        self.assertEqual(PHASE_SCHEMA_VERSION, 4)
        self.assertEqual(document["world_frame"], ARIA_MPS_WORLD_FRAME)
        self.assertEqual(document["frames"][0]["phase"], "OPERATION")
        self.assertFalse(document["frames"][0]["is_finished"])
        self.assertAlmostEqual(document["frames"][0]["operation_confidence"], 0.8)

    def test_finished_frame_serialization(self):
        frame = PhaseFrame(179, 179, FINISHED_MODE, 0.0, 0.0, 0.0)

        document = frame.to_dict()

        self.assertEqual(document["phase"], "FINISHED")
        self.assertFalse(document["is_operation"])
        self.assertTrue(document["is_finished"])

    def test_exact_minimum_length_uses_fixed_three_phase_boundaries(self):
        scores = np.zeros(MIN_UNIT_FRAME_COUNT, dtype=np.float64)

        modes, operation_confidence, non_operation_confidence = (
            PhaseSegmentationGenerator._classify_phases(scores, self._phase_cfg())
        )

        middle_end = MIN_UNIT_FRAME_COUNT - FINISHED_TAIL_FRAMES
        np.testing.assert_array_equal(
            modes[:FORCED_NON_OPERATION_PREFIX_FRAMES],
            np.full(FORCED_NON_OPERATION_PREFIX_FRAMES, NON_OPERATION_MODE),
        )
        np.testing.assert_array_equal(
            modes[FORCED_NON_OPERATION_PREFIX_FRAMES:middle_end],
            np.full(MIN_CLASSIFIED_MIDDLE_FRAMES, OPERATION_MODE),
        )
        np.testing.assert_array_equal(
            modes[middle_end:],
            np.full(FINISHED_TAIL_FRAMES, FINISHED_MODE),
        )
        self.assertTrue(np.all(operation_confidence[:120] == 0.0))
        self.assertTrue(np.all(non_operation_confidence[:120] == 1.0))
        self.assertTrue(np.all(operation_confidence[-10:] == 0.0))
        self.assertTrue(np.all(non_operation_confidence[-10:] == 0.0))
        ratios = [
            np.mean(modes == OPERATION_MODE),
            np.mean(modes == NON_OPERATION_MODE),
            np.mean(modes == FINISHED_MODE),
        ]
        self.assertAlmostEqual(sum(ratios), 1.0)

    def test_scores_cannot_override_fixed_prefix_or_finished_tail(self):
        scores = np.concatenate(
            (
                np.zeros(FORCED_NON_OPERATION_PREFIX_FRAMES),
                np.ones(MIN_CLASSIFIED_MIDDLE_FRAMES),
                np.ones(FINISHED_TAIL_FRAMES),
            )
        )

        modes, _, _ = PhaseSegmentationGenerator._classify_phases(
            scores,
            self._phase_cfg(),
        )

        self.assertTrue(
            np.all(modes[:FORCED_NON_OPERATION_PREFIX_FRAMES] == NON_OPERATION_MODE)
        )
        self.assertTrue(np.all(modes[-FINISHED_TAIL_FRAMES:] == FINISHED_MODE))

    def test_candidate_segments_only_include_operation(self):
        frames = tuple(
            PhaseFrame(index, index, mode, 0.0, 0.0, 0.0)
            for index, mode in enumerate(
                [NON_OPERATION_MODE, OPERATION_MODE, OPERATION_MODE, FINISHED_MODE]
            )
        )

        segments = PhaseSegmentationGenerator._candidate_segments(frames)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].start_frame_idx, 1)
        self.assertEqual(segments[0].end_frame_idx, 2)

    @staticmethod
    def _phase_cfg():
        return OmegaConf.create(
            {
                "non_operation_enter_threshold": 0.6,
                "non_operation_exit_threshold": 0.35,
                "non_operation_enter_frames": 1,
                "non_operation_exit_frames": 1,
                "min_non_operation_frames": 1,
            }
        )

    def test_short_non_operation_run_is_suppressed(self):
        scores = np.asarray([0.1] * 5 + [0.9] * 3 + [0.1] * 5)

        modes = PhaseSegmentationOps.classify_binary_phases(
            scores,
            enter_threshold=0.6,
            exit_threshold=0.35,
            enter_frames=2,
            exit_frames=2,
            min_non_operation_frames=5,
        )

        np.testing.assert_array_equal(modes, np.zeros(len(scores), dtype=np.int32))

    def test_hysteresis_keeps_sustained_non_operation(self):
        scores = np.asarray([0.1] * 4 + [0.8] * 8 + [0.2] * 4)

        modes = PhaseSegmentationOps.classify_binary_phases(
            scores,
            enter_threshold=0.6,
            exit_threshold=0.35,
            enter_frames=3,
            exit_frames=2,
            min_non_operation_frames=5,
        )

        self.assertEqual(modes[4], 1)
        self.assertEqual(modes[11], 1)
        self.assertTrue(np.all(modes[:3] == 0))
        self.assertTrue(np.all(modes[-2:] == 0))

    def test_hand_operation_evidence_reduces_non_operation_score(self):
        camera_score = np.asarray([0.65, 0.65])
        presence = np.asarray([0.0, 0.9])
        motion = np.asarray([0.0, 0.8])
        grasp = np.asarray([0.0, 0.9])
        available = np.asarray([True, True])

        scores = PhaseSegmentationOps.combine_non_operation_score(
            camera_score,
            presence,
            motion,
            grasp,
            available,
            0.3,
            0.45,
            0.25,
            0.3,
            0.35,
        )

        self.assertGreater(scores[0], scores[1])
        self.assertLess(scores[1], 0.65)

    def test_invalid_hysteresis_thresholds_raise(self):
        with self.assertRaises(ValueError):
            PhaseSegmentationOps.classify_binary_phases(
                np.asarray([0.1, 0.2]),
                enter_threshold=0.3,
                exit_threshold=0.4,
                enter_frames=1,
                exit_frames=1,
                min_non_operation_frames=1,
            )

    def test_negative_fusion_weight_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no_hand_weight"):
            PhaseSegmentationOps.combine_non_operation_score(
                np.asarray([0.5]),
                np.asarray([0.5]),
                np.asarray([0.5]),
                np.asarray([0.5]),
                np.asarray([True]),
                0.3,
                0.45,
                0.25,
                0.3,
                -0.1,
            )


if __name__ == "__main__":
    unittest.main()
