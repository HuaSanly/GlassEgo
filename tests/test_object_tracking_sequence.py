import unittest

from omegaconf import OmegaConf

from preprocess.data_types.PhaseTypes import (
    FINISHED_MODE,
    NON_OPERATION_MODE,
    OPERATION_MODE,
    PhaseFrame,
    PhaseSequence,
)
from preprocess.object_tracking.ObjectTrackingGenerator import (
    ObjectTrackingGenerator,
)


class ObjectTrackingSequenceTests(unittest.TestCase):
    def test_contiguous_runs_split_phase_gaps(self):
        runs = ObjectTrackingGenerator._contiguous_runs([0, 1, 2, 7, 8])

        self.assertEqual(runs, [[0, 1, 2], [7, 8]])

    def test_tracking_uses_all_operation_frames_in_order(self):
        generator = ObjectTrackingGenerator.__new__(ObjectTrackingGenerator)
        generator.phase_result = PhaseSequence(
            frames=tuple(
                PhaseFrame(index, index, mode, 0.0, 0.0, 0.0)
                for index, mode in enumerate([0, 0, 1, 0, 0, 0, 0])
            ),
            candidate_segments=(),
            summary={},
        )

        self.assertEqual(
            generator._raw_manipulation_frames(),
            [0, 1, 3, 4, 5, 6],
        )

    def test_object_centric_uses_frames_before_first_operation(self):
        generator = self._generator_with_modes([1, 1, 1, 1, 1, 0, 0])

        self.assertEqual(generator._build_object_centric_indices(), [1, 2, 3, 4])

    def test_object_centric_supplements_short_preamble_from_operation(self):
        generator = self._generator_with_modes([1, 1, 0, 0, 0, 0])

        self.assertEqual(generator._build_object_centric_indices(), [0, 1, 2, 3])

    def test_merge_preserves_order_and_removes_overlapping_frames(self):
        object_centric = [0, 1, 2, 3, 4]
        raw_manipulation = [3, 4, 5, 6, 5]

        tracking_frames = ObjectTrackingGenerator._merge_tracking_frames(
            object_centric,
            raw_manipulation,
        )

        self.assertEqual(tracking_frames, [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(object_centric, [0, 1, 2, 3, 4])
        self.assertEqual(raw_manipulation, [3, 4, 5, 6, 5])

    def test_training_frames_include_explicit_finished_phase_only(self):
        generator = self._generator_with_modes(
            [
                NON_OPERATION_MODE,
                OPERATION_MODE,
                NON_OPERATION_MODE,
                FINISHED_MODE,
                FINISHED_MODE,
            ]
        )

        training_frames, finished_frames = generator._training_frame_sets()

        self.assertEqual(training_frames, {1, 3, 4})
        self.assertEqual(finished_frames, {3, 4})

    @staticmethod
    def _generator_with_modes(modes):
        generator = ObjectTrackingGenerator.__new__(ObjectTrackingGenerator)
        generator.cfg = OmegaConf.create(
            {
                "indices": {
                    "object_centric_max_frames": 4,
                    "object_centric_min_frames": 4,
                }
            }
        )
        generator.phase_result = PhaseSequence(
            frames=tuple(
                PhaseFrame(index, index, mode, 0.0, 0.0, 0.0)
                for index, mode in enumerate(modes)
            ),
            candidate_segments=(),
            summary={},
        )
        return generator


if __name__ == "__main__":
    unittest.main()
