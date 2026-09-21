import unittest
from unittest.mock import patch

import torch

from preprocess.object_tracking import OrientAnything


class _FakeOrientAnythingModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(1))
        self.constructor_device = self.weight.device.type
        self.constructor_kwargs = kwargs
        self.loaded_state_dict = None
        self.assign = None
        self.is_evaluating = False
        self.target_device = None

    def load_state_dict(self, state_dict, *, assign=False):
        self.loaded_state_dict = state_dict
        self.assign = assign

    def eval(self):
        self.is_evaluating = True
        return self

    def to(self, device):
        self.target_device = device
        return self


class OrientAnythingLoadingTest(unittest.TestCase):
    def setUp(self):
        OrientAnything._VLM_MODEL_INSTANCE = None

    def tearDown(self):
        OrientAnything._VLM_MODEL_INSTANCE = None

    def test_loads_checkpoint_without_materializing_a_second_model(self):
        state_dict = {"weight": object()}
        with (
            patch.object(OrientAnything, "ORIENT_ANYTHING_AVAILABLE", True),
            patch(
                "orient_anything.vision_tower.VGGT_OriAny_Ref",
                _FakeOrientAnythingModel,
            ),
            patch(
                "huggingface_hub.hf_hub_download",
                return_value="/tmp/orient-anything.pt",
            ),
            patch.object(OrientAnything.torch.cuda, "is_available", return_value=True),
            patch.object(
                OrientAnything.torch.cuda,
                "get_device_capability",
                return_value=(8, 6),
            ),
            patch.object(
                OrientAnything.torch,
                "load",
                return_value=state_dict,
            ) as load_checkpoint,
        ):
            model = OrientAnything._get_vlm_model()
            cached_model = OrientAnything._get_vlm_model()

        self.assertIs(model, cached_model)
        self.assertEqual(model.constructor_device, "meta")
        self.assertEqual(
            model.constructor_kwargs,
            {"out_dim": 900, "dtype": torch.bfloat16, "nopretrain": True},
        )
        load_checkpoint.assert_called_once_with(
            "/tmp/orient-anything.pt",
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        self.assertIs(model.loaded_state_dict, state_dict)
        self.assertTrue(model.assign)
        self.assertTrue(model.is_evaluating)
        self.assertEqual(model.target_device, torch.device("cuda"))


if __name__ == "__main__":
    unittest.main()
