import cache_env
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
from train_shared_heads import assign_objects, object_loss
from shared_semantic_model import SharedSemantic, flatten_objects

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sw"))
from fall_lite import anchors_for_shape
from semantic_features import region_descriptor, OBJECT_CLASSES


class SupervisionContracts(unittest.TestCase):
    def test_partial_person_annotation_does_not_invent_other_background(self):
        item = dict(
            known_background=[False] * len(OBJECT_CLASSES),
            objects=[dict(label=0, box=[80, 30, 150, 160], crowd=False)],
        )
        target, offsets, positive = assign_objects(item, anchors_for_shape())
        self.assertTrue(positive.any())
        self.assertTrue((target[:, 1:] == -1).all())
        self.assertFalse((target == 0).any())
        self.assertTrue(np.isfinite(offsets).all())

    def test_crowd_is_ignored_without_box_regression(self):
        anchors = np.array([[50, 50, 20, 20], [150, 50, 20, 20]], np.float32)
        item = dict(
            known_background=[True] * len(OBJECT_CLASSES),
            objects=[dict(label=0, box=[0, 0, 100, 100], crowd=True)],
        )
        target, _, positive = assign_objects(item, anchors)
        self.assertEqual(target[0, 0], -1)
        self.assertEqual(target[1, 0], 0)
        self.assertFalse(positive.any())

    def test_unknown_labels_create_zero_loss_and_zero_gradient(self):
        n = len(anchors_for_shape())
        k = len(OBJECT_CLASSES)
        pred = torch.randn(1, n, k + 4, requires_grad=True)
        loss = object_loss(
            pred,
            torch.full((1, n, k), -1),
            torch.zeros(1, n, 4),
            torch.zeros(1, n, dtype=torch.bool),
        )
        self.assertEqual(float(loss), 0)
        loss.backward()
        self.assertEqual(float(pred.grad.abs().sum()), 0)

    def test_roi_uses_local_and_global_visual_features(self):
        features = [
            np.full((64, h, w), i + 1, np.float32)
            for i, (h, w) in enumerate(((24, 40), (12, 20), (6, 10)))
        ]
        vector = region_descriptor(features, [10, 10, 100, 100])
        self.assertEqual(vector.shape, (1920,))
        np.testing.assert_allclose(vector[:640], 1)
        with self.assertRaises(ValueError):
            region_descriptor(features, [400, 0, 450, 20])

    def test_training_semantic_head_cannot_update_accepted_fall_path(self):
        path = Path("E:/Zynq_project/ac880_fall/models/fall_lite_v6_varied/best.pt")
        if not path.exists():
            self.skipTest("Local accepted checkpoint required")
        torch.set_num_threads(2)
        torch.manual_seed(8)
        model = SharedSemantic(path)
        sample = torch.randn(1, 3, 192, 320)
        model.eval()
        with torch.no_grad():
            before = tuple(v.clone() for v in model(sample)[:3])
            original = model.base(sample)
        for a, b in zip(before, original):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        model.train()
        optimizer = torch.optim.SGD(model.objects.parameters(), lr=0.001)
        loss = flatten_objects(model.object_forward(model.pyramid(sample))).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            after = model(sample)[:3]
        for a, b in zip(before, after):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(all(v.grad is None for v in model.base.parameters()))


if __name__ == "__main__":
    unittest.main()
