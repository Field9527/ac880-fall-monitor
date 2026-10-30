import cache_env
import unittest
import numpy as np
import torch
from train_nano_fall import masked_loss, sampling_weights, selection_key, target_for


class NanoSupervisionContracts(unittest.TestCase):
    def test_unknown_coco_pose_has_no_fall_loss_or_gradient(self):
        row = dict(regions=[], fall_absent=None, fallback_region_class=None)
        cls, offset = target_for(row, np.array([[10, 10, 5, 5], [20, 20, 5, 5]], np.float32))
        pred = torch.zeros(1, 100, 5, requires_grad=True)
        loss = masked_loss(pred, torch.full((1, 100), -1), torch.zeros(1, 100, 4))
        self.assertTrue((cls == -1).all())
        self.assertEqual(float(loss), 0)
        loss.backward()
        self.assertEqual(float(pred.grad.abs().sum()), 0)

    def test_normal_people_do_not_share_clothing_sampling_budget(self):
        rows = [dict(source="original_falldataset")] + [
            dict(source="Fashion-MNIST") for _ in range(2000)
        ]
        targets = {i: (np.zeros(1, np.int8), None) for i in range(len(rows))}
        ids, w, counts = sampling_weights(rows, targets)
        self.assertAlmostEqual(float(w[0] / w[1:].sum()), 3.0)

    def test_one_positive_domain_cannot_hide_regression_in_another(self):
        def m(a, b):
            return dict(reference_tp=a, nano_tp=b, gt=a, negative_images=0, nano_fp=0)

        worse = dict(original=m(571, 639), cauca=m(14, 10))
        retained = dict(original=m(571, 571), cauca=m(14, 14))
        self.assertGreater(selection_key(retained), selection_key(worse))


if __name__ == "__main__":
    unittest.main()
