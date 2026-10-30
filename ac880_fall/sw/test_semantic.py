"""验证场景证据路由和 NanoDet 网格解码。"""

import unittest
import numpy as np
from semantic_policy import SemanticPolicy
from semantic_probe import decode


def candidate(box=(10, 10, 90, 90), score=0.95):
    return dict(box=box, score=score)


class EvidenceScenarios(unittest.TestCase):
    def run_case(self, persons=(), falls=(), objects=(), screens=(), mode="live"):
        return SemanticPolicy().evaluate(1, 10, 10.1, persons, falls, objects, screens, mode)

    def test_finger_without_person_is_uncertain_not_safe(self):
        r = self.run_case(falls=[candidate((10, 45, 90, 55))])
        self.assertEqual(r["items"][0]["status"], "needs_review")
        self.assertFalse(r["confirmed_alarm"])

    def test_horizontal_object_class_does_not_create_human(self):
        r = self.run_case(falls=[candidate()], objects=[dict(box=(10, 10, 90, 90), label="bottle")])
        self.assertEqual(r["items"][0]["status"], "needs_review")

    def test_phone_elsewhere_cannot_suppress_real_person(self):
        r = self.run_case(
            persons=[candidate()],
            falls=[candidate()],
            objects=[dict(box=(100, 100, 200, 200), label="cell phone")],
        )
        self.assertEqual(r["items"][0]["status"], "risk_candidate")

    def test_generic_phone_box_is_not_verified_screen(self):
        r = self.run_case(
            persons=[candidate()],
            falls=[candidate()],
            objects=[dict(box=(0, 0, 100, 100), label="cell phone")],
        )
        self.assertEqual(r["items"][0]["status"], "risk_candidate")

    def test_verified_screen_is_local_to_one_person(self):
        people = [candidate(), candidate((150, 10, 250, 90))]
        r = self.run_case(
            persons=people, falls=people, screens=[dict(box=(0, 0, 100, 100), verified=True)]
        )
        self.assertEqual([v["status"] for v in r["items"]], ["screen_content", "risk_candidate"])

    def test_phone_replay_does_not_disable_screen_people(self):
        r = self.run_case(
            persons=[candidate()],
            falls=[candidate()],
            screens=[dict(box=(0, 0, 100, 100), verified=True)],
            mode="phone_replay",
        )
        self.assertEqual(r["items"][0]["status"], "risk_candidate")
        self.assertEqual(r["items"][0]["source"], "phone_screen_replay")

    def test_already_lying_person_does_not_require_standing(self):
        r = self.run_case(
            persons=[candidate((0, 40, 120, 80))], falls=[candidate((0, 40, 120, 80))]
        )
        self.assertEqual(r["items"][0]["status"], "risk_candidate")

    def test_duplicate_results_never_count_twice(self):
        gate = SemanticPolicy()
        gate.evaluate(4, 10, 10.1, [candidate()], [candidate()])
        self.assertEqual(gate.evaluate(4, 10, 11, [], [])["status"], "duplicate_or_out_of_order")
        self.assertEqual(gate.evaluate(3, 10, 11, [], [])["status"], "duplicate_or_out_of_order")

    def test_old_results_and_future_timestamps_are_stale(self):
        self.assertEqual(SemanticPolicy().evaluate(1, 1, 10, [], [])["status"], "stale")
        self.assertEqual(SemanticPolicy().evaluate(1, 11, 10, [], [])["status"], "stale")

    def test_bad_confidence_and_box_fail_closed(self):
        with self.assertRaises(ValueError):
            self.run_case(persons=[candidate(score=float("nan"))])
        with self.assertRaises(ValueError):
            self.run_case(falls=[candidate((5, 5, 2, 2))])


class DecoderScenarios(unittest.TestCase):
    def outputs(self):
        return [
            v
            for stride in (8, 16, 32)
            for v in (
                np.zeros(((64 // stride) ** 2, 80), np.float32),
                np.full(((64 // stride) ** 2, 32), -50, np.float32),
            )
        ]

    def test_known_distribution_maps_to_pixel_box(self):
        out = self.outputs()
        idx = 3 * 8 + 3
        out[0][idx, 0] = 0.9
        for side in range(4):
            out[1][idx, side * 8 + 2] = 50
        items = decode(out, 64, 64, (1, 1, 0, 0), (64, 64, 3))
        self.assertEqual(items[0]["name"], "person")
        np.testing.assert_allclose(items[0]["box"], [12, 12, 44, 44], atol=1e-5)

    def test_class_aware_nms_preserves_person_phone_overlap(self):
        out = self.outputs()
        idx = 3 * 8 + 3
        for k, label in ((idx, 0), (idx + 1, 67)):
            out[0][k, label] = 0.9
            for side in range(4):
                out[1][k, side * 8 + 3] = 50
        items = decode(out, 64, 64, (1, 1, 0, 0), (64, 64, 3))
        self.assertEqual({v["name"] for v in items}, {"person", "cell phone"})

    def test_nonfinite_scores_are_rejected(self):
        out = self.outputs()
        out[0][0, 0] = np.nan
        with self.assertRaises(ValueError):
            decode(out, 64, 64, (1, 1, 0, 0), (64, 64, 3))

    def test_grid_mismatch_is_rejected(self):
        out = self.outputs()
        out[0] = out[0][:-1]
        with self.assertRaises(ValueError):
            decode(out, 64, 64, (1, 1, 0, 0), (64, 64, 3))


if __name__ == "__main__":
    unittest.main()
