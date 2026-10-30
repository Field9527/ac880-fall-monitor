"""验证手部复核不会整体屏蔽独立倒地目标。"""

import copy
import unittest
from hand_region_review import review_hand_region


class HandRegionTests(unittest.TestCase):
    def observation(self):
        close = dict(bbox=[20, 30, 100, 120], score=0.99)
        person = dict(bbox=[300, 100, 250, 110], score=0.92)
        return dict(
            detections=[close, person],
            bbox=close["bbox"],
            score=0.99,
            area=12000,
            _roi="old hand crop",
            source_monotonic=10,
            hand_sampled_at=10,
            hand=dict(
                bbox=[18, 28, 104, 124],
                landmarks=[[30, 40]] * 21,
                palm_score=0.9,
                landmark_score=0.99,
            ),
        )

    def test_review_only_hand_box_and_preserve_independent_fallen_person(self):
        before = self.observation()
        after = review_hand_region(before, 11)
        self.assertEqual(len(after["detections"]), 1)
        self.assertEqual(after["bbox"], [300, 100, 250, 110])
        self.assertEqual(after["score"], 0.92)
        self.assertEqual(after["reviewed_candidates"][0]["region"], "verified_hand_region")
        self.assertNotIn("_roi", after)
        self.assertEqual(len(before["detections"]), 2)

    def test_small_hand_inside_whole_body_box_does_not_veto_body(self):
        value = self.observation()
        value["detections"] = [dict(bbox=[0, 0, 640, 360], score=0.99)]
        self.assertIs(review_hand_region(value, 11), value)

    def test_stale_mismatched_or_low_quality_hand_cannot_suppress_anything(self):
        for change in ("stale", "time_mismatch", "low_palm", "low_landmark", "missing_points"):
            value = self.observation()
            if change == "stale":
                value["hand_sampled_at"] = 6
            if change == "time_mismatch":
                value["source_monotonic"] = 7
            if change == "low_palm":
                value["hand"]["palm_score"] = 0.79
            if change == "low_landmark":
                value["hand"]["landmark_score"] = 0.94
            if change == "missing_points":
                value["hand"]["landmarks"] = []
            self.assertIs(review_hand_region(value, 11), value, change)

    def test_removing_last_hand_candidate_clears_score_and_bbox(self):
        value = self.observation()
        value["detections"] = value["detections"][:1]
        result = review_hand_region(value, 11)
        self.assertEqual(result["detections"], [])
        self.assertIsNone(result["bbox"])
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["reason"], "hand_region_reviewed_nonfall")


if __name__ == "__main__":
    unittest.main()
