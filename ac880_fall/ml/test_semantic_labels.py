import unittest
from pathlib import Path
from prepare_semantic_labels import legacy_record, coco_record


class PartialLabelPolicy(unittest.TestCase):
    def test_fall_box_does_not_label_other_humans_as_background(self):
        row = legacy_record(dict(path="example.jpg", boxes=[[1, 2, 30, 40]], split="train"))
        self.assertEqual(row["objects"][0]["posture"], "fallen")
        self.assertFalse(row["supervision"]["object_background_known"])

    def test_missing_annotation_is_not_a_verified_negative(self):
        row = legacy_record(dict(path="unknown.jpg", boxes=[], split="train"))
        self.assertIsNone(row["supervision"]["fall_absent"])

    def test_verified_hand_negative_still_has_unknown_person_background(self):
        row = legacy_record(
            dict(path="hand.jpg", boxes=[], label_status="no_visible_fallen_person")
        )
        self.assertTrue(row["supervision"]["fall_absent"])
        self.assertFalse(row["supervision"]["object_background_known"])

    def test_coco_person_never_implies_not_fallen(self):
        item = dict(
            path="images/1.jpg",
            id=1,
            sha256="example",
            split="train",
            license={},
            annotations=[dict(category_id=1, bbox=[1, 2, 20, 30], iscrowd=0)],
        )
        row = coco_record(item, {1: "person"}, Path("dataset"))
        self.assertIsNone(row["objects"][0]["posture"])
        self.assertIsNone(row["supervision"]["fall_absent"])
        self.assertNotIn("hand_region", row["supervision"]["background_known_classes"])
        self.assertFalse(row["supervision"]["additional_classes_background_known"])


if __name__ == "__main__":
    unittest.main()
