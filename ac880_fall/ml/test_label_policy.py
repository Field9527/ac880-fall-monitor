import cache_env
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from train_fall_lite import split_index


class LabelPolicyTests(unittest.TestCase):
    def test_missing_xml_is_excluded_without_explicit_review(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "img").mkdir()
            (root / "xml").mkdir()
            for n in range(3):
                cv2.imwrite(
                    str(root / "img" / ("%d.png" % n)), np.full((20, 30, 3), n * 50, np.uint8)
                )
            (root / "xml/0.xml").write_text(
                "<annotation><object><name>fall</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>20</xmax><ymax>16</ymax></bndbox></object></annotation>"
            )
            review = root / "review.json"
            review.write_text('{"items":[]}')
            output = root / "split.json"
            items = split_index(root, output, review)
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["label_status"], "source_xml")
            self.assertEqual(json.loads(output.read_text())["excluded_unlabelled"], 2)
            # 复用原完整划分时，不重排保留集分组。
            p = root / "img/1.png"
            import random

            keys = sorted(
                hashlib.md5(q.read_bytes()).hexdigest() for q in (root / "img").glob("*.png")
            )
            random.Random(0).shuffle(keys)
            split = "val" if hashlib.md5(p.read_bytes()).hexdigest() == keys[0] else "train"
            record = dict(
                filename=p.name,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                label="no_visible_fallen_person",
                split=split,
            )
            review.write_text(json.dumps(dict(items=[record])))
            selected = split_index(root, output, review)
            self.assertEqual(len(selected), 2)
            reviewed = next(i for i in selected if not i["has_xml"])
            self.assertEqual(reviewed["split"], split)
            self.assertEqual(reviewed["boxes"], [])
            record["sha256"] = "mismatch"
            review.write_text(json.dumps(dict(items=[record])))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                split_index(root, output, review)


if __name__ == "__main__":
    unittest.main()
