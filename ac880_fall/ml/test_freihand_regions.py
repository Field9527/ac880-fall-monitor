import unittest
import numpy as np
from prepare_freihand_regions import project_region


class HandProjection(unittest.TestCase):
    def points(self):
        return np.c_[np.linspace(-0.1, 0.1, 21), np.linspace(-0.08, 0.08, 21), np.ones(21)]

    def test_camera_projection_and_margin_use_image_pixels(self):
        k = np.array([[500, 0, 112], [0, 500, 112], [0, 0, 1]])
        box, uv = project_region(self.points(), k, 224, 224)
        np.testing.assert_allclose(uv[0], [62, 72])
        np.testing.assert_allclose(uv[-1], [162, 152])
        np.testing.assert_allclose(box, [47, 60, 177, 164])

    def test_crop_boundary_clips_region(self):
        k = np.array([[500, 0, 30], [0, 500, 30], [0, 0, 1]])
        box, _ = project_region(self.points(), k, 224, 224)
        self.assertEqual(box[:2], [0, 0])
        self.assertLessEqual(box[2], 224)

    def test_invalid_geometry_is_rejected(self):
        for bad in (np.zeros((21, 3)), np.full((21, 3), np.nan), np.ones((20, 3))):
            with self.assertRaises(ValueError):
                project_region(bad, np.eye(3), 224, 224)


if __name__ == "__main__":
    unittest.main()
