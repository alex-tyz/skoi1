import unittest

import cv2
import numpy as np

from sobel import KX, KY, _reflect101, sobel_native, sobel_numpy, sobel_opencv


class SobelTests(unittest.TestCase):
    def test_kernels_sum_to_zero_and_are_transposed(self):
        self.assertEqual(sum(map(sum, KX)), 0)
        self.assertEqual(sum(map(sum, KY)), 0)
        self.assertEqual([list(r) for r in zip(*KX)], KY)

    def test_reflect101(self):
        self.assertEqual([_reflect101(i, 5) for i in (-2, -1, 0, 4, 5, 6)], [2, 1, 0, 4, 3, 2])

    def test_constant_image_has_zero_gradient(self):
        image = np.full((6, 8), 123, dtype=np.uint8)
        _, _, mag = sobel_native(image)
        self.assertEqual(np.abs(np.asarray(mag)).max(), 0)

    def test_vertical_edge_gives_horizontal_gradient(self):
        image = np.zeros((5, 6), dtype=np.uint8)
        image[:, 3:] = 255
        gx, gy, _ = sobel_native(image)
        self.assertTrue(np.all(np.asarray(gy) == 0))
        self.assertTrue(np.all(np.asarray(gx)[:, 2:4] == 4 * 255))

    def test_implementations_agree_with_opencv(self):
        random = np.random.default_rng(1)
        for h, w in ((3, 3), (4, 7), (17, 11), (64, 64)):
            image = random.integers(0, 256, (h, w), dtype=np.uint8)
            ref = sobel_opencv(image)
            native = sobel_native(image)
            numpy_ = sobel_numpy(image)
            for r, n, v in zip(ref, native, numpy_):
                np.testing.assert_allclose(np.asarray(n), r, atol=1e-9)
                np.testing.assert_allclose(v, r, atol=1e-9)

    def test_real_image(self):
        image = cv2.imread("images/input.png", cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (128, 128))
        np.testing.assert_allclose(np.asarray(sobel_native(image)[2]), sobel_opencv(image)[2], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
