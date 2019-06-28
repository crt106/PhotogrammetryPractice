from unittest import TestCase
import numpy as np

from RelativeOrientation2 import getR


class TestGetR(TestCase):
    def test_getR(self):
        ele = np.mat(np.arange(0, 12, 1)).T
        self.fail()

    def test_2(self):
        a1 = np.mat([[1, 2, 3], [3, 4, 5], [3, 4, 5]])
        a2 = np.mat([[3, 2, 1], [3, 2, 1], [3, 2, 1]])
        print(a1 * a2)
        print(a1.dot(a2))
        print(np.multiply(a1,a2))
