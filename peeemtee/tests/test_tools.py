import numpy as np
from unittest import TestCase
from peeemtee.tools import calculate_charges, calculate_histogram_data


class TestTools(TestCase):

    def test_calculate_charges(self):
        data = np.array([[0, 1, -45, -53, 0, -1],
                              [-5, 3, -145, -253, 3, -5],
                              [0, 0, -44, -12, -1, 1]])
        self.assertListEqual(list(calculate_charges(data, 0, 2, 2, 4)),
                             [99, 396, 56])

    def test_calculate_histogram_data(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8,9]
        histogram_data = calculate_histogram_data(data, bins=5)
        x_result = [1.8,
                    3.4000000000000004,
                    5.0,
                    6.6000000000000005,
                    8.2000000000000011]
        y_result = [2, 2, 1, 2, 2]
        self.assertListEqual(list(histogram_data[0]), x_result)
        self.assertListEqual(list(histogram_data[1]), y_result)
