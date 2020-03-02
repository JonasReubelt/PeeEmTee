import numpy as np
from unittest import TestCase
from peeemtee.tools import calculate_charges, bin_data, peak_finder


class TestTools(TestCase):
    def test_calculate_charges(self):
        data = np.array(
            [
                [0, 1, -45, -53, 0, -1],
                [-5, 3, -145, -253, 3, -5],
                [0, 0, -44, -12, -1, 1],
            ]
        )
        self.assertListEqual(
            list(calculate_charges(data, 0, 2, 2, 4)), [99, 396, 56]
        )

    def test_bin_data(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        histogram_data = bin_data(data, bins=5)
        x_result = [
            1.8,
            3.4000000000000004,
            5.0,
            6.6000000000000005,
            8.2000000000000011,
        ]
        y_result = [2, 2, 1, 2, 2]
        self.assertListEqual(list(histogram_data[0]), x_result)
        self.assertListEqual(list(histogram_data[1]), y_result)

    def test_peak_finder(self):
        test_waveforms = np.array(
            [
                [0, 0, 0, -1, -2, -1, 0, 0, 0, 0, -2, -3, -4, -2, 0, 0],
                [0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3],
                [-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        peak_positions = peak_finder(test_waveforms, -1)
        result = [[4.0, 11.5], [2.5], [15.0], [0.0]]
        self.assertListEqual(peak_positions, result)
