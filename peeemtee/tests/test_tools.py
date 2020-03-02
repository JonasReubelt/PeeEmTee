import numpy as np
from unittest import TestCase
from peeemtee.tools import (
    calculate_charges,
    bin_data,
    peak_finder,
    gaussian,
    gaussian_with_offset,
    calculate_transit_times,
)


class TestTools(TestCase):
    def test_gaussian(self):
        assert gaussian(0, 0, 1, 1) == 0.3989422804014327
        assert gaussian(0.345, 1.234, 0.5432, 108) == 20.78525811770294
        assert gaussian(1.098, -1.342, 12.34, 1029387.234) == 32635.01097991775

    def test_gaussian_with_offset(self):
        assert gaussian_with_offset(0, 0, 1, 1, 1) == 1.3989422804014327
        assert (
            gaussian_with_offset(1.2234, -2.34, 2.345, 123.23, -12.4)
            == -5.792028722690032
        )
        assert (
            gaussian_with_offset(-0.9857, 12.34, 24.345, 123.23, 34.4)
            == 36.13842765114078
        )

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

    def test_calculate_transit_times(self):
        data = np.array(
            [
                [0, -1, 1, -53, 0, 1],
                [-5, 3, -14, -253, 143, 3],
                [100, 102, 85, -9, -1, -15],
            ]
        )
        self.assertListEqual(
            list(calculate_transit_times(data, 0, 2, -10)), [3, 2, 2]
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
