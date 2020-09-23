import numpy as np
from unittest import TestCase
from peeemtee.tools import (
    calculate_charges,
    bin_data,
    peak_finder,
    gaussian,
    gaussian_with_offset,
    calculate_transit_times,
    find_nominal_hv,
    calculate_rise_times,
    calculate_mean_signal,
    calculate_persist_data,
    read_spectral_scan,
    read_datetime,
    convert_to_secs,
    choose_ref,
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

    def test_find_nominal_hv(self):
        assert (
            find_nominal_hv(
                "peeemtee/tests/samples/waveform_data_dummy.h5", 5e6
            )
            == 1100
        )

    def test_calculate_rise_times(self):
        waveforms = np.array(
            [
                [0, 0, 0, -1, -2, -3, -2, -1, 0, 0, 0],
                [-0, -1, 2, -25, -35, -50, -30, -15, 0, -1, 3],
                [1, 0, 1, 0, -5, -15, -10, 5, 12, 15, 14],
            ]
        )
        rise_times = calculate_rise_times(waveforms, (0.1, 0.9))
        self.assertListEqual(list(rise_times), [2, 2, 1])

    def test_calculate_mean_signal(self):
        signals = np.array(
            [
                [0, 0.1, 1.2, -1.04, -5.213, -11.1, -15.43, -8.435, -1.1, -0],
                [0, 0.5, -1.8, -2.04, -15.456, -13.4, -10.56, -6.355, -1.0, -0],
                [
                    0,
                    0.2,
                    0.67,
                    -3.67,
                    -9.893,
                    -14.65,
                    -29.783,
                    -6.6587,
                    -1.5,
                    -0,
                ],
            ]
        )
        mean_signal = np.array(
            [
                0.1,
                0.62333333,
                -1.40333333,
                -5.63533333,
                -9.26333333,
                -20.223,
                -9.4979,
                -4.38666667,
                -2.11833333,
                -0.33333333,
            ]
        )
        np.testing.assert_array_almost_equal(
            calculate_mean_signal(signals), mean_signal
        )

    def test_calculate_persist_data(self):
        data = np.array([[-1, 0, 1], [1, 0, -1], [0, 0, 0]])
        x, y, z = calculate_persist_data(
            data, bins=(3, 3), range=((-0.5, 2.5), (-1.5, 1.5))
        )
        np.testing.assert_array_equal(
            x, np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        )
        np.testing.assert_array_equal(
            y, np.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
        )
        np.testing.assert_array_equal(
            z, np.array([1.0, 1.0, 1.0, 0.0, 3.0, 0.0, 1.0, 1.0, 1.0])
        )

    def test_read_spectral_scan(self):
        wl_true = np.array(
            [
                250.0,
                255.0,
                260.0,
                265.0,
                270.0,
                275.0,
                280.0,
                285.0,
                290.0,
                295.0,
                300.0,
                305.0,
                310.0,
                315.0,
                320.0,
                325.0,
                330.0,
                335.0,
                340.0,
                345.0,
                350.0,
                355.0,
                360.0,
                365.0,
                370.0,
                375.0,
                380.0,
                385.0,
                390.0,
                395.0,
                400.0,
                405.0,
                410.0,
                415.0,
                420.0,
                425.0,
                430.0,
                435.0,
                440.0,
                445.0,
                450.0,
                455.0,
                460.0,
                465.0,
                470.0,
                475.0,
                480.0,
                485.0,
                490.0,
                495.0,
                500.0,
                505.0,
                510.0,
                515.0,
                520.0,
                525.0,
                530.0,
                535.0,
                540.0,
                545.0,
                550.0,
                555.0,
                560.0,
                565.0,
                570.0,
                575.0,
                580.0,
                585.0,
                590.0,
                595.0,
                600.0,
                605.0,
                610.0,
                615.0,
                620.0,
                625.0,
                630.0,
                635.0,
                640.0,
                645.0,
                650.0,
                655.0,
                660.0,
                665.0,
                670.0,
                675.0,
                680.0,
                685.0,
                690.0,
                695.0,
                700.0,
            ]
        )
        i_true = np.array(
            [
                2.9000e-12,
                3.7000e-12,
                6.5000e-12,
                5.3000e-12,
                3.7000e-12,
                6.1000e-12,
                9.7000e-12,
                1.9900e-11,
                8.6900e-11,
                1.6050e-10,
                3.1430e-10,
                4.8090e-10,
                7.1510e-10,
                9.2110e-10,
                1.1603e-09,
                1.3731e-09,
                1.6197e-09,
                1.8747e-09,
                2.1375e-09,
                2.3783e-09,
                2.6893e-09,
                2.9387e-09,
                3.2403e-09,
                3.4995e-09,
                3.7627e-09,
                4.0113e-09,
                4.2457e-09,
                4.4869e-09,
                4.7043e-09,
                4.9221e-09,
                5.1887e-09,
                5.4751e-09,
                5.4109e-09,
                5.4907e-09,
                5.5895e-09,
                5.6699e-09,
                5.7017e-09,
                5.7039e-09,
                5.7085e-09,
                5.8863e-09,
                5.7905e-09,
                5.7125e-09,
                5.8201e-09,
                6.0673e-09,
                6.3737e-09,
                7.1737e-09,
                7.1165e-09,
                5.1479e-09,
                5.4909e-09,
                4.7877e-09,
                5.0921e-09,
                4.4365e-09,
                4.2901e-09,
                4.0823e-09,
                3.9767e-09,
                3.7987e-09,
                3.5707e-09,
                3.2545e-09,
                2.8745e-09,
                2.4909e-09,
                2.1729e-09,
                1.9031e-09,
                1.7167e-09,
                1.5563e-09,
                1.4543e-09,
                1.3097e-09,
                1.2369e-09,
                1.0981e-09,
                1.0243e-09,
                9.0890e-10,
                8.3030e-10,
                7.2470e-10,
                6.5470e-10,
                5.6970e-10,
                5.0850e-10,
                4.4070e-10,
                3.8090e-10,
                3.2890e-10,
                2.7570e-10,
                2.6570e-10,
                2.0070e-10,
                1.9370e-10,
                1.7010e-10,
                1.4410e-10,
                1.2210e-10,
                1.0810e-10,
                9.7300e-11,
                8.7900e-11,
                7.8100e-11,
                6.8700e-11,
                6.1100e-11,
            ]
        )
        wl, i = read_spectral_scan("peeemtee/tests/samples/BA0796.txt")
        np.testing.assert_array_almost_equal(wl_true, wl)
        np.testing.assert_array_almost_equal(i_true, i)

    def test_read_datetime(self):
        datetime = read_datetime("peeemtee/tests/samples/BA0796.txt")
        assert datetime == "2020-02-12;16:54:20"

    def test_convert_to_secs(self):
        assert convert_to_secs("2020-02-12;16:54:20") == 1581526460.0

    def test_choose_ref(self):
        pmt_filename = "peeemtee/tests/samples/BA0796.txt"
        phd_filenames = [
            "peeemtee/tests/samples/phd_1645.txt",
            "peeemtee/tests/samples/phd_1750.txt",
        ]
        assert choose_ref(phd_filenames, pmt_filename) == phd_filenames[0]
