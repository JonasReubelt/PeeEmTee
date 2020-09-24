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
    remove_double_peaks,
    peaks_with_signal,
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
                -4.00000000e-13,
                4.73333333e-13,
                3.34666667e-12,
                2.22000000e-12,
                6.93333333e-13,
                3.16666667e-12,
                6.84000000e-12,
                1.71133333e-11,
                8.41866667e-11,
                1.57860000e-10,
                3.11733333e-10,
                4.78406667e-10,
                7.12680000e-10,
                9.18753333e-10,
                1.15802667e-09,
                1.37090000e-09,
                1.61757333e-09,
                1.87264667e-09,
                2.13552000e-09,
                2.37639333e-09,
                2.68746667e-09,
                2.93694000e-09,
                3.23861333e-09,
                3.49788667e-09,
                3.76116000e-09,
                4.00983333e-09,
                4.24430667e-09,
                4.48558000e-09,
                4.70305333e-09,
                4.92092667e-09,
                5.18760000e-09,
                5.47407333e-09,
                5.40994667e-09,
                5.48982000e-09,
                5.58869333e-09,
                5.66916667e-09,
                5.70104000e-09,
                5.70331333e-09,
                5.70798667e-09,
                5.88586000e-09,
                5.79013333e-09,
                5.71220667e-09,
                5.81988000e-09,
                6.06715333e-09,
                6.37362667e-09,
                7.17370000e-09,
                7.11657333e-09,
                5.14804667e-09,
                5.49112000e-09,
                4.78799333e-09,
                5.09246667e-09,
                4.43694000e-09,
                4.29061333e-09,
                4.08288667e-09,
                3.97736000e-09,
                3.79943333e-09,
                3.57150667e-09,
                3.25538000e-09,
                2.87545333e-09,
                2.49192667e-09,
                2.17400000e-09,
                1.90427333e-09,
                1.71794667e-09,
                1.55762000e-09,
                1.45569333e-09,
                1.31116667e-09,
                1.23844000e-09,
                1.09971333e-09,
                1.02598667e-09,
                9.10660000e-10,
                8.32133333e-10,
                7.26606667e-10,
                6.56680000e-10,
                5.71753333e-10,
                5.10626667e-10,
                4.42900000e-10,
                3.83173333e-10,
                3.31246667e-10,
                2.78120000e-10,
                2.68193333e-10,
                2.03266667e-10,
                1.96340000e-10,
                1.72813333e-10,
                1.46886667e-10,
                1.24960000e-10,
                1.11033333e-10,
                1.00306667e-10,
                9.09800000e-11,
                8.12533333e-11,
                7.19266667e-11,
                6.44000000e-11,
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

    def test_remove_double_peaks(self):
        a = [[0, 19, 45, 66]]
        ar = remove_double_peaks(a, distance=20)
        np.testing.assert_array_equal(ar, np.array([[0, 45, 66]]))
        b = [[1000.5, 1020.5, 1023, 1042]]
        br = remove_double_peaks(b, distance=20)
        np.testing.assert_array_equal(br, np.array([[1000.5, 1020.5, 1042.0]]))
        c = [[25, 26, 27, 45, 54, 77]]
        cr = remove_double_peaks(c, distance=10)
        np.testing.assert_array_equal(cr, np.array([[25, 45, 77]]))

    def test_peaks_with_signal(self):
        a = [[5, 10, 20, 25]]
        signal_range = (10, 20)
        ar = peaks_with_signal(a, signal_range=signal_range)
        np.testing.assert_array_equal(ar, [])
        b = [[0, 11, 20, 25]]
        signal_range = (10, 20)
        br = peaks_with_signal(b, signal_range=signal_range)
        np.testing.assert_array_equal(br, [[0, 11, 20, 25]])
        c = [[3, 21, 29, 55]]
        signal_range = (20, 30)
        cr = peaks_with_signal(c, signal_range=signal_range)
        np.testing.assert_array_equal(cr, [[3, 21, 29, 55]])
