import numpy as np
from unittest import TestCase
from peeemtee.pmt_resp_func import ChargeHistFitter


class TestTools(TestCase):
    def test_fit_pmt_resp_func(self):
        x, y = np.loadtxt(
            "./peeemtee/tests/samples/charge_distribution.txt", unpack=True
        )
        fitter = ChargeHistFitter()
        fitter.pre_fit(x, y, print_level=0)
        fitter.fit_pmt_resp_func(x, y, print_level=0)

        assert (
            np.sum((y - fitter.opt_prf_values) ** 2 / fitter.opt_prf_values)
            / 195
            < 1.5
        )
        fitter = ChargeHistFitter()
        fitter.pre_fit(x, y, print_level=0, valley=0.1, calculate_hesse=True)
        fitter.fit_pmt_resp_func(x, y, print_level=0, fixed_parameters=[])

        assert (
            np.sum((y - fitter.opt_prf_values) ** 2 / fitter.opt_prf_values)
            / 195
            < 2
        )

    def test_fit_pmt_resp_func_uap(self):
        x, y = np.loadtxt(
            "./peeemtee/tests/samples/charge_distribution.txt", unpack=True
        )
        fitter = ChargeHistFitter()
        fitter.pre_fit(x, y, print_level=0)
        fitter.fit_pmt_resp_func(x, y, print_level=0, mod="uap")

        assert (
            np.sum((y - fitter.opt_prf_values) ** 2 / fitter.opt_prf_values)
            / 195
            < 2
        )

    def test_fit_pmt_resp_func_fixed_ped_spe(self):
        x, y = np.loadtxt(
            "./peeemtee/tests/samples/charge_distribution_fixed_ped_spe.txt",
            unpack=True,
        )
        ped_mean = -0.002328480440935881
        ped_sigma = 0.001512095679762637
        spe_charge = 0.019105690941272496
        spe_sigma = 0.009400739884996042

        fitter = ChargeHistFitter()
        fitter.fix_ped_spe(ped_mean, ped_sigma, spe_charge, spe_sigma)
        fitter.pre_fit(x, y)
        fitter.fit_pmt_resp_func(x, y, n_gaussians=25)
        assert (
            np.sum((y - fitter.opt_prf_values) ** 2 / fitter.opt_prf_values)
            / 195
            < 20
        )

    def test_single_gaussian_values(self):
        x, y = np.loadtxt(
            "./peeemtee/tests/samples/single_gaussian_values.txt", unpack=True
        )
        popt = {
            "entries": 254.67033173856782,
            "nphe": 0.09948503314900548,
            "ped_mean": -0.0010005760657697902,
            "ped_sigma": 0.012413241559483338,
            "spe_charge": 0.2016217091151677,
            "spe_sigma": 0.10110260065504383,
        }
        fitter = ChargeHistFitter()
        fitter.popt_prf = popt
        np.testing.assert_array_equal(y, fitter.single_gaussian_values(x, 1))
