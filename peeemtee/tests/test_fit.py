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
