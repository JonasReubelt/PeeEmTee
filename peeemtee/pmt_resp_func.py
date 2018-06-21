#!/usr/bin/env python

import numpy as np
from scipy.stats.distributions import poisson
from iminuit import Minuit


def gaussian(x, mean, sigma, A):
    return A / np.sqrt(2*np.pi) / sigma * np.exp(-.5 * (x-mean)**2 / sigma**2)


def fit_gaussian(x, y):
    """
    Fit a gaussian to data using iminuit migrad

    Parameters
    ----------
    x: np.array
        x values of the data
    y: np.array
        y values of the data

    Returns
    -------
    dict(float):
        optimal parameters {"mean": mean, "sigma": sigma, "A": A}

    """
    def make_quality_function(x, y):
        def quality_function(mean, sigma, A):
            return np.sum(((gaussian(x, mean, sigma, A) - y))**2)
        return quality_function

    mean_start = x[y.argmax()]
    above_half_max = x[y >= y.max() / 2]
    sigma_start = (above_half_max[-1] - above_half_max[0]) / 2.355
    A_start = y.max() * np.sqrt(2 * np.pi) * sigma_start

    qfunc = make_quality_function(x, y)

    kwargs = {"mean": mean_start, "sigma": sigma_start, "A": A_start}

    m = Minuit(qfunc, errordef=10, **kwargs)
    m.migrad()

    return m.values

class ChargeHistFitter(object):
    """
    Class that provides simple gaussian fit methods and
    fit of pmt response function to pmt charge histogram.
    """

    def __init__(self):
        self.fixed_spe = False

    def gaussian(self, x, mean, sigma, A):
        return A / np.sqrt(2*np.pi) / sigma * np.exp(-.5 * (x-mean)**2 / sigma**2)

    def pmt_resp_func(self,
                      x,
                      nphe,
                      spe_charge,
                      spe_sigma,
                      entries):
        func = .0
        for i in range(self.n_gaussians):
            pois = poisson.pmf(int(i), nphe)
            sigma = np.sqrt(i * spe_sigma**2 + self.popt_ped["sigma"]**2)
            arg = (x - (i * spe_charge + self.popt_ped["mean"])) / sigma
            func += pois / sigma * np.exp(-0.5 * arg**2)
        return  entries * func / np.sqrt(2 * np.pi)

    def pre_fit(self, x, y, valley=None, spe_upper_bound=None, n_sigma=5):
        """
        Performs single gaussian fits to pedestal and single p.e. peaks

        Parameters
        ----------
        x: np.array
            bin centres of the charge histogram
        y: np.array
            bin counts of the charge histogram
        valley: float (default=None)
            user set x position to split hisogram in pedestal and spe
        spe_upper_bound: float (default=None)
            user set upper bound for gaussian fit of spe peak
        n_sigma: float
            spe fit range starts at:
            mean of pedestal + n_sigma * sigma of pedestal

        """
        if valley is None:
            x_ped, y_ped = x, y
        else:
            cond = x < valley
            x_ped, y_ped = x[cond], y[cond]

        popt_ped = fit_gaussian(x_ped, y_ped)

        if valley is None:
            if spe_upper_bound is None:
                cond = x > (popt_ped["mean"] + n_sigma * popt_ped["sigma"])
            else:
                cond = ((x > (popt_ped["mean"] + n_sigma * popt_ped["sigma"]))
                        & (x < spe_upper_bound))
        else:
            if spe_upper_bound is None:
                cond = x > valley
            else:
                cond = (x > valley) & (x < spe_upper_bound)
        x_spe, y_spe = x[cond], y[cond]

        popt_spe = fit_gaussian(x_spe, y_spe)

        self.popt_ped = popt_ped
        self.popt_spe = popt_spe

        self.spe_charge = popt_spe["mean"] - popt_ped["mean"]
        self.nphe = -np.log(popt_ped["A"] / (popt_ped["A"] + popt_spe["A"]))

    def fix_ped_spe(self, ped_mean, ped_sigma, spe_charge, spe_sigma):
        """
        Fixes ped and spe in fit_pmt_resp_func and sets fixed parameters

        Parameters
        ----------
        ped_mean: float
            mean of gaussian fit of pedestal
        ped_sigma: float
            sigma of gaussian fit of pedestal
        spe_charge: float
            charge of spe (spe_mean - spe_charge)
        spe_sigma: float
            sigma of gaussian fit of spe peak

        """
        self.fixed_spe = True
        self.popt_ped = {"mean": ped_mean, "sigma": ped_sigma}
        self.popt_spe = {"sigma": spe_sigma}
        self.spe_charge = spe_charge


    def fit_pmt_resp_func(self, x, y, n_gaussians):
        """
        Performs fit of pmt response function to charge histogram

        Parameters
        ----------
        x: np.array
            bin centres of the charge histogram
        y: np.array
            bin counts of the charge histogram
        n_gaussians: int
            number of gaussians to be fitted
        fixed_spe: bool
            if True: fixes spe_charge and spe_sigma in order to fit higher nphe
            charge spectra

        """
        self.n_gaussians = n_gaussians
        func = self.pmt_resp_func

        def make_quality_function(x, y):
            def quality_function(nphe, spe_charge, spe_sigma, entries):
                return np.sum(((func(x, nphe, spe_charge,
                                     spe_sigma, entries) - y))**2)
            return quality_function

        qfunc = make_quality_function(x, y)

        if self.fixed_spe:
            entries_start = self.entries
        else:
            entries_start = (self.popt_ped["A"] + self.popt_spe["A"])

        kwargs = {"nphe": self.nphe, "spe_charge": self.spe_charge,
                  "spe_sigma": self.popt_spe["sigma"], "entries": entries_start}
        if self.fixed_spe:
            kwargs["fix_spe_charge"] = True
            kwargs["fix_spe_sigma"] = True

        m = Minuit(qfunc, errordef=10, **kwargs)
        m.migrad()

        self.popt_pmt_resp_func = m.values
        self.n_gaussians = n_gaussians
