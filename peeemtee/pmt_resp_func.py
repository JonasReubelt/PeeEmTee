#!/usr/bin/env python

import numpy as np
from scipy import optimize
from scipy.stats.distributions import poisson
from math import factorial
import matplotlib.pylab as plt
from iminuit import Minuit


def gaussian(x, mean, sigma, A):
    return A/np.sqrt(2*np.pi) / sigma * np.exp(- .5 * (x-mean)**2 / sigma**2)


def fit_gaussian(x, y):
    """
    Fit a gaussian to data using scipy.optimize.minimize

    Parameters
    ----------
    x: np.array
        x values of the data
    y: np.array
        y values of the data

    Returns
    -------
    list(float):
        optimal parameters [x, x0, sigma, A]

    """
    def make_quality_function(x, y):
        def quality_function(mean, sigma, A):
            return np.sum(((gaussian(x, mean, sigma, A) - y))**2)
        return quality_function

    mean_0 = x[y.argmax()]
    above_half_max = x[y >= y.max() / 2]
    sigma_0 = (above_half_max[-1] - above_half_max[0]) / 2.355
    A_0 = y.max() * np.sqrt(2 * np.pi) * sigma_0

    start_values = [mean_0, sigma_0, A_0]

    qfunc = make_quality_function(x, y)

    #bounds = [(mean_0 - sigma_0, mean_0 + sigma_0),
    #          (.5 * sigma_0, 2 * sigma_0),
    #          (.5 * A_0, 2 * A_0)]
    kwargs = {"mean": mean_0, "sigma": sigma_0, "A": A_0}
    #opt = optimize.minimize(qfunc, start_values, bounds=bounds)

    m = Minuit(qfunc, errordef=10, **kwargs)
    m.migrad()

    return m.values

class ChargeHistFitter(object):
    """
    Class that provides simple gaussian fit methods and
    fit of pmt response function to pmt charge histogram.
    """

    def __init__(self):
        self.fit_parameters = {}

    def pmt_resp_func(self,
                      x,
                      nphe,
                      spe_charge,
                      spe_sigma,
                      entries,
                      n_gaussians):
        func = .0
        for i in range(n_gaussians):
            pois = poisson.pmf(int(i), nphe)
            sigma = np.sqrt(i * spe_sigma**2 + self.ped_sigma**2)
            arg = (x - (i * spe_charge + self.ped_mean)) / sigma
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

        self.ped_mean = popt_ped["mean"]
        self.ped_sigma = popt_ped["sigma"]
        self.ped_A = popt_ped["A"]

        self.spe_mean = popt_spe["mean"]
        self.spe_sigma = popt_spe["sigma"]
        self.spe_A = popt_spe["A"]

        self.spe_charge = self.spe_mean - self.ped_mean

        self.nphe = -np.log(self.ped_A / (self.ped_A + self.spe_A))

        self.fit_parameters["ped_mean"] = self.ped_mean
        self.fit_parameters["ped_sigma"] = self.ped_sigma
        self.fit_parameters["ped_A"] = self.ped_A

        self.fit_parameters["spe_mean"] = self.spe_mean
        self.fit_parameters["spe_sigma"] = self.spe_sigma
        self.fit_parameters["spe_A"] = self.spe_A

        self.fit_parameters["spe_charge"] = self.spe_charge
        self.fit_parameters["nphe"] = self.nphe



    def fit_pmt_resp_func(self, x, y, n_gaussians, fixed_spe=False, min_method="Nelder-Mead"):
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

        func = self.pmt_resp_func

        def make_quality_function(x, y, n_gaussians):
            def quality_function(nphe, spe_charge, spe_sigma, entries):
                return np.sum(((func(x, nphe, spe_charge,
                                     spe_sigma, entries, n_gaussians) - y))**2)
            return quality_function

        qfunc = make_quality_function(x, y, n_gaussians)

        if fixed_spe:
            entries_start = self.entries
        else:
            entries_start = (self.ped_A + self.spe_A)

        kwargs = {"nphe": self.nphe, "spe_charge": self.spe_charge,
                  "spe_sigma": self.spe_sigma, "entries": entries_start}
        if fixed_spe:
            kwargs["fix_spe_charge"] = True
            kwargs["fix_spe_sigma"] = True

        m = Minuit(qfunc, errordef=10, **kwargs)
        m.migrad()
        opt_params = m.values
        self.n_gaussians = n_gaussians
        if fixed_spe:
            self.nphe = opt_params["nphe"]
            self.entries = opt_params["entries"]
        else:
            self.nphe = opt_params["nphe"]
            self.spe_charge = opt_params["spe_charge"]
            self.spe_sigma = opt_params["spe_sigma"]
            self.entries = opt_params["entries"]

        self.fit_parameters["n_gaussians"] = self.n_gaussians
        self.fit_parameters["nphe"] = self.nphe
        self.fit_parameters["spe_charge"] = self.spe_charge
        self.fit_parameters["spe_sigma"] = self.spe_sigma
        self.fit_parameters["entries"] = self.entries


    def plot_pre_fit(self, xs):
        """
        Plots prefit

        parameters
        ----------
        xs: np.array
            plots gaussian(xs)
        """
        plt.plot(xs, gaussian(xs, self.ped_mean, self.ped_sigma, self.ped_A))
        plt.plot(xs, gaussian(xs, self.spe_mean, self.spe_sigma, self.spe_A))

    def plot_pmt_resp_func(self, xs):
        """
        Plots pmt response function

        parameters
        ----------
        xs: np.array
            plots self.pmt_resp_func(xs)
        """

        self.nphe, self.spe_charge, self.spe_sigma, self.entries
        plt.plot(xs, self.pmt_resp_func(xs, self.nphe, self.spe_charge, self.spe_sigma, self.entries, self.n_gaussians))
