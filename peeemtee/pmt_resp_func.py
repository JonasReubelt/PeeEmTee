#!/usr/bin/env python

import numpy as np
from scipy import optimize
from math import factorial
import matplotlib.pylab as plt


def gaussian(x, x0, sigma, A):
    return A/np.sqrt(2*np.pi) / sigma * np.exp(- .5 * (x-x0)**2 / sigma**2)

def poisson(x, l):
    return np.exp(-l) * l**x / factorial(x)



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
        def quality_function(s_params):
            return np.sum(((gaussian(x, *s_params) - y))**2)
        return quality_function

    mean_0 = x[y.argmax()]
    above_half_max = x[y >= y.max() / 2]
    sigma_0 = (above_half_max[-1] - above_half_max[0]) / 2.355
    A_0 = y.max() * np.sqrt(2 * np.pi) * sigma_0

    start_values = [mean_0, sigma_0, A_0]

    qfunc = make_quality_function(x, y)

    bounds = [(mean_0 - sigma_0, mean_0 + sigma_0),
              (.5 * sigma_0, 2 * sigma_0),
              (.5 * A_0, 2 * A_0)]

    opt = optimize.minimize(qfunc, start_values, bounds=bounds)

    return opt.x

class ChargeHistFitter(object):
    """
    Class that provides simple gaussian fit methods and
    fit of pmt response function to pmt charge histogram.
    """

    def __init__(self):
        self.fit_parameters = {}

    def pmt_resp_func(self, x, params, n_gaussians):
        func = .0
        for i in range(n_gaussians):
            pois = poisson(i, params[0])
            sigma = np.sqrt(i * params[2]**2 + self.ped_sigma**2)
            arg = (x - (i * params[1] + self.ped_mean)) / sigma
            func += pois / sigma * np.exp(-0.5 * arg**2)
        return  params[3] * func / np.sqrt(2 * np.pi)

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
                cond = x > (popt_ped[0] + n_sigma * popt_ped[1])
            else:
                cond = ((x > (popt_ped[0] + n_sigma * popt_ped[1]))
                        & (x < spe_upper_bound))
        else:
            if spe_upper_bound is None:
                cond = x > valley
            else:
                cond = (x > valley) & (x < spe_upper_bound)
        x_spe, y_spe = x[cond], y[cond]

        popt_spe = fit_gaussian(x_spe, y_spe)

        self.ped_mean = popt_ped[0]
        self.ped_sigma = popt_ped[1]
        self.ped_A = popt_ped[2]

        self.spe_mean = popt_spe[0]
        self.spe_sigma = popt_spe[1]
        self.spe_A = popt_spe[2]

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



    def fit_pmt_resp_func(self, x, y, n_gaussians, scale_x_values=False):
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
        scale_x_values: bool
            might be helpful if x values are very small

        """
        def make_quality_function(x, y, n_gaussians):
            def quality_function(params):
                return np.sum(((self.pmt_resp_func(x, params, n_gaussians) - y))**2)
            return quality_function

        scale_factor = 1
        if scale_x_values:
            scale_factor = np.max(x)
            x = x / scale_factor


        qfunc = make_quality_function(x, y, n_gaussians)

        entries_start = (self.ped_A + self.spe_A) / scale_factor
        spe_charge_start = self.spe_charge / scale_factor
        spe_sigma_start = self.spe_sigma / scale_factor

        start_params = [self.nphe,
                        spe_charge_start,
                        spe_sigma_start,
                        entries_start]

        bounds = [(.5 * self.nphe, 2 * self.nphe),
                  (.5 * spe_charge_start, 1.5 * spe_charge_start),
                  (.5 * spe_sigma_start, 1.5 * spe_sigma_start),
                  (entries_start / 10, entries_start * 10)]

        self.ped_mean = self.ped_mean / scale_factor
        self.ped_sigma = self.ped_sigma / scale_factor

        opt = optimize.minimize(qfunc, start_params, bounds=bounds)

        self.ped_mean = self.ped_mean * scale_factor
        self.ped_sigma = self.ped_sigma * scale_factor

        self.n_gaussians = n_gaussians
        self.nphe = opt.x[0]
        self.spe_charge = opt.x[1] * scale_factor
        self.spe_sigma = opt.x[2] * scale_factor
        self.entries = opt.x[3] * scale_factor

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

        params = [self.nphe, self.spe_charge, self.spe_sigma, self.entries]

        plt.plot(xs, self.pmt_resp_func(xs, params, self.n_gaussians))
