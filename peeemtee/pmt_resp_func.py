# !/usr/bin/env python

import numpy as np
from scipy.stats.distributions import poisson
from iminuit import Minuit
from .tools import gaussian


def fit_gaussian(x, y, errordef=10, print_level=1, calculate_hesse=False):
    """
    Fit a gaussian to data using iminuit migrad

    Parameters
    ----------
    x: np.array
        x values of the data
    y: np.array
        y values of the data
    errordef: int, default: 10
    print_level: int, default: 1
        0: quiet, 1: print fit details

    Returns
    -------
    dict, dict:
        optimal parameters {"mean": mean, "sigma": sigma, "A": A}
        covariance matrix {("mean","mean"): cov_mean, ("mean", "sigma"): ...}
    """
    def make_quality_function(x, y):
        def quality_function(mean, sigma, A):
            return np.sum(((gaussian(x, mean, sigma, A) - y))**2)

        return quality_function

    mean_start = x[y.argmax()]
    above_half_max = x[y >= y.max() / 2]
    if len(above_half_max) == 1:
        sigma_start = 1
    else:
        sigma_start = (above_half_max[-1] - above_half_max[0]) / 2.355
    A_start = y.max() * np.sqrt(2 * np.pi) * sigma_start

    qfunc = make_quality_function(x, y)

    kwargs = {"mean": mean_start, "sigma": sigma_start, "A": A_start}

    m = Minuit(
        qfunc,
        errordef=errordef,
        pedantic=False,
        print_level=print_level,
        **kwargs,
    )
    m.migrad()
    if calculate_hesse:
        m.hesse()
        return m.values, m.covariance
    return m.values, None


class ChargeHistFitter(object):
    """
    Class that provides simple gaussian fit methods and
    fit of pmt response function to pmt charge histogram.

    Usage:
        fitter = ChargeHistFitter()
        # if fixed ped and spe is required for higher nphe measurements
            fitter.fix_ped_spe(ped_mean, ped_sigma, spe_charge, spe_sigma)
        fitter.pre_fit(x, y)
        fitter.fit_pmt_resp_func(x, y)

    Plotting the results:
        plt.plot(x, fitter.pmt_resp_func(x, **fitter.popt_prf))

    """
    def __init__(self):
        self.fixed_ped_spe = False

    def _calculate_single_gaussian_shape(self, n):
        popt = self.popt_prf
        A = poisson.pmf(int(n), popt["nphe"]) * popt["entries"]
        sigma = np.sqrt(n * popt["spe_sigma"]**2 + popt["ped_sigma"]**2)
        mean = n * popt["spe_charge"] + popt["ped_mean"]
        return mean, sigma, A

    def single_gaussian_values(self, x, n):
        """
        Calculates single gaussian y-values corresponding to the input x-values

        Parameters
        ----------
        x: np.array
            1D array of x-values the gaussian y-values should be calculated for
        n: int
            multiplicity of the gaussian the y-values should be calculated for

        Returns
        -------
        y: np.array
            y-values of the gaussian

        """
        mean, sigma, A = self._calculate_single_gaussian_shape(n)
        return gaussian(x, mean, sigma, A)

    def pmt_resp_func(self, x, nphe, ped_mean, ped_sigma, spe_charge,
                      spe_sigma, entries):
        func = 0.0
        for i in range(self.n_gaussians):
            pois = poisson.pmf(int(i), nphe)
            sigma = np.sqrt(i * spe_sigma**2 + ped_sigma**2)
            arg = (x - (i * spe_charge + ped_mean)) / sigma
            func += pois / sigma * np.exp(-0.5 * arg**2)
        func = entries * func / np.sqrt(2 * np.pi)
        return func

    def pmt_resp_func_uap(
        self,
        x,
        nphe,
        ped_mean,
        ped_sigma,
        spe_charge,
        spe_sigma,
        entries,
        uap_mean,
        uap_sigma,
        uap_A,
    ):
        func = 0.0
        for i in range(self.n_gaussians):
            pois = poisson.pmf(int(i), nphe)
            sigma = np.sqrt(i * spe_sigma**2 + ped_sigma**2)
            arg = (x - (i * spe_charge + ped_mean)) / sigma
            func += pois / sigma * np.exp(-0.5 * arg**2)
        func = entries * func / np.sqrt(2 * np.pi)
        func += gaussian(x, uap_mean, uap_sigma, uap_A)
        return func

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
        self.fixed_ped_spe = True
        self.popt_ped = {"mean": ped_mean, "sigma": ped_sigma}
        self.popt_spe = {"sigma": spe_sigma}
        self.spe_charge = spe_charge

    def pre_fit(
        self,
        x,
        y,
        valley=None,
        spe_upper_bound=None,
        n_sigma=3,
        errordef=10,
        print_level=1,
        calculate_hesse=False,
    ):
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
        print_level: int, default: 1
            0: quiet, 1: print fit details
        """

        if self.fixed_ped_spe:
            popt, pcov = fit_gaussian(
                x,
                y,
                errordef=errordef,
                print_level=print_level,
                calculate_hesse=calculate_hesse,
            )
            self.popt_gauss = popt
            self.nphe = popt["mean"] / self.spe_charge
            if self.nphe < 0:
                self.nphe = 0
            self.entries = np.max(y)
            self.n_gaussians = int(self.nphe * 2)
            if print_level > 0:
                print(self.entries)

        else:

            if valley is None:
                x_ped, y_ped = x, y
            else:
                cond = x < valley
                x_ped, y_ped = x[cond], y[cond]

            popt_ped, pcov_ped = fit_gaussian(
                x_ped,
                y_ped,
                print_level=print_level,
                calculate_hesse=calculate_hesse,
            )

            if valley is None:
                if spe_upper_bound is None:
                    cond = x > (popt_ped["mean"] + n_sigma * popt_ped["sigma"])
                else:
                    cond = (x >
                            (popt_ped["mean"] + n_sigma * popt_ped["sigma"])
                            ) & (x < spe_upper_bound)
            else:
                if spe_upper_bound is None:
                    cond = x > valley
                else:
                    cond = (x > valley) & (x < spe_upper_bound)
            x_spe, y_spe = x[cond], y[cond]

            popt_spe, pcov_spe = fit_gaussian(
                x_spe,
                y_spe,
                errordef=errordef,
                print_level=print_level,
                calculate_hesse=calculate_hesse,
            )

            self.popt_ped = popt_ped
            self.pcov_ped = pcov_ped
            self.popt_spe = popt_spe
            self.pcov_spe = pcov_spe
            self.opt_ped_values = gaussian(x, **popt_ped)
            self.opt_spe_values = gaussian(x, **popt_spe)

            self.spe_charge = popt_spe["mean"] - popt_ped["mean"]
            self.nphe = -np.log(popt_ped["A"] /
                                (popt_ped["A"] + popt_spe["A"]))
            if self.nphe < 0:
                self.nphe = 0
            self.n_gaussians = 10

    def fit_pmt_resp_func(
        self,
        x,
        y,
        n_gaussians=None,
        errordef=10,
        print_level=1,
        mod=False,
        strong_limits=True,
        fixed_parameters=["ped_mean", "ped_sigma"],
    ):
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
        errordef: int
            parses "errordef" from iminuit
        mod: string
            if False: no modification
            "uap": fits an additional underamplified pulse gaussian
        print_level: int, default: 1
            0: quiet, 1: print fit details
        """
        if n_gaussians:
            self.n_gaussians = n_gaussians
        if not mod:
            func = self.pmt_resp_func

        if mod == "uap":
            func = self.pmt_resp_func_uap

        self.used_fit_function = func

        def make_quality_function(x, y, mod):
            if not mod:

                def quality_function(nphe, ped_mean, ped_sigma, spe_charge,
                                     spe_sigma, entries):
                    model = func(
                        x,
                        nphe,
                        ped_mean,
                        ped_sigma,
                        spe_charge,
                        spe_sigma,
                        entries,
                    )
                    mask = y != 0
                    return np.sum(((model[mask] - y[mask]))**2 / y[mask])

            if mod == "uap":

                def quality_function(
                    nphe,
                    spe_charge,
                    ped_mean,
                    ped_sigma,
                    spe_sigma,
                    entries,
                    uap_mean,
                    uap_sigma,
                    uap_A,
                ):
                    model = func(
                        x,
                        nphe,
                        ped_mean,
                        ped_sigma,
                        spe_charge,
                        spe_sigma,
                        entries,
                        uap_mean,
                        uap_sigma,
                        uap_A,
                    )
                    mask = y != 0
                    return np.sum(((model[mask] - y[mask]))**2 / y[mask])

            return quality_function

        qfunc = make_quality_function(x, y, mod=mod)
        self.qfunc = qfunc
        if self.fixed_ped_spe:
            entries_start = self.entries
        else:
            entries_start = self.popt_ped["A"] + self.popt_spe["A"]

        kwargs = {
            "nphe": self.nphe,
            "spe_charge": self.spe_charge,
            "spe_sigma": self.popt_spe["sigma"],
            "entries": entries_start,
            "ped_mean": self.popt_ped["mean"],
            "ped_sigma": self.popt_ped["sigma"],
        }
        if strong_limits:
            kwargs["limit_nphe"] = (0, self.nphe * 2)
            kwargs["limit_spe_charge"] = (
                self.spe_charge / 2,
                self.spe_charge * 2,
            )
            kwargs["limit_spe_sigma"] = (0, self.popt_spe["sigma"] * 2)
            kwargs["limit_entries"] = (0, entries_start * 2)

        for parameter in fixed_parameters:
            kwargs[f"fix_{parameter}"] = True
        if self.fixed_ped_spe:
            kwargs["fix_spe_charge"] = True
            kwargs["fix_spe_sigma"] = True
            kwargs["fix_ped_mean"] = True
            kwargs["fix_ped_sigma"] = True

        if mod == "uap":
            kwargs["uap_mean"] = self.popt_spe["mean"] / 5
            kwargs["uap_sigma"] = self.popt_spe["sigma"] / 5
            kwargs["uap_A"] = entries_start / 50
            kwargs["limit_uap_mean"] = (
                self.popt_ped["mean"],
                self.popt_spe["mean"],
            )
            kwargs["limit_uap_A"] = (0, entries_start / 10)
            kwargs["limit_uap_sigma"] = (0, self.popt_spe["sigma"])

        self.m = Minuit(
            qfunc,
            errordef=errordef,
            pedantic=False,
            print_level=print_level,
            throw_nan=True,
            **kwargs,
        )
        try:
            self.m.migrad()
        except RuntimeError:
            self.success = False
        else:
            self.success = True
        # self.m.hesse()
        self.popt_prf = self.m.values
        self.opt_prf_values = func(x, **self.m.values)
        self.pcov_prf = self.m.covariance
