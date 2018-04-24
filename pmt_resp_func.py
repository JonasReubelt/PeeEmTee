#!/usr/bin/env python

import numpy as np
from scipy import optimize


def gaussian(x, x0, sigma, A):
    return A/np.sqrt(2*np.pi) / sigma * np.exp(- .5 * (x-x0)**2 / sigma**2)

def fit_gaussian(x, y):
   
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

def pre_fit(x, y, valley=None, spe_upper_bound=None, n_sigma=5):
                    
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
    
    return popt_ped, popt_spe
