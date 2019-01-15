#!/usr/bin/env python

import numpy as np
from scipy.optimize import curve_fit

def gaussian(x, mean, sigma, A):
    return A / np.sqrt(2*np.pi) / sigma * np.exp(-.5 * (x-mean)**2 / sigma**2)

def calculate_charges(waveforms, ped_min, ped_max, sig_min, sig_max):
    """
    Calculates the charges of an array of waveforms

    Parameters
    ----------
    waveforms: np.array
        2D numpy array with one waveform in each row
        [[waveform1],
        [waveform2],
        ...]
    ped_min: int
        minimum of window for pedestal integration
    ped_max: int
        maximum of window for pedestal integration
    sig_min: int
        minimum of window for signal integration
    sig_max: int
        maximum of window for signal integration

    Returns
    -------
    charges: np.array
        1D array with charges matching axis 0 of the waveforms array

    """
    pedestals = np.sum(waveforms[:, ped_min:ped_max], axis=1)
    charges = -(np.sum(waveforms[:, sig_min:sig_max], axis=1) - pedestals)

    return charges


def calculate_histogram_data(data, bins=10, range=None):
    """
    Calculates values and bin centres of a histogram of a set of data

    Parameters
    ----------
    data: list or np.array
        1D array of input data
    bins: int
        number of bins of the histogram
    range: tuple(int)
        lower and upper range of the bins

    Returns
    -------
    x: np.array
        bin centres of the histogram
    y: np.array
        values of the histogram

    """
    y, x = np.histogram(data, bins=bins, range=range)
    x = x[:-1]
    x = x + (x[1] - x[0]) / 2
    return x, y


def calculate_persist_data(waveforms, bins=(10, 10), range=None):
    """
    Calculates 2D histogram data like persistence mode on oscilloscope

    Parameters
    ----------
    waveforms: np.array
        2D numpy array with one waveform in each row
        [[waveform1],
        [waveform2],
        ...]
    bins: tuple(int)
        number of bins in both directions
    range: tuple(tuple(int))
        lower and upper range of the x-bins and y-bins

    Returns
    -------
    x: np.array
        x-bin centres of the histogram
    y: np.array
        y-bin centres of the histogram
    z: np.array
        z values of the histogram

    """
    times = np.tile(np.arange(waveforms.shape[1]), (waveforms.shape[0], 1))
    z, xs, ys = np.histogram2d(times.flatten(),
                                   waveforms.flatten(),
                                   bins=bins,
                                   range=range)
    xs = (xs + (xs[1] - xs[0])/2)[:-1]
    ys = (ys + (ys[1] - ys[0])/2)[:-1]
    x = np.array([[x] * bins[0] for x in xs])
    y = np.array(list(ys) * bins[1])
    return x.flatten(), y.flatten(), z.flatten()


def calculate_mean_signal(signals, xs, signal_range, p0=None):
    """
    Calculates mean signals from several PMT signals

    Parameters
    ----------
    signals: np.array
        2D numpy array with one signal (y-values) in each row
        [[signal1],
        [signal2],
        ...]
    xs: np.array
        x values of the signals
    signal_range: tuple(float)
        range around mean in which mean signal is calculated
    p0: list
        start parameters for gaussian fit

    Returns
    -------
    mean_signal: (np.array, np.array)
        x and y values of mean signal
    """
    means = []
    for signal in signals:
        popt, _ = curve_fit(gaussian, xs, signal, p0=p0)
        means.append(popt[0])
    n = len(xs)
    h_int = xs[-1]/n
    shifted_xs = [np.linspace(-m, n * h_int - m, n) for m in means]
    shifted_xs = np.array(shifted_xs)
    tics = round((signal_range[1] - signal_range[0]) / h_int)
    dig = np.array([np.digitize(s_xs, bins=np.linspace(signal_range[0],
                                              signal_range[1],
                                              tics))
           for s_xs in shifted_xs])
    shifted_signals = []
    for signal, d in zip(signals, dig):
        shifted_signals.append(signal[(d>0)&(d<tics)])

    shifted_signals = np.array([ss for ss in shifted_signals if len(ss)==tics])
    mean_signal_x = np.linspace(signal_range[0], signal_range[1], tics)
    mean_signal_y = np.mean(shifted_signals, axis=0)
    return mean_signal_x, mean_signal_y
