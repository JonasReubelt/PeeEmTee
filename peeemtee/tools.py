#!/usr/bin/env python

import numpy as np

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
    times = np.tile(np.arange(waveform.shape[1]), (waveforms.shape[0], 1))
    z, xs, ys = np.histogram2d(times.flatten(),
                                   waveforms.flatten(),
                                   bins=bins,
                                   range=range)
    xs = (xs + (xs[1] - xs[0])/2)[:-1]
    ys = (ys + (ys[1] - ys[0])/2)[:-1]
    x = np.array([[x] * bins[0] for x in xs])
    y = np.array(list(ys) * bins[1])
    return x, y, z
