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
