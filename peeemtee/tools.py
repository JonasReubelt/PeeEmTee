#!/usr/bin/env python

import numpy as np
from scipy.optimize import curve_fit
from numba import jit


def gaussian(x, mean, sigma, A):
    return (
        A
        / np.sqrt(2 * np.pi)
        / sigma
        * np.exp(-0.5 * (x - mean) ** 2 / sigma ** 2)
    )


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


def calculate_transit_times(signals, baseline_min, baseline_max, threshold):
    """
    Calculates transit times of signals

    Parameters
    ----------
    signals: np.array
        2D numpy array with one signal waveform in each row
        [[signal1],
        [signal2],
        ...]
    baseline_min: int
        minimum of baseline calculation window
    baseline_max: int
        maximum of baseline calculation window
    threshold: float
        transit time is calculated when signal crosses threshold


    Returns
    -------
    charges: np.array
        1D array with transit times matching axis 0 of the signals array

    """
    zeroed_signals = signals - np.mean(signals[:, baseline_min:baseline_max])
    transit_times = np.argmax(zeroed_signals < threshold, axis=1)
    return transit_times[transit_times != 0]


def calculate_histogram_data(data, bins=10, range=None, normed=False):
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
    normed: boolean
        set to True to norm the histogram data

    Returns
    -------
    x: np.array
        bin centres of the histogram
    y: np.array
        values of the histogram

    """
    y, x = np.histogram(data, bins=bins, range=range, normed=normed)
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
    z, xs, ys = np.histogram2d(
        times.flatten(), waveforms.flatten(), bins=bins, range=range
    )
    xs = (xs + (xs[1] - xs[0]) / 2)[:-1]
    ys = (ys + (ys[1] - ys[0]) / 2)[:-1]
    x = np.array([[x] * bins[1] for x in xs])
    y = np.array(list(ys) * bins[0])
    return x.flatten(), y.flatten(), z.flatten()


def calculate_mean_signal(signals, use_for_shift="min", p0=None, print_level=1):
    """
    Calculates mean signals from several PMT signals

    Parameters
    ----------
    signals: np.array
        2D numpy array with one signal (y-values) in each row
        [[signal1],
        [signal2],
        ...]
    use_for_shift: string
        "min" to use minimum of signal or "fit" to use mean of gaussian fit
    p0: list
        start parameters for gaussian fit

    Returns
    -------
    mean_signal: (np.array, np.array)
        x and y values of mean signal
    """
    rolled_signals = []
    nx = signals.shape[1]
    xs = np.arange(nx)
    for signal in signals:
        if use_for_shift == "fit":
            try:
                popt, _ = curve_fit(gaussian, xs, signal, p0=p0)
                shift = int(round(popt[0]))
            except RuntimeError:
                if print_level > 0:
                    print("bad fit!")
        elif use_for_shift == "min":
            shift = np.argmin(signal)
        else:
            print(f'Unknown option for use_for_shift: "{use_for_shift}"')
            print('options are: "min" or "fit"')
            return None
        rolled_signals.append(np.roll(signal, -shift + int(nx / 2)))
    mean_signal = np.mean(rolled_signals, axis=0)
    return mean_signal


@jit(nopython=True)
def peak_finder(waveforms, threshold):
    """
    Finds peaks in waveforms

    Parameters
    ----------
    waveforms: np.array
        2D numpy array with one waveform (y-values) in each row
        [[waveform1],
        [waveform2],
        ...]
    threshold: float
        voltage value the waveform has to cross in order to identify a peak

    Returns
    -------
    peak_positions: list(list(floats))
        x and y values of mean signal
    """
    peak_positions = []
    I, J = waveforms.shape
    for i in range(I):
        peaks = []
        X = 0
        x = 0
        for j in range(J):
            if waveforms[i][j] <= threshold:
                X += j
                x += 1
                if j + 1 >= J or waveforms[i][j + 1] > threshold:
                    peaks.append(X / x)
                    X = 0
                    x = 0
        if len(peaks) > 0:
            peak_positions.append(peaks)
    return peak_positions
