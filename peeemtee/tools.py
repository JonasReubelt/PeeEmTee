#!/usr/bin/env python

import os
import numpy as np
from scipy.optimize import curve_fit
from numba import jit
import h5py
import codecs
import datetime
import pytz.reference

TIMEZONE = pytz.reference.LocalTimezone()


def gaussian(x, mean, sigma, A):
    return (
        A
        / np.sqrt(2 * np.pi)
        / sigma
        * np.exp(-0.5 * (x - mean) ** 2 / sigma ** 2)
    )


def gaussian_with_offset(x, mean, sigma, A, offset):
    return (
        A
        / np.sqrt(2 * np.pi)
        / sigma
        * np.exp(-0.5 * (x - mean) ** 2 / sigma ** 2)
        + offset
    )


def calculate_charges(
    waveforms, ped_min, ped_max, sig_min, sig_max, method="sum"
):
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
    method: string
        method used for "integration"
        "sum" -> np.sum
        "trapz" -> np.trapz

    Returns
    -------
    charges: np.array
        1D array with charges matching axis 0 of the waveforms array

    """
    sig_ped_ratio = (sig_max - sig_min) / (ped_max - ped_min)
    if method == "sum":
        func = np.sum
    elif method == "trapz":
        func = np.trapz
    else:
        print("unknown method. try sum or trapz!")
        return 0
    pedestals = func(waveforms[:, ped_min:ped_max], axis=1)
    signals = func(waveforms[:, sig_min:sig_max], axis=1)
    charges = -(signals - pedestals * sig_ped_ratio)

    return charges


def calculate_transit_times(
    signals, baseline_min, baseline_max, threshold, polarity="negative"
):
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
    polarity: str
        'positive' if PMT signals have positive polarity,
        'negative' if PMT signals have negative polarity

    Returns
    -------
    charges: np.array
        1D array with transit times matching axis 0 of the signals array

    """
    zeroed_signals = (
        signals.T - np.mean(signals[:, baseline_min:baseline_max], axis=1)
    ).T
    if polarity == "negative":
        transit_times = np.argmax(zeroed_signals < threshold, axis=1)
    elif polarity == "positive":
        transit_times = np.argmax(zeroed_signals > threshold, axis=1)
    else:
        print("polarity has to be 'positive' or 'negative'")
        return None
    return transit_times[transit_times != 0]


def bin_data(data, bins=10, range=None, density=False):
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
    y, x = np.histogram(data, bins=bins, range=range, density=density)
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


def calculate_mean_signal(signals, shift_by="min"):
    """
    Calculates mean signals from several PMT signals with shifting the signals
    by their minimum or maximum to correct for time jitter

    Parameters
    ----------
    signals: np.array
        2D numpy array with one signal (y-values) in each row
        [[signal1],
        [signal2],
        ...]
    shift_by: str
        shift by "min" or "max" of the signal to correct for time jitter
    Returns
    -------
    mean_signal: (np.array, np.array)
        x and y values of mean signal
    """
    rolled_signals = []
    if shift_by == "min":
        f = np.argmin
    elif shift_by == "max":
        f = np.argmax
    else:
        print("can only shift by 'min' or 'max'")
    nx = signals.shape[1]
    xs = np.arange(nx)
    for signal in signals:
        shift = f(signal)
        rolled_signals.append(np.roll(signal, -shift + int(nx / 2)))
    mean_signal = np.mean(rolled_signals, axis=0)
    return mean_signal


@jit(nopython=True)
def peak_finder(waveforms, threshold):  # pragma: no cover
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


def find_nominal_hv(filename, nominal_gain):
    """
    Finds nominal HV of a measured PMT dataset

    Parameters
    ----------
    filename: string
    nominal gain: float
        gain for which the nominal HV should be found

    Returns
    -------
    nominal_hv: int
        nominal HV
    """

    f = h5py.File(filename, "r")
    gains = []
    hvs = []
    keys = f.keys()
    for key in keys:
        gains.append(f[key]["fit_results"]["gain"][()])
        hvs.append(int(key))
    f.close()
    gains = np.array(gains)
    hvs = np.array(hvs)

    diff = abs(np.array(gains) - nominal_gain)
    nominal_hv = int(hvs[diff == np.min(diff)])
    return nominal_hv


def calculate_rise_times(waveforms, relative_thresholds=(0.1, 0.9)):
    """
    Calculates rise times of waveforms

    Parameters
    ----------
    waveforms: np.array
        2D numpy array with one waveform (y-values) in each row
        [[waveform1],
        [waveform2],
        ...]
    relative_thresholds: tuple(float)
        relative lower and upper threshold inbetween which to calculate rise time

    Returns
    -------
    rise_times: np.array
        rise times
    """
    mins = np.min(waveforms, axis=1)
    argmins = np.argmin(waveforms, axis=1)
    rise_times = []
    for min, argmin, waveform in zip(mins, argmins, waveforms):
        below_first_thr = waveform > (min * relative_thresholds[0])
        below_second_thr = waveform > (min * relative_thresholds[1])
        try:
            first_time = argmin - np.argmax(below_first_thr[:argmin][::-1])
            second_time = argmin - np.argmax(below_second_thr[:argmin][::-1])
        except ValueError:
            first_time = 0
            second_time = 0
        rise_times.append(second_time - first_time)
    return np.array(rise_times)


def read_spectral_scan(filename):
    """Reads wavelengths and currents from spectral PMT or PHD scan

    Parameters
    ----------
    filename: str

    Returns
    -------
    (wavelengths, currents): (np.array(float), np.array(float))
    """
    data = np.loadtxt(filename, unpack=True, encoding="latin1")
    with codecs.open(filename, "r", encoding="utf-8", errors="ignore") as f:
        dcs = f.read().split("\n")[-2].split("\t")
    wavelengths = data[0]
    currents = data[1]
    dc = np.linspace(float(dcs[-2]), float(dcs[-1]), len(currents))
    currents = currents - dc
    return wavelengths, currents


def read_datetime(filename):
    """Reads time of a spectral PMT or PHD scan

    Parameters
    ----------
    filename: str

    Returns
    -------
    time: str
    """
    f = codecs.open(filename, "r", encoding="utf-8", errors="ignore")
    datetime_string = f.read().split("\n")[2]
    f.close()
    return datetime_string.split(" ")[1] + ";" + datetime_string.split(" ")[2]


def convert_to_secs(date_time):
    """Converts time string to seconds

    Parameters
    ----------
    date_time: str

    Returns
    -------
    unix time in seconds: int
    """
    t = datetime.datetime.strptime(date_time, "%Y-%m-%d;%H:%M:%S")
    return t.timestamp() + TIMEZONE.utcoffset(t).seconds


def choose_ref(phd_filenames, pmt_filename):
    """Chooses reference measurement closest (in time) to the actual measurement

    Parameters
    ----------
    phd_filenames: list(str)
    pmt_filename: str

    Returns
    -------
    phd_filename: str
    """
    diffs = []
    pmt_time = convert_to_secs(read_datetime(pmt_filename))
    for filename in phd_filenames:
        phd_time = convert_to_secs(read_datetime(filename))
        diffs.append(abs(pmt_time - phd_time))
    phd_filename = phd_filenames[np.argmin(diffs)]
    return phd_filename


def remove_double_peaks(peaks, distance=20):
    """Removes secondary peaks with a distance <= distance from the primary
       peak from 2D array of peaks

    Parameters
    ----------
    peaks: 2D array of peaks
    distance: float

    Returns
    -------
    new_peaks: 2D np.array
    """
    new_peaks = []
    for peak in peaks:
        mp = -(distance + 1)
        new_peak = []
        for p in peak:
            if np.fabs(mp - p) >= distance:
                new_peak.append(p)
                mp = p
        new_peaks.append(new_peak)
    return np.array(new_peaks)


def peaks_with_signal(peaks, signal_range):
    """Returns peaks with at least one peak in signal_range

    Parameters
    ----------
    peaks: 2D array of peaks
    signal_range: tuple(float)
        (min, max) of signal window

    Returns
    -------
    peaks_with_signal: 2D np.array
    """
    peaks_with_signal = []
    for peak in peaks:
        got_signal = False
        for p in peak:
            if p > signal_range[0] and p < signal_range[1]:
                got_signal = True
        if got_signal:
            peaks_with_signal.append(peak)
    return peaks_with_signal
