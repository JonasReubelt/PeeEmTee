#!/usr/bin/env python

import numpy as np
import codecs
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import thepipe as tp
import h5py
from .tools import gaussian, calculate_charges, bin_data, calculate_rise_times
from .pmt_resp_func import ChargeHistFitter
from .constants import hama_phd_qe


class FilePump(tp.Module):
    """
    iterates over a set of filenames
    """

    def configure(self):
        self.filenames = self.get("filenames")
        self.max_count = len(self.filenames)
        self.index = 0

    def process(self, blob):
        if self.index >= self.max_count:
            raise StopIteration
        blob["filename"] = self.filenames[self.index]
        self.index += 1
        return blob

    def finish(self):
        self.cprint(f"Read {self.index} files!")


class QECalibrator(tp.Module):
    """Reads measured currents from PMT and PHD and calculates the QE of the PMT
    """

    def configure(self):
        self.phd_filenames = self.get("phd_filenames")
        self.global_qe_shift = self.get("global_qe_shift")
        phd_qe = hama_phd_qe.T
        self.phd_qe_interp = interp1d(phd_qe[0], phd_qe[1], kind="cubic")

    def process(self, blob):
        pmt_filename = blob["filename"]
        phd_filename = choose_ref(self.phd_filenames, pmt_filename)
        wl_pmt, i_pmt = read_spectral_scan(pmt_filename)
        wl_phd, i_phd = read_spectral_scan(phd_filename)
        if np.allclose(wl_pmt, wl_phd):
            wl = wl_pmt + self.global_qe_shift
        else:
            self.log.error("PMT and PHD wavelengths do not match!")
            raise StopIteration
        qe = i_pmt / i_phd * self.phd_qe_interp(wl) / 100
        blob["wl"] = wl
        blob["qe"] = qe
        blob["pmt_id"] = pmt_filename.split("/")[-1].split(".")[0]
        blob["global_qe_shift"] = self.global_qe_shift
        return blob


def read_spectral_scan(filename):
    """Reads wavelengths and currents from spectral PMT or PHD scan

    Parameters
    ----------
    filename: str

    Returns
    -------
    (wavelengths, currents): (np.array(float), np.array(float))
    """
    print(filename)
    data = np.loadtxt(filename, unpack=True, encoding="latin1")
    with codecs.open(filename, "r", encoding="utf-8", errors="ignore") as f:
        dcs = f.read().split("\n")[-2].split("\t")
    dc = (float(dcs[-2]) + float(dcs[-1])) / 2
    wavelengths = data[0]
    currents = data[1] - dc
    return wavelengths, currents


def read_time(filename):
    """Reads time of a spectral PMT or PHD scan

    Parameters
    ----------
    filename: str

    Returns
    -------
    time: str
    """
    f = codecs.open(filename, "r", encoding="utf-8", errors="ignore")
    time = f.read().split("\n")[2].split(" ")[2]
    return time


def get_sec(time_str):
    """Converts time string to seconds

    Parameters
    ----------
    time_str: str

    Returns
    -------
    seconds: int
    """
    h, m, s = time_str.split(":")
    seconds = int(h) * 3600 + int(m) * 60 + int(s)
    return seconds


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
    pmt_time = get_sec(read_time(pmt_filename))
    for filename in phd_filenames:
        phd_time = get_sec(read_time(filename))
        diffs.append(abs(pmt_time - phd_time))
    phd_filename = phd_filenames[np.argmin(diffs)]
    return phd_filename


class QEWriter(tp.Module):
    """Writes QE in text file"""

    def configure(self):
        self.filepath = self.get("filepath")

    def process(self, blob):
        shift = blob["global_qe_shift"]
        pmt_id = blob["pmt_id"]
        qe_filename = f"{self.filepath}/qe_{pmt_id}_wl_shift_{shift}_nm.txt"
        np.savetxt(qe_filename, np.array([blob["wl"], blob["qe"]]).T)
        return blob


class NominalHVFinder(tp.Module):
    """
    finds the nominal HV for a certain gain
    """

    def configure(self):
        self.gain = self.get("gain")

    def process(self, blob):
        filename = blob["filename"]
        f = h5py.File(filename, "r")
        nominal_hv = self.find_nominal_hv(f, self.gain)
        blob["nominal_gain"] = self.gain
        blob["nominal_hv"] = nominal_hv
        return blob

    def find_nominal_hv(self, f, nominal_gain):
        gains = []
        hvs = []
        keys = f.keys()
        for key in keys:
            gains.append(f[key]["fit_results"]["gain"][()])
            hvs.append(int(key))
        gains = np.array(gains)
        hvs = np.array(hvs)

        diff = abs(np.array(gains) - nominal_gain)
        nominal_hv = int(hvs[diff == np.min(diff)])
        return nominal_hv


class FileReader(tp.Module):
    """
    pipe module that reads h5py files and writes waveforms and waveform info
    into the blob
    """

    def process(self, blob):
        filename = blob["filename"]
        blob["pmt_id"] = filename.split("/")[-1].split(".")[0]
        self.cprint(f"Reading file: {filename}")
        f = h5py.File(filename, "r")
        nominal_hv = blob["nominal_hv"]
        blob["waveforms"] = f[f"{nominal_hv}"]["waveforms"][:]
        blob["h_int"] = f[f"{nominal_hv}"]["waveform_info/h_int"][()]
        blob["v_gain"] = f[f"{nominal_hv}"]["waveform_info/v_gain"][()]
        blob["gain_norm"] = blob["v_gain"] * blob["h_int"] / 50 / 1.6022e-19
        f.close()
        return blob


class ChargeCalculator(tp.Module):
    """
    pipe module that calculates charges of waveforms
    """

    def configure(self):
        self.ped_range = self.get("ped_range")
        self.sig_range = self.get("sig_range")

    def process(self, blob):
        charges = calculate_charges(
            blob["waveforms"],
            self.ped_range[0],
            self.ped_range[1],
            self.sig_range[0],
            self.sig_range[1],
        )
        charges = charges * blob["gain_norm"]
        blob["charges"] = charges
        x, y = bin_data(charges, bins=200, range=(-0.3e7, 4e7))
        blob["charge_distribution"] = (x, y)
        return blob


class PMTResponseFitter(tp.Module):
    """
    pipe module that fits charge distribution with
    PMT response function
    """

    def configure(self):
        self.mod = self.get("mod")

    def process(self, blob):
        x, y = blob["charge_distribution"]
        fitter = ChargeHistFitter()
        fitter.pre_fit(x, y, print_level=0)
        fitter.fit_pmt_resp_func(
            x, y, mod=self.mod, print_level=0, fixed_parameters=[]
        )
        if not fitter.success:
            return
        blob["popt_prf"] = fitter.popt_prf
        blob["popt_ped"] = fitter.popt_ped
        blob["popt_spe"] = fitter.popt_spe
        blob["fit_function"] = fitter.used_fit_function
        return blob


class PeakToValleyCalculator(tp.Module):
    """
    pipe module that calculates the peak-to-valley ratio of a PMT response
    """

    def process(self, blob):
        fit_function = blob["fit_function"]
        fine_xs = np.linspace(-0.5e7, 3e7, 10000)
        valley_mask = (fine_xs > blob["popt_ped"]["mean"]) & (
            fine_xs < blob["popt_spe"]["mean"]
        )
        valley = np.min(fit_function(fine_xs, **blob["popt_prf"])[valley_mask])
        peak_mask = fine_xs > (
            blob["popt_ped"]["mean"] + blob["popt_ped"]["sigma"] * 3
        )
        peak = np.max(fit_function(fine_xs, **blob["popt_prf"])[peak_mask])
        blob["peak_to_valley"] = peak / valley
        return blob


class TransitTimeCalculator(tp.Module):
    """
    pipe module that calculates transit times and TTS of pmt signals
    """

    def configure(self):
        self.charge_threshold = self.get("charge_threshold")
        self.voltage_threshold = self.get("voltage_threshold")

    def process(self, blob):
        charges = blob["charges"]
        signal_mask = charges > self.charge_threshold * (
            blob["popt_prf"]["spe_charge"] + blob["popt_ped"]["mean"]
        )
        zeroed_signals = blob["waveforms"][signal_mask] - np.mean(
            blob["waveforms"][signal_mask][:, :200]
        )

        transit_times = (
            np.argmax(
                zeroed_signals * blob["v_gain"] < self.voltage_threshold, axis=1
            )
            * blob["h_int"]
            * 1e9
        )
        transit_times = transit_times[transit_times != 0]
        blob["transit_times"] = transit_times
        t, n = bin_data(transit_times, range=(0, 100), bins=200)
        blob["tt_distribution"] = (t, n)
        mean_0 = t[np.argmax(n)]
        try:
            tt_popt, _ = curve_fit(gaussian, t, n, p0=[mean_0, 2, 1e5])
        except RuntimeError:
            tt_popt = [0, 0, 0]
        blob["tt_popt"] = tt_popt
        blob["TTS"] = tt_popt[1]
        return blob


class RiseTimeCalculator(tp.Module):
    """
    rise time calculator

    Parameters
    ----------
    relative_thresholds: tuple(float)
        relative lower and upper threshold inbetween which to calculate
        rise time
    relative_charge_range: tuple(float)
        relative range of spe charge which are used for the rise time
        calculation

    """

    def configure(self):
        self.relative_thresholds = self.get("relative_thresholds")
        self.relative_charge_range = self.get("relative_charge_range")

    def process(self, blob):
        spe_charge_peak = (
            blob["popt_prf"]["spe_charge"] - blob["popt_prf"]["ped_mean"]
        )
        signals = blob["waveforms"][
            (blob["charges"] > spe_charge_peak * self.relative_charge_range[0])
            & (
                blob["charges"]
                < spe_charge_peak * self.relative_charge_range[1]
            )
        ]
        rise_times = calculate_rise_times(signals, self.relative_thresholds)
        blob["rise_time"] = np.mean(rise_times)
        return blob


class PrePulseCalculator(tp.Module):
    """
    calculates pre-pulse probability
    """

    def configure(self):
        self.time_range = self.get("time_range")

    def process(self, blob):
        max_time = blob["tt_popt"][0] + self.time_range[1]
        min_time = blob["tt_popt"][0] + self.time_range[0]
        blob["pre_max_time"] = max_time
        blob["pre_min_time"] = min_time
        transit_times = blob["transit_times"]
        n_pre_pulses = len(
            transit_times[
                (transit_times > min_time) & (transit_times < max_time)
            ]
        )
        blob["pre_pulse_prob"] = n_pre_pulses / len(transit_times)
        return blob


class PrePulseChargeCalculator(tp.Module):
    """
    calculates pre-pulse probability via their charges
    """

    def configure(self):
        self.n_sigma = self.get("n_sigma", default=5)

    def process(self, blob):
        waveforms = blob["waveforms"]
        blob["precharges_n_sigma"] = self.n_sigma
        peak_position = np.argmin(np.mean(waveforms, axis=0))
        pre_charges = calculate_charges(
            waveforms, 0, 70, peak_position - 100, peak_position - 30
        )
        pre_x, pre_y = bin_data(pre_charges, range=(-500, 3000), bins=200)
        blob["precharge_pre_charge_distribution"] = (pre_x, pre_y)
        charges = calculate_charges(
            waveforms, 0, 130, peak_position - 30, peak_position + 100
        )
        x, y = bin_data(charges, range=(-500, 3000), bins=200)
        blob["precharge_charge_distribution"] = (x, y)

        popt_pre, pcov_pre = curve_fit(
            gaussian, pre_x, pre_y, p0=[0, 100, 100000]
        )
        popt, pcov = curve_fit(gaussian, x, y, p0=[0, 100, 100000])
        blob["precharge_popt_pre"] = popt_pre
        blob["precharge_popt"] = popt

        charge_sum = np.sum(charges[charges > popt[0] + popt[1] * self.n_sigma])
        charge_sum_pre = np.sum(
            pre_charges[pre_charges > popt_pre[0] + popt_pre[1] * self.n_sigma]
        )
        pre_prob_charge = charge_sum_pre / charge_sum
        blob["pre_pulse_prob_charge"] = pre_prob_charge
        return blob


class DelayedPulseCalculator(tp.Module):
    """
    calculates delayed pulse probability
    """

    def process(self, blob):
        min_time = blob["tt_popt"][0] + 15
        blob["delayed_min_time"] = min_time
        transit_times = blob["transit_times"]
        n_delayed_pulses = len(transit_times[transit_times > min_time])
        blob["delayed_pulse_prob"] = n_delayed_pulses / len(transit_times)
        return blob


class ResultWriter(tp.Module):
    """
    writes fitted and calculated PMT parameters to text file
    """

    def configure(self):
        self.filename = self.get("filename")
        self.outfile = open(self.filename, "w")
        self.outfile.write(
            f"pmt_id "
            f"hv "
            f"nphe peak_to_valley TT[ns] TTS[ns] "
            f"pre_pulse_prob delayed_pulse_prob "
            f"pre_pulse_prob_charge spe_res rise_time\n"
        )

    def process(self, blob):
        self.outfile.write(
            f"{blob['pmt_id']} "
            f"{blob['nominal_hv']} "
            f"{blob['popt_prf']['nphe']} "
            f"{blob['peak_to_valley']} "
            f"{blob['tt_popt'][0]} "
            f"{blob['TTS']} "
            f"{blob['pre_pulse_prob']} "
            f"{blob['delayed_pulse_prob']} "
            f"{blob['pre_pulse_prob_charge']} "
            f"{blob['popt_prf']['spe_sigma'] / blob['popt_prf']['spe_charge']} "
            f"{blob['rise_time']}\n"
        )
        self.outfile.flush()
        return blob

    def finish(self):
        self.outfile.close()


class ResultPlotter(tp.Module):
    """
    pipe module that plots and saves figures of:
    - charge distribution,
    - PMT response fit,
    - transit time distribution,
    - pre pulse charges
    if available
    """

    def configure(self):
        self.file_path = self.get("file_path")

    def process(self, blob):
        fig, ax = plt.subplots(2, 2, figsize=(16, 12))
        ax = ax.flatten()
        fig.suptitle(
            f"PMT: {blob['pmt_id']};   "
            f"nominal gain: {blob['nominal_gain']};   "
            f"nominal HV: {blob['nominal_hv']}"
        )
        ax[0].set_title("charge distribution")
        ax[1].set_title("TT distribution")
        if "charge_distribution" in blob:
            x, y = blob["charge_distribution"]
            ax[0].semilogy(x, y)
            ax[0].set_ylim(0.1, 1e5)
            ax[0].set_xlabel("gain")
            ax[0].set_ylabel("counts")
        if "popt_prf" in blob:
            func = blob["fit_function"]
            popt_prf = blob["popt_prf"]
            ax[0].semilogy(x, func(x, **popt_prf))
            gain = blob["popt_prf"]["spe_charge"]
            nphe = blob["popt_prf"]["nphe"]
            ax[0].text(1e7, 10000, f"gain: {round(gain)}")
            ax[0].text(1e7, 3000, f"nphe: {round(nphe, 3)}")
            if "peak_to_valley" in blob:
                ax[0].text(
                    1e7,
                    1000,
                    f"peak to valley: {round(blob['peak_to_valley'], 3)}",
                )
        if "tt_distribution" in blob:
            x, y = blob["tt_distribution"]
            ax[1].semilogy(x, y)
            ax[1].set_ylim(0.1, 1e4)
            ax[1].set_xlabel("transit time [ns]")
            ax[1].set_ylabel("counts")
        if "tt_popt" in blob:
            ax[1].semilogy(x, gaussian(x, *blob["tt_popt"]))
            ax[1].text(0, 3000, f"TT: {round(blob['tt_popt'][0], 3)} ns")
            ax[1].text(0, 1000, f"TTS: {round(blob['tt_popt'][1], 3)} ns")
        if "pre_pulse_prob" in blob:
            pre_pulse_prob = blob["pre_pulse_prob"]
            ax[1].text(70, 3000, f"pre: {round(pre_pulse_prob, 3)}")
            ax[1].axvline(blob["pre_max_time"], color="black", lw=1)
        if "delayed_pulse_prob" in blob:
            delayed_pulse_prob = blob["delayed_pulse_prob"]
            ax[1].text(70, 1000, f"delayed: {round(delayed_pulse_prob, 3)}")
            ax[1].axvline(blob["delayed_min_time"], color="black", lw=1)
        if "pre_pulse_prob_charge" in blob:
            x, y = blob["precharge_charge_distribution"]
            x_pre, y_pre = blob["precharge_pre_charge_distribution"]
            popt = blob["precharge_popt"]
            popt_pre = blob["precharge_popt_pre"]
            n_sigma = blob["precharges_n_sigma"]
            ax[2].semilogy(x, y)
            ax[2].plot(x, gaussian(x, *popt))
            ax[2].set_ylim(0.1, 1e5)
            ax[2].axvline(
                popt[0] + popt[1] * n_sigma,
                color="black",
                label=f"mean(ped) + {n_sigma} * sigma(ped)",
                lw=1,
            )
            ax[2].set_xlabel("charge [A.U.]")
            ax[2].set_ylabel("counts")
            ax[2].legend()

            ax[3].semilogy(x_pre, y_pre)
            ax[3].plot(x_pre, gaussian(x_pre, *popt_pre))
            ax[3].set_ylim(0.1, 1e5)
            ax[3].axvline(
                popt_pre[0] + popt_pre[1] * n_sigma,
                color="black",
                label=f"mean(ped) + {n_sigma} * sigma(ped)",
                lw=1,
            )
            ax[3].set_xlabel("charge [A.U.]")
            ax[3].set_ylabel("counts")
            ax[3].text(
                1000,
                1000,
                f"pre_charge: {round(blob['pre_pulse_prob_charge'], 3)}",
            )
            ax[3].legend()

        fig.savefig(
            f"{self.file_path}/{blob['pmt_id']}_{blob['nominal_gain']}.png",
            bbox_inches="tight",
        )

        return blob
