#!/usr/bin/env python

import os
import numpy as np
import codecs
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from collections import defaultdict
import thepipe as tp
import h5py
from .tools import (
    gaussian,
    calculate_charges,
    bin_data,
    calculate_rise_times,
    read_spectral_scan,
    read_datetime,
    convert_to_secs,
    choose_ref,
    peak_finder,
    remove_double_peaks,
    peaks_with_signal,
)
from .pmt_resp_func import ChargeHistFitter
from .constants import hama_phd_qe
from .core import WavesetReader


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
        blob["spe_resolution"] = (
            fitter.popt_prf["spe_sigma"] / fitter.popt_prf["spe_charge"]
        )
        blob["nphe"] = fitter.popt_prf["nphe"]
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
        blob["transit_time"] = tt_popt[0]
        return blob


class RiseTimeCalculator(tp.Module):
    """
    rise time calculator

    Parameters
    ----------
    relative_thresholds: tuple(float)
        relative lower and upper t
        blob["zeroed_signals"] = zeroed_signalshreshold inbetween which to calculate
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


class MeanSpeAmplitudeCalculator(tp.Module):
    """
    mean spe amplitude calculator

    Parameters
    ----------
    relative_charge_range: tuple(float) (0.8, 1.2)
        relative range of spe charge which are used for the mean amplitude
        calculation

    """

    def configure(self):
        self.relative_charge_range = self.get("relative_charge_range")

    def process(self, blob):
        spe_charge_peak = (
            blob["popt_prf"]["spe_charge"] - blob["popt_prf"]["ped_mean"]
        )
        spe_mask = (
            blob["charges"] > (spe_charge_peak * self.relative_charge_range[0])
        ) & (
            blob["charges"] < (spe_charge_peak * self.relative_charge_range[1])
        )
        spes = blob["waveforms"][spe_mask] * blob["v_gain"]
        zeroed_spes = (spes.T - np.mean(spes[:, :150], axis=1)).T
        spe_amplitudes = np.min(zeroed_spes, axis=1)
        blob["mean_spe_amplitude"] = np.mean(spe_amplitudes)
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
        self.results = defaultdict(list)
        self.parameters_to_write = [
            "pmt_id",
            "nominal_hv",
            "nphe",
            "peak_to_valley",
            "transit_time",
            "TTS",
            "pre_pulse_prob",
            "delayed_pulse_prob",
            "pre_pulse_prob_charge",
            "spe_resolution",
            "rise_time",
            "mean_spe_amplitude",
        ]

    def process(self, blob):
        for parameter in self.parameters_to_write:
            self.results[parameter].append(blob[parameter])
        return blob

    def finish(self):
        outfile = open(self.filename, "w")
        for parameter in self.parameters_to_write:
            outfile.write(f"{parameter} ")
        outfile.write("\n")
        for i in range(len(self.results[self.parameters_to_write[0]])):
            for parameter in self.parameters_to_write:
                outfile.write(f"{self.results[parameter][i]} ")
            outfile.write("\n")
        outfile.close()


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
        self.results = defaultdict(list)
        self.parameters_for_plots = [
            "pmt_id",
            "nominal_gain",
            "nominal_hv",
            "charge_distribution",
            "popt_prf",
            "fit_function",
            "peak_to_valley",
            "tt_distribution",
            "tt_popt",
            "pre_pulse_prob",
            "pre_max_time",
            "delayed_pulse_prob",
            "delayed_min_time",
            "pre_pulse_prob_charge",
            "precharge_charge_distribution",
            "precharge_pre_charge_distribution",
            "precharge_popt",
            "precharge_popt_pre",
            "precharges_n_sigma",
            "pre_pulse_prob_charge",
        ]

    def process(self, blob):
        for parameter in self.parameters_for_plots:
            self.results[parameter].append(blob[parameter])
        return blob

    def finish(self):
        os.mkdir(self.file_path)
        for i in range(len(self.results[self.parameters_for_plots[0]])):
            fig, ax = plt.subplots(2, 2, figsize=(16, 12))
            ax = ax.flatten()
            fig.suptitle(
                f"PMT: {self.results['pmt_id'][i]};   "
                f"nominal gain: {self.results['nominal_gain'][i]};   "
                f"nominal HV: {self.results['nominal_hv'][i]}"
            )
            ax[0].set_title("charge distribution")
            ax[1].set_title("TT distribution")
            if "charge_distribution" in self.results:
                x, y = self.results["charge_distribution"][i]
                ax[0].semilogy(x, y)
                ax[0].set_ylim(0.1, 1e5)
                ax[0].set_xlabel("gain")
                ax[0].set_ylabel("counts")
            if "popt_prf" in self.results:
                func = self.results["fit_function"][i]
                popt_prf = self.results["popt_prf"][i]
                ax[0].semilogy(x, func(x, **popt_prf))
                gain = self.results["popt_prf"][i]["spe_charge"]
                nphe = self.results["popt_prf"][i]["nphe"]
                ax[0].text(1e7, 10000, f"gain: {round(gain)}")
                ax[0].text(1e7, 3000, f"nphe: {round(nphe, 3)}")
                if "peak_to_valley" in self.results:
                    ax[0].text(
                        1e7,
                        1000,
                        f"peak to valley: {round(self.results['peak_to_valley'][i], 3)}",
                    )
            if "tt_distribution" in self.results:
                x, y = self.results["tt_distribution"][i]
                ax[1].semilogy(x, y)
                ax[1].set_ylim(0.1, 1e4)
                ax[1].set_xlabel("transit time [ns]")
                ax[1].set_ylabel("counts")
            if "tt_popt" in self.results:
                tt_popt = self.results["tt_popt"][i]
                ax[1].semilogy(x, gaussian(x, *tt_popt))
                ax[1].text(0, 3000, f"TT: {round(tt_popt[0], 3)} ns")
                ax[1].text(0, 1000, f"TTS: {round(tt_popt[1], 3)} ns")
            if "pre_pulse_prob" in self.results:
                pre_pulse_prob = self.results["pre_pulse_prob"][i]
                ax[1].text(70, 3000, f"pre: {round(pre_pulse_prob, 3)}")
                ax[1].axvline(
                    self.results["pre_max_time"][i], color="black", lw=1
                )
            if "delayed_pulse_prob" in self.results:
                delayed_pulse_prob = self.results["delayed_pulse_prob"][i]
                ax[1].text(70, 1000, f"delayed: {round(delayed_pulse_prob, 3)}")
                ax[1].axvline(
                    self.results["delayed_min_time"][i], color="black", lw=1
                )
            if "pre_pulse_prob_charge" in self.results:
                x, y = self.results["precharge_charge_distribution"][i]
                x_pre, y_pre = self.results[
                    "precharge_pre_charge_distribution"
                ][i]
                popt = self.results["precharge_popt"][i]
                popt_pre = self.results["precharge_popt_pre"][i]
                n_sigma = self.results["precharges_n_sigma"][i]
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
                    f"pre_charge: {round(self.results['pre_pulse_prob_charge'][i], 3)}",
                )
                ax[3].legend()

            fig.savefig(
                f"{self.file_path}/{self.results['pmt_id'][i]}_{self.results['nominal_gain'][i]}.png",
                bbox_inches="tight",
            )
            plt.close(fig)


class AfterpulseFileReader(tp.Module):
    """
    pipe module that reads h5py afterpulse files and writes waveforms
    and waveform info of reference and main measurement into the blob
    """

    def process(self, blob):
        filename = blob["filename"]
        blob["pmt_id"] = os.path.basename(filename).split("_")[-1].split(".")[0]
        self.cprint(f"Reading file: {filename}")
        reader = WavesetReader(filename)
        for k in reader.wavesets:
            if "ref" in k:
                blob["waveforms_ref"] = reader[k].waveforms
                blob["h_int_ref"] = reader[k].h_int
            else:
                blob["waveforms"] = reader[k].waveforms
                blob["h_int"] = reader[k].h_int
        return blob


class AfterpulseNpheCalculator(tp.Module):
    """
    pipe module that calculates the mean number of photoelectrons of
    the afterpulse data - used for correction of the afterpulse probability later

    Parameters
    ----------
    ped_sig_range_ref: tuple(int)
        pedestal and signal integration range for reference measurement (spe regime)
    ap_integration_window_half_width: int
        half width of pedestal and signal integration range of afterpulse data
    n_gaussians: int
        number of gaussians used for afterpulse charge spectrum fit
    fit_mod_ref: str
        fit mod used for PMT response fit to reference data
    """

    def configure(self):
        self.ped_sig_range_ref = self.get(
            "ped_sig_range", default=(0, 200, 200, 400)
        )
        self.ap_integration_window_half_width = self.get(
            "ap_integration_window_half_width", default=25
        )
        self.n_gaussians = self.get("n_gaussians", default=25)
        self.fit_mod_ref = self.get("fit_mod_ref", default=False)

    def process(self, blob):
        blob["fit_info"] = {}
        charges_ref = calculate_charges(
            blob["waveforms_ref"], *self.ped_sig_range_ref
        )
        x_ref, y_ref = bin_data(charges_ref, bins=200)
        fitter_ref = ChargeHistFitter()
        fitter_ref.pre_fit(x_ref, y_ref, print_level=0)
        fitter_ref.fit_pmt_resp_func(
            x_ref, y_ref, print_level=0, mod=self.fit_mod_ref
        )
        blob["fit_info"]["x_ref"] = x_ref
        blob["fit_info"]["y_ref"] = y_ref
        blob["fit_info"]["prf_values_ref"] = fitter_ref.opt_prf_values
        blob["fit_info"]["ped_values_ref"] = fitter_ref.opt_ped_values
        blob["fit_info"]["spe_values_ref"] = fitter_ref.opt_spe_values

        time_bin_ratio = blob["h_int_ref"] / blob["h_int"]

        waveforms = blob["waveforms"]
        sig_pos = np.argmin(np.mean(waveforms, axis=0))
        blob["sig_pos"] = sig_pos
        ped_sig_range = [
            sig_pos - 3 * self.ap_integration_window_half_width,
            sig_pos - self.ap_integration_window_half_width,
            sig_pos - self.ap_integration_window_half_width,
            sig_pos + self.ap_integration_window_half_width,
        ]
        charges = calculate_charges(waveforms, *ped_sig_range)
        x, y = bin_data(charges, bins=200)

        fitter = ChargeHistFitter()
        fitter.fix_ped_spe(
            fitter_ref.popt_prf["ped_mean"] * time_bin_ratio,
            fitter_ref.popt_prf["ped_sigma"] * time_bin_ratio,
            fitter_ref.popt_prf["spe_charge"] * time_bin_ratio,
            fitter_ref.popt_prf["spe_sigma"] * time_bin_ratio,
        )
        fitter.pre_fit(x, y, print_level=0)
        fitter.fit_pmt_resp_func(
            x,
            y,
            n_gaussians=self.n_gaussians,
            strong_limits=False,
            print_level=0,
        )
        blob["fit_info"]["x"] = x
        blob["fit_info"]["y"] = y
        blob["fit_info"]["fit_values"] = fitter.opt_prf_values
        blob["ap_nphe"] = fitter.popt_prf["nphe"]
        return blob


class AfterpulseCalculator(tp.Module):
    """
    pipe module that calculates afterpulse probability

    Parameters
    ----------
    threshold: float
        threshold for peak finder
    rel_ap_time_window: tuple(float)
        time window relative to main signal in which afterpulses are counted
    double_peak_distance: int
        max distance between peaks to count as double peak
    ap_integration_window_half_width: int
        half width of pedestal and signal integration range of afterpulse data
    """

    def configure(self):
        self.threshold = self.get("threshold")
        self.rel_ap_range = self.get("rel_ap_range", default=(0.1, 12))
        self.double_peak_distance = self.get("double_peak_distance", default=20)
        self.ap_integration_window_half_width = self.get(
            "ap_integration_window_half_width", default=25
        )

    def process(self, blob):
        S_TO_US = 1e6
        waveforms = blob["waveforms"]
        waveforms = (waveforms.T - np.mean(waveforms[:, :100], axis=1)).T
        peaks = peak_finder(waveforms, self.threshold)
        peaks = remove_double_peaks(peaks, distance=self.double_peak_distance)
        sig_pos = blob["sig_pos"]
        peaks = peaks_with_signal(
            peaks,
            (
                sig_pos - self.ap_integration_window_half_width,
                sig_pos + self.ap_integration_window_half_width,
            ),
        )
        flat_peaks = []
        for peak in peaks:
            for p in peak:
                flat_peaks.append(p)
        ap_dist_us = np.array(flat_peaks) * blob["h_int"] * S_TO_US
        sig_pos_us = sig_pos * blob["h_int"] * S_TO_US
        blob["ap_dist_us"] = ap_dist_us
        dr_window_len = sig_pos_us - 0.05
        ap_window_len = self.rel_ap_range[1] - self.rel_ap_range[0]
        n_dr_hits = (
            np.sum(ap_dist_us < (sig_pos_us - 0.05))
            * ap_window_len
            / dr_window_len
        )
        blob["n_dr_hits"] = n_dr_hits
        n_afterpulses = np.sum(
            (ap_dist_us > (sig_pos_us + self.rel_ap_range[0]))
            & (ap_dist_us < (sig_pos_us + self.rel_ap_range[1]))
        )
        blob["n_afterpulses"] = n_afterpulses
        afterpulse_prob = (
            (n_afterpulses - n_dr_hits) / len(peaks) / blob["ap_nphe"]
        )
        blob["afterpulse_prob"] = afterpulse_prob
        return blob


class AfterpulseResultWriter(tp.Module):
    def configure(self):
        self.filename = self.get("filename")

    def process(self, blob):
        outfile = open(self.filename, "a")
        outfile.write(f"{blob['pmt_id']} {blob['afterpulse_prob']}\n")
        outfile.close()
        return blob


class AfterpulsePlotter(tp.Module):
    def configure(self):
        self.file_path = self.get("file_path")

    def process(self, blob):
        fig, ax = plt.subplots(2, 2, figsize=(12, 5))
        ax = ax.flatten()
        ax[0].semilogy(blob["fit_info"]["x_ref"], blob["fit_info"]["y_ref"])
        ax[0].semilogy(
            blob["fit_info"]["x_ref"], blob["fit_info"]["prf_values_ref"]
        )
        ax[0].set_ylim(0.1, 1e4)
        ax[0].set_xlabel("charge [A.U.]")
        ax[0].set_ylabel("counts")
        ax[1].semilogy(blob["fit_info"]["x"], blob["fit_info"]["y"])
        ax[1].semilogy(blob["fit_info"]["x"], blob["fit_info"]["fit_values"])
        ax[1].set_ylim(0.1, 1e4)
        ax[1].set_xlabel("charge [A.U.]")
        ax[1].set_ylabel("counts")
        ax[1].text(0, 1e3, f"nphe: {round(blob['ap_nphe'], 2)}")
        ax[2].hist(blob["ap_dist_us"], bins=200, log=True)
        ax[2].set_xlabel("time [us]")
        ax[2].set_ylabel("counts")
        ax[2].text(
            5,
            2e3,
            f"afterpulse probability: {round(blob['afterpulse_prob'], 2)}",
        )
        fig.savefig(f"{self.file_path}/{blob['pmt_id']}_afterpulse.png")
        plt.close(fig)
        return blob
