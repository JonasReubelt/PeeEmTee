#!/usr/bin/env python

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import thepipe as tp
import h5py
from .tools import gaussian, calculate_charges, calculate_histogram_data
from .pmt_resp_func import ChargeHistFitter


def find_nominal_hv(f, nominal_gain):
    gains = []
    hvs = []
    keys = f.keys()
    for key in keys:
        gains.append(f[key]["fit_results"]["gain"].value)
        hvs.append(int(key))
    gains = np.array(gains)
    hvs = np.array(hvs)

    diff = abs(np.array(gains) - nominal_gain)
    nominal_hv = int(hvs[diff==np.min(diff)])
    return nominal_hv

class FileReader(tp.Module):
    """
    pipe module that reads h5py files and writes waveforms and waveform info
    into the blob
    """
    def configure(self):
        self.filenames = self.get("filenames")
        self.gain = self.get("gain")
        self.max_count = len(self.filenames)
        self.index = 0

    def process(self, blob):
        if self.index >= self.max_count:
            self.log.critical("All done!")
            raise StopIteration
        filename = self.filenames[self.index]
        blob["pmt_id"] = filename.split("/")[-1].split(".")[0]
        self.print(f"Current File: {filename}")
        f = h5py.File(filename, "r")
        nominal_hv = find_nominal_hv(f, self.gain)
        blob["nominal_gain"] = self.gain
        blob["nominal_hv"] = nominal_hv
        blob["waveforms"] = f[f"{nominal_hv}"]["waveforms"][:]
        blob["h_int"] = f[f"{nominal_hv}"]["waveform_info/h_int"].value
        blob["v_gain"] = f[f"{nominal_hv}"]["waveform_info/v_gain"].value
        blob["gain_norm"] = blob["v_gain"] * blob["h_int"] / 50 / 1.6022e-19
        f.close()
        self.index += 1
        return blob



    def finish(self):
        self.print(f"Read {self.index} files!")


class ChargeCalculator(tp.Module):
    """
    pipe module that calculates charges of waveforms
    """
    def configure(self):
        self.ped_range = self.get("ped_range")
        self.sig_range = self.get("sig_range")

    def process(self, blob):
        charges = calculate_charges(blob["waveforms"],
                                    self.ped_range[0],
                                    self.ped_range[1],
                                    self.sig_range[0],
                                    self.sig_range[1])
        charges = charges * blob["gain_norm"]
        blob["charges"] = charges
        x, y = calculate_histogram_data(charges, bins=200, range=(-.5e7, 3e7))
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
        fitter.fit_pmt_resp_func(x, y, mod=self.mod, print_level=0)
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
        fine_xs = np.linspace(-.5e7, 3e7, 10000)
        valley_mask = ((fine_xs>blob["popt_ped"]["mean"])
                       & (fine_xs<blob["popt_spe"]["mean"]))
        valley = np.min(fit_function(fine_xs, **blob["popt_prf"])[valley_mask])
        peak_mask = fine_xs>(blob["popt_ped"]["mean"]
                             + blob["popt_ped"]["sigma"]*3)
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
        signal_mask = (charges>self.charge_threshold
                       * (blob["popt_prf"]["spe_charge"]
                          + blob["popt_ped"]["mean"]))
        zeroed_signals = (blob["waveforms"][signal_mask]
                          - np.mean(blob["waveforms"][signal_mask][:, :200]))

        transit_times = (np.argmax(zeroed_signals
                                   * blob["v_gain"] < self.voltage_threshold,
                                   axis=1)
                         * blob["h_int"]
                         * 1e9)
        transit_times = transit_times[transit_times!=0]
        blob["transit_times"] = transit_times
        t, n = calculate_histogram_data(transit_times,
                                           range=(0, 100),
                                           bins=200)
        blob["tt_distribution"] = (t, n)
        mean_0 = t[np.argmax(n)]
        try:
            tt_popt, _ = curve_fit(gaussian, t, n, p0=[mean_0, 2, 1e5])
        except RuntimeError:
            tt_popt = [0, 0, 0]
        blob["tt_popt"] = tt_popt
        blob["TTS"] = tt_popt[1]
        return blob


class PrePulseCalculator(tp.Module):
    """
    calculates pre-pulse probability
    """
    def process(self, blob):
        max_time = blob["tt_popt"][0] - 10
        transit_times = blob["transit_times"]
        n_pre_pulses = len(transit_times[transit_times < max_time])
        blob["pre_pulse_prob"] = n_pre_pulses/len(transit_times)
        return blob


class PrePulseChargeCalculator(tp.Module):
    """
    calculates pre-pulse probability via their charges
    """
    def process(self, blob):
        waveforms = blob["waveforms"]
        peak_position = np.argmin(np.mean(waveforms, axis=0))
        pre_charges = pt.calculate_charges(waveforms, 0, 70, peak_position - 100, peak_position - 30)
        pre_x, pre_y = pt.calculate_histogram_data(pre_charges, range=(-500, 3000), bins=200)
        charges = pt.calculate_charges(waveforms, 0, 130, peak_position - 30, peak_position +100)
        x, y = pt.calculate_histogram_data(charges, range=(-500, 3000), bins=200)

        popt_pre, pcov_pre = curve_fit(pt.gaussian, pre_x, pre_y, p0=[0, 100, 100000])
        popt, pcov = curve_fit(pt.gaussian, x, y, p0=[0, 100, 100000])

        charge_sum = np.sum(charges)
        charge_sum_pre = np.sum(pre_charges)
        charge_sum_3_sig = np.sum(charges[charges >popt[0]+popt[1]*3])
        charge_sum_pre_3_sig = np.sum(pre_charges[pre_charges >popt_pre[0]+popt_pre[1]*3])
        charge_sum_5_sig = np.sum(charges[charges >popt[0]+popt[1]*5])
        charge_sum_pre_5_sig = np.sum(pre_charges[pre_charges >popt_pre[0]+popt_pre[1]*5])
        pre_prob_charge = charge_sum_pre / charge_sum
        pre_prob_charge_3_sig = charge_sum_pre_3_sig / charge_sum_3_sig
        pre_prob_charge_5_sig = charge_sum_pre_5_sig / charge_sum_5_sig
        blob["pre_pulse_prob_via_charge"] = pre_prob_charge_3_sig
        return blob

class DelayedPulseCalculator(tp.Module):
    """
    calculates delayed pulse probability
    """
    def process(self, blob):
        min_time = blob["tt_popt"][0] +15
        transit_times = blob["transit_times"]
        n_delayed_pulses = len(transit_times[transit_times > min_time])
        blob["delayed_pulse_prob"] = n_delayed_pulses/len(transit_times)
        return blob


class ResultWriter(tp.Module):
    """
    writes fitted and calculated PMT parameters to text file
    """
    def configure(self):
        self.filename = self.get("filename")
        self.outfile = open(self.filename, "w")
        self.outfile.write(f"hv@5e6 "\
                           f"nphe peak_to_valley TT[ns] TTS[ns] "\
                           f"pre_pulse_prob delayed_pulse_prob\n")

    def process(self, blob):
        self.outfile.write(f"{blob['nominal_hv']} "\
                           f"{blob['popt_prf']['nphe']} "\
                           f"{blob['peak_to_valley']} "\
                           f"{blob['tt_popt'][0]} "\
                           f"{blob['TTS']} "\
                           f"{blob['pre_pulse_prob']} "\
                           f"{blob['delayed_pulse_prob']}\n")
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
        fig.suptitle(f"PMT: {blob['pmt_id']};   "\
                     f"nominal gain: {blob['nominal_gain']};   "\
                     f"nominal HV: {blob['nominal_hv']}")
        ax[0].set_title("charge distribution")
        ax[1].set_title("TT distribution")
        if "charge_distribution" in blob:
            x, y = blob["charge_distribution"]
            ax[0].semilogy(x, y)
            ax[0].set_ylim(.1, 1e5)
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
                ax[0].text(1e7,
                           1000,
                           f"peak to valley: {round(blob['peak_to_valley'], 3)}")
        if "tt_distribution" in blob:
            x, y = blob["tt_distribution"]
            ax[1].semilogy(x, y)
            ax[1].set_ylim(.1, 1e4)
            ax[1].set_xlabel("transit time [ns]")
            ax[1].set_ylabel("counts")
        if "tt_popt" in blob:
            ax[1].semilogy(x, gaussian(x, *blob["tt_popt"]))
            ax[1].text(0, 3000, f"TT: {round(blob['tt_popt'][0], 3)} ns")
            ax[1].text(0, 1000, f"TTS: {round(blob['tt_popt'][1], 3)} ns")
        fig.savefig(f"{self.file_path}/{blob['pmt_id']}.png", bbox_inches="tight")
        return blob
