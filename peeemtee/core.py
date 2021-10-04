#!/usr/bin/env python
import numpy as np
import h5py


class WavesetReader:
    def __init__(self, filename):
        self.filename = filename
        self._wavesets = None

    @property
    def wavesets(self):
        if self._wavesets is None:
            with h5py.File(self.filename, "r") as f:
                self._wavesets = list(f.keys())
        return self._wavesets

    def __getitem__(self, key):
        with h5py.File(self.filename, "r") as f:
            raw_waveforms = f[f"{key}/waveforms"][:]
            v_gain = f[f"{key}/waveform_info/v_gain"][()]
            h_int = f[f"{key}/waveform_info/h_int"][()]
        return Waveset(raw_waveforms, v_gain, h_int)


class Waveset:
    def __init__(self, raw_waveforms, v_gain, h_int):
        self.raw_waveforms = raw_waveforms
        self.v_gain = v_gain
        self.h_int = h_int
        self._waveforms = None

    @property
    def waveforms(self):
        if self._waveforms is None:
            self._waveforms = self.raw_waveforms * self.v_gain
        return self._waveforms

    def zeroed_waveforms(self, baseline_min, baseline_max):
        return (
            self.waveforms.T
            - np.mean(self.waveforms[:, baseline_min:baseline_max], axis=1)
        ).T
