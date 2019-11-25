#!/usr/bin/env python
import h5py


class WavesetReader:
    def __init__(self, filename):
        self.file = h5py.File(filename, "r")

    @property
    def wavesets(self):
        return [int(key) for key in self.file.keys()]

    def __getitem__(self, key):
        raw_waveforms = self.file[f"{key}/waveforms"][:]
        v_gain = self.file[f"{key}/waveform_info/v_gain"][()]
        h_int = self.file[f"{key}/waveform_info/h_int"][()]
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
