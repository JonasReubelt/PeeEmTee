import numpy as np
from unittest import TestCase
from peeemtee.core import WavesetReader, Waveset


class TestTools(TestCase):
    def test_WavesetReader(self):
        reader = WavesetReader("peeemtee/tests/samples/waveform_data_dummy.h5")
        wavesets = reader.wavesets
        hvs = [900.0, 950.0, 1000.0, 1050.0, 1100.0]
        assert sorted(wavesets) == hvs
        for i, hv in enumerate(hvs):
            waveset = reader[int(hv)]
            a = np.arange(1, 10, dtype=np.int8).reshape(3, 3) + i * 10
            np.testing.assert_array_equal(waveset.raw_waveforms, a)
            v_gain = waveset.v_gain
            assert v_gain == i * 0.1
            np.testing.assert_array_equal(waveset.waveforms, a * v_gain)
            assert waveset.h_int == i * 10
