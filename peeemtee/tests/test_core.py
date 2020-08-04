import numpy as np
from unittest import TestCase
from peeemtee.core import WavesetReader, Waveset


class TestTools(TestCase):
    def test_WavesetReader(self):
        reader = WavesetReader("peeemtee/tests/samples/waveform_data_dummy.h5")
        wavesets = reader.wavesets
        hvs = ["900", "950", "1000", "1050", "1100"]
        assert sorted(wavesets) == sorted(hvs)
        b = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
        for i, hv in enumerate(hvs):
            waveset = reader[int(hv)]
            a = np.arange(1, 10, dtype=np.int8).reshape(3, 3) + i * 10
            np.testing.assert_array_equal(waveset.raw_waveforms, a)
            v_gain = waveset.v_gain
            assert v_gain == i * 0.1
            np.testing.assert_array_equal(waveset.waveforms, a * v_gain)
            assert waveset.h_int == i * 10
            np.testing.assert_array_almost_equal(
                waveset.zeroed_waveforms(0, 1), b * v_gain
            )
