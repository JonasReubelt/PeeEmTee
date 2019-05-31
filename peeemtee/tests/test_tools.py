import numpy as np
from unittest import TestCase
from peeemtee.tools import calculate_charges


class TestTools(TestCase):

    def test_calculate_charges(self):
        waveforms = np.array([[0, 1, -45, -53, 0, -1],
                              [-5, 3, -145, -253, 3, -5],
                              [0, 0, -44, -12, -1, 1]])
        self.assertListEqual(list(calculate_charges(waveforms, 0, 2, 2, 4)),
                             [99, 396, 56])
