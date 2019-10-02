"""
Author      Nicolas Peschke
Date        02.10.2019

This file is part of the `plugy` python module

Copyright
2018-2019
EMBL, Heidelberg University
File author(s): Dénes Türei (turei.denes@gmail.com)
                Nicolas Peschke
Distributed under the GPLv3 License.
See accompanying file LICENSE.txt or copy at
    http://www.gnu.org/licenses/gpl-3.0.html

"""
import logging
import unittest
import unittest.mock

import numpy as np
import pandas as pd
import scipy.signal as sig
import scipy.ndimage.filters as fil

from ..data import plug
from ..data import pmt

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%d.%m.%y %H:%M:%S')


class TestPlugData(unittest.TestCase):
    def setUp(self) -> None:
        self.pure_data = pd.DataFrame()
        self.noisy_data = pd.DataFrame()

        self.signal_length = 21
        self.acquisition_rate = 300
        self.filter_size = self.acquisition_rate / 2
        self.time = np.linspace(0, self.signal_length, self.signal_length * self.acquisition_rate)

        self.pure_data = self.pure_data.assign(time=self.time)
        self.noisy_data = self.noisy_data.assign(time=self.time)

        self.pure_data = self.pure_data.assign(green=sig.square(self.pure_data.time + np.pi) + 1)
        self.noisy_data = self.noisy_data.assign(green=fil.convolve1d(input=self.pure_data.green, weights=np.array(np.repeat(1, self.filter_size))) / self.filter_size)

    def test_plug_detect_simple_thresh(self):
        signal_length = 21  # seconds
        acquisition_rate = 300
        filter_size = acquisition_rate / 2
        x = np.linspace(0, signal_length, signal_length * acquisition_rate)

        plt.plot(x, sig.square(x + np.pi) + 1)

        y = (fil.convolve1d(sig.square(x + np.pi) + 1, weights=np.array(np.repeat(1, filter_size)))) / filter_size
        plt.plot(x, y)

        np.random.seed(1234)
        y_noise = y + np.random.normal(scale=0.25, size=len(x))
        plt.plot(x, y_noise)

        plt.xlabel('Time [s]')
        plt.ylabel('PMT Output [V]')
        plt.axis('tight')
        plt.show()
        with unittest.mock.patch.object(target=pmt.PmtData, attribute="__post_init__", new=lambda _: self.test_df):
            pass

    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
