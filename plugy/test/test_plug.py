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

import pathlib as pl

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
        self.clean_data = pd.DataFrame()
        self.noisy_data = pd.DataFrame()

        self.signal_length = 21
        self.acquisition_rate = 300
        self.filter_size = self.acquisition_rate / 2

        # Get precise simulated experiment time
        self.time = np.linspace(0, self.signal_length, self.signal_length * self.acquisition_rate)

        self.clean_data = self.clean_data.assign(time=self.time)
        self.noisy_data = self.noisy_data.assign(time=self.time)

        # Create clean square wave with period 2 pi and with its first rising edge at pi
        self.clean_data = self.clean_data.assign(green=(sig.square(self.clean_data.time + np.pi) + 1) / 2)

        # Filter the clean signal with a mean filter to get slightly rounded edges
        self.noisy_data = self.noisy_data.assign(green=fil.convolve1d(input=self.clean_data.green, weights=np.array(np.repeat(1, self.filter_size))) / self.filter_size)
        np.random.seed(1234)
        # Add gaussian noise to the clean square wave
        self.noisy_data = self.noisy_data.assign(green=self.noisy_data.green + np.random.normal(scale=0.25, size=len(self.noisy_data.green)))

    # @unittest.skip
    def test_plot_test_data(self):
        plt.plot(self.clean_data.time, self.clean_data.green)
        plt.plot(self.noisy_data.time, self.noisy_data.green)

        plt.xlabel('Time [s]')
        plt.ylabel('PMT Output [V]')
        plt.axis('tight')
        plt.show()
        self.assertTrue(True)

    def test_plug_detect_simple_thresh(self):
        """
        Tests detecting simple plugs from clean data
        :return:
        """
        with unittest.mock.patch.object(target=pmt.PmtData, attribute="read_txt", new=lambda _: self.clean_data):
            # noinspection PyTypeChecker
            plug.PlugData(pmt_data=pmt.PmtData(input_file=pl.Path()), plug_sequence=None, channel_map=None)

        self.assertTrue(False)

    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
