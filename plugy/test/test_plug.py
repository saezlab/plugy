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
import pandas.testing as pd_test
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

        self.signal_length = 7
        self.acquisition_rate = 300
        self.filter_size = self.acquisition_rate / 6
        self.seed = 1234
        self.noise_sigma = 0.15

        # Get precise simulated experiment time
        self.time = np.linspace(0, self.signal_length, self.signal_length * self.acquisition_rate)

        self.clean_data = self.clean_data.assign(time=self.time)
        self.noisy_data = self.noisy_data.assign(time=self.time)

        # Create clean square wave with period 2 pi and with its first rising edge at pi
        # self.clean_data = self.clean_data.assign(green=(sig.square(self.clean_data.time + np.pi) + 1) / 2)
        self.clean_data = self.clean_data.assign(green=((sig.square(2 * np.pi * 0.5 * (self.clean_data.time - 1))) + 1) / 2)
        self.clean_data = self.clean_data.assign(uv=self.clean_data.green)
        self.clean_data = self.clean_data.assign(orange=self.clean_data.green)

        self.clean_data.loc[self.clean_data.time > 2, "uv"] = 0
        self.clean_data.loc[(self.clean_data.time < 3) | (self.clean_data.time > 4), "green"] = 0
        self.clean_data.loc[(self.clean_data.time < 3) | (self.clean_data.time > 6), "orange" ] = 0

        self.clean_data = self.clean_data.assign(green=self.clean_data.green * 0.9)
        self.clean_data = self.clean_data.assign(orange=self.clean_data.orange * 0.8)

        # Filter the clean signal with a mean filter to get slightly rounded edges
        self.noisy_data = self.noisy_data.assign(green=fil.convolve1d(input=self.clean_data.green, weights=np.array(np.repeat(1, self.filter_size))) / self.filter_size)
        self.noisy_data = self.noisy_data.assign(uv=fil.convolve1d(input=self.clean_data.uv, weights=np.array(np.repeat(1, self.filter_size))) / self.filter_size)
        self.noisy_data = self.noisy_data.assign(orange=fil.convolve1d(input=self.clean_data.orange, weights=np.array(np.repeat(1, self.filter_size))) / self.filter_size)

        # Add gaussian noise to the clean square wave
        np.random.seed(self.seed)
        self.noisy_data = self.noisy_data.assign(green=self.noisy_data.green + np.random.normal(scale=self.noise_sigma, size=len(self.noisy_data.green)))
        np.random.seed(self.seed)
        self.noisy_data = self.noisy_data.assign(uv=self.noisy_data.uv + np.random.normal(scale=self.noise_sigma, size=len(self.noisy_data.uv)))
        np.random.seed(self.seed)
        self.noisy_data = self.noisy_data.assign(orange=self.noisy_data.orange + np.random.normal(scale=self.noise_sigma, size=len(self.noisy_data.orange)))

        # Generate ground truth DataFrame
        self.plug_data = pd.DataFrame({"start_time": [1.0, 3.0, 5.0],
                                       "end_time": [2.0, 4.0, 5.0],
                                       "bc_peak_max": [1.0, 0.0, 0.0],
                                       "bc_peak_median": [1.0, 0.0, 0.0],
                                       "bc_peak_mean": [1.0, 0.0, 0.0],
                                       "control_peak_max": [0.0, 0.8, 0.8],
                                       "control_peak_median": [0.0, 0.8, 0.8],
                                       "control_peak_mean": [0.0, 0.8, 0.8],
                                       "cell_peak_max": [0.0, 0.9, 0.0],
                                       "cell_peak_median": [0.0, 0.9, 0.0],
                                       "cell_peak_mean": [0.0, 0.9, 0.0],
                                       "barcode": [True, False, False]})

    # @unittest.skip
    def test_plot_test_data(self):
        test_data_fig, test_data_ax = plt.subplots(1, 2, figsize=(20, 10))
        test_data_ax[0].plot(self.clean_data.time, self.clean_data.green, color="green")
        test_data_ax[0].plot(self.clean_data.time, self.clean_data.uv, color="blue")
        test_data_ax[0].plot(self.clean_data.time, self.clean_data.orange, color="orange")

        test_data_ax[1].plot(self.noisy_data.time, self.noisy_data.green, color="green")
        test_data_ax[1].plot(self.noisy_data.time, self.noisy_data.uv, color="blue")
        test_data_ax[1].plot(self.noisy_data.time, self.noisy_data.orange, color="orange")
        for i in range(2):
            test_data_ax[i].set_xlabel("Time [s]")
            test_data_ax[i].set_ylabel("PMT Output [V]")
        test_data_fig.tight_layout()
        test_data_fig.show()
        self.assertTrue(True)

    def test_plug_detect_simple_thresh(self):
        """
        Tests detecting simple plugs from clean data
        :return:
        """
        with unittest.mock.patch.object(target=pmt.PmtData, attribute="read_txt", new=lambda _: self.clean_data):
            # noinspection PyTypeChecker
            plug_data = plug.PlugData(pmt_data=pmt.PmtData(input_file=pl.Path()), plug_sequence=None, channel_map=None)

        pd_test.assert_frame_equal(self.plug_data, plug_data.plug_df)

    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
