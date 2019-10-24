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
        self.seed = 0
        self.noise_sigma = 0.13

        # Get precise simulated experiment time
        self.time = np.linspace(0, self.signal_length, self.signal_length * self.acquisition_rate)

        self.clean_data = self.clean_data.assign(time=self.time)

        # Create clean square wave with period 2 pi and with its first rising edge at pi
        # self.clean_data = self.clean_data.assign(green=(sig.square(self.clean_data.time + np.pi) + 1) / 2)
        self.clean_data = self.clean_data.assign(green=((sig.square(2 * np.pi * 0.5 * (self.clean_data.time - 1))) + 1) / 2)
        self.clean_data = self.clean_data.assign(uv=self.clean_data.green)
        self.clean_data = self.clean_data.assign(orange=self.clean_data.green)

        self.clean_data.loc[self.clean_data.time > 4, "orange"] = 0
        self.clean_data.loc[(self.clean_data.time < 4) | (self.clean_data.time > 6), "uv"] = 0
        self.clean_data.loc[(self.clean_data.time < 3) | (self.clean_data.time > 4), "green"] = 0
        # self.clean_data.loc[(self.clean_data.time < 3) | (self.clean_data.time > 6), "orange"] = 0

        repeats = 5
        for _ in range(repeats):
            try:
                # noinspection PyUnboundLocalVariable
                tmp_data = tmp_data.append(self.clean_data)
            except UnboundLocalError:
                tmp_data = self.clean_data

        # noinspection PyUnboundLocalVariable
        self.clean_data = tmp_data.reset_index(drop=True)
        #     self.single_plug_data = self.single_plug_data.append(self.single_plug_data)
        # self.single_plug_data = self.single_plug_data.reset_index(drop=True)

        self.clean_data = self.clean_data.assign(time=np.linspace(0, self.signal_length * repeats, self.signal_length * repeats * self.acquisition_rate))

        # End of cycle
        self.clean_data.loc[(self.clean_data.time > 15) & (self.clean_data.time < 20), "orange"] = 0
        self.clean_data.loc[(self.clean_data.time > 15) & (self.clean_data.time < 20), "green"] = 0

        self.clean_data.loc[(self.clean_data.time > 15) & (self.clean_data.time < 16), "uv"] = 1
        self.clean_data.loc[(self.clean_data.time > 17) & (self.clean_data.time < 18), "uv"] = 1

        self.clean_data = self.clean_data.assign(green=self.clean_data.green * 0.9)
        self.clean_data = self.clean_data.assign(orange=self.clean_data.orange * 0.8)

        # Filter the clean signal with a mean filter to get slightly rounded edges
        self.noisy_data = self.noisy_data.assign(time=self.clean_data.time)
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
        # self.single_plug_data = pd.DataFrame({"start_time": [1.0, 3.0, 5.0],
        #                                       # "center_time": [1.5, 3.5, 5.5],
        #                                       "end_time": [2.0, 4.0, 6.0],
        #                                       # "bc_peak_max": [1.0, 0.0, 0.0],
        #                                       "barcode_peak_median": [0.0, 0.0, 1.0],
        #                                       # "bc_peak_mean": [1.0, 0.0, 0.0],
        #                                       # "control_peak_max": [0.0, 0.8, 0.8],
        #                                       "control_peak_median": [0.8, 0.8, 0.0],
        #                                       # "control_peak_mean": [0.0, 0.8, 0.8],
        #                                       # "cell_peak_max": [0.0, 0.9, 0.0],
        #                                       "readout_peak_median": [0.0, 0.9, 0.0],
        #                                       # "cell_peak_mean": [0.0, 0.9, 0.0],
        #                                       "barcode": [False, False, True]})

        self.cycle_data = pd.DataFrame({"start_time": [1.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 17.0, 19.0, 22.0, 24.0, 26.0, 29.0, 31.0, 33.0],
                                        "end_time": [2.0, 4.0, 6.0, 9.0, 11.0, 13.0, 16.0, 18.0, 20.0, 23.0, 25.0, 27.0, 30.0, 32.0, 34.0],
                                        "barcode_peak_median": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                                        "control_peak_median": [0.8, 0.8, 0.0, 0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.8, 0.8, 0.0, 0.8, 0.8, 0.0],
                                        "readout_peak_median": [0.0, 0.9, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.9, 0.0],
                                        "barcode": [False, False, True, False, False, True, True, True, True, False, False, True, False, False, True]})
        self.sample_data = self.cycle_data.assign(cycle=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], sample=[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1])

    @unittest.skip
    def test_plot_test_data(self):
        test_data_fig, test_data_ax = plt.subplots(1, 2, figsize=(40, 10))
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

    # @unittest.skip
    def test_plot_detected_data(self):
        """
        Tests plotting of the plug data together with the pmt data
        """
        with unittest.mock.patch.object(target=pmt.PmtData, attribute="read_txt", new=lambda _: self.noisy_data):
            # noinspection PyTypeChecker
            plug_data = plug.PlugData(pmt_data=pmt.PmtData(input_file=pl.Path("MOCK")), plug_sequence=None, channel_map=None, peak_min_distance=0.03)

        plug_data_fig, plug_data_ax = plt.subplots(figsize=(40, 10))
        plug_data_ax = plug_data.plot_plug_pmt_data(axes=plug_data_ax)

        plug_data_fig.tight_layout()
        plug_data_fig.show()

        self.assertTrue(True)

    # noinspection DuplicatedCode
    def test_plug_detect_clean_data(self):
        """
        Tests detecting simple plugs from clean data
        """
        with unittest.mock.patch.object(target=pmt.PmtData, attribute="read_txt", new=lambda _: self.clean_data):
            # noinspection PyTypeChecker
            plug_data = plug.PlugData(pmt_data=pmt.PmtData(input_file=pl.Path("MOCK")), plug_sequence=None, channel_map=None, peak_min_distance=0.03)

        pd_test.assert_frame_equal(self.cycle_data.round(), plug_data.plug_df.round())

    # noinspection DuplicatedCode
    def test_plug_detect_noisy_data(self):
        """
        Tests detecting plugs with a large amount of noise
        """
        with unittest.mock.patch.object(target=pmt.PmtData, attribute="read_txt", new=lambda _: self.noisy_data):
            # noinspection PyTypeChecker
            plug_data = plug.PlugData(pmt_data=pmt.PmtData(input_file=pl.Path("MOCK")), plug_sequence=None, channel_map=None, peak_min_distance=0.03)

        pd_test.assert_frame_equal(self.cycle_data.round(), plug_data.plug_df.round())


if __name__ == '__main__':
    unittest.main()
