#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# This file is part of the `plugy` python module
#
# Copyright
# 2018-2021
# EMBL & Heidelberg University
#
# Author(s): Dénes Türei (turei.denes@gmail.com)
#            Nicolas Peschke
#            Olga Ivanova
#
# Distributed under the GPLv3 License.
# See accompanying file LICENSE.txt or copy at
#     http://www.gnu.org/licenses/gpl-3.0.html
#
# Webpage: https://github.com/saezlab/plugy
#


import logging
import tempfile
import unittest
import unittest.mock

import pathlib as pl

import numpy as np
import pandas as pd
import pandas._testing as pd_test
import scipy.signal as sig
import scipy.ndimage.filters as fil

from ..data import plug
from ..data import pmt
from ..data import bd
from ..data import config

import matplotlib.pyplot as plt

logging.basicConfig(
    level = logging.DEBUG,
    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt = '%d.%m.%y %H:%M:%S',
)

# disable annoying pandas SettingWithCopy warnings
pd.options.mode.chained_assignment = None


class TestPlugData(unittest.TestCase):


    def setUp(self) -> None:

        self.clean_data = pd.DataFrame()
        self.noisy_data = pd.DataFrame()

        self.signal_length = 8
        self.acquisition_rate = 300
        self.filter_size = self.acquisition_rate / 6
        self.seed = 0
        self.noise_sigma = 0.13
        self.pseudocount = 0.00001

        # Get precise simulated experiment time
        self.time = np.linspace(
            0,
            self.signal_length,
            int(self.signal_length * self.acquisition_rate)
        )

        self.clean_data = self.clean_data.assign(time = self.time)

        # Create clean square wave with period 2 pi and
        # with its first rising edge at pi
        self.clean_data = self.clean_data.assign(
            green = (
                (sig.square(
                    2 * np.pi * 0.5 * (self.clean_data.time - 1)
                )) +
                1
            ) / 2
        )
        self.clean_data = self.clean_data.assign(
            uv = self.clean_data.green
        )
        self.clean_data = self.clean_data.assign(
            orange = self.clean_data.green
        )

        self.clean_data.loc[self.clean_data.time > 4, "orange"] = 0
        self.clean_data.loc[
            (self.clean_data.time < 4) | (self.clean_data.time > 8),
            "uv"
        ] = 0
        self.clean_data.loc[
            (self.clean_data.time < 3) | (self.clean_data.time > 4),
            "green"
        ] = 0

        repeats = 5
        for _ in range(repeats):
            try:
                # noinspection PyUnboundLocalVariable
                tmp_data = tmp_data.append(self.clean_data) # noqa: F821
            except UnboundLocalError:
                tmp_data = self.clean_data

        # noinspection PyUnboundLocalVariable
        self.clean_data = tmp_data.reset_index(drop = True)

        self.clean_data = self.clean_data.assign(
            time = np.linspace(
                0,
                self.signal_length * repeats,
                int(self.signal_length * repeats * self.acquisition_rate)
            )
        )
        self.clean_data = pd.concat(
            [self.clean_data] +
            [self.clean_data.tail(1)] *
            100
        )
        self.clean_data.iloc[-100:].time = np.linspace(
            max(self.clean_data.time),
            max(self.clean_data.time) + .33,
            100
        )
        self.clean_data.reset_index()

        # End of cycle
        self.clean_data.loc[
            (self.clean_data.time > 13) & (self.clean_data.time < 24),
            "orange"
        ] = 0
        self.clean_data.loc[
            (self.clean_data.time > 13) & (self.clean_data.time < 24),
            "green"
        ] = 0

        self.clean_data.loc[
            (self.clean_data.time > 13) & (self.clean_data.time < 14),
            "uv"
        ] = 1
        self.clean_data.loc[
            (self.clean_data.time > 15) & (self.clean_data.time < 16),
            "uv"
        ] = 1
        self.clean_data.loc[
            (self.clean_data.time > 17) & (self.clean_data.time < 18),
            "uv"
        ] = 1

        self.clean_data = self.clean_data.assign(
            green = self.clean_data.green * 0.9
        )
        self.clean_data = self.clean_data.assign(
            orange = self.clean_data.orange * 0.8
        )

        self.clean_data = self.clean_data.assign(
            green = self.clean_data.green + self.pseudocount,
            uv = self.clean_data.uv + self.pseudocount,
            orange = self.clean_data.orange + self.pseudocount
        )

        # Filter the clean signal with a mean filter
        # to get slightly rounded edges
        self.noisy_data = self.noisy_data.assign(time = self.clean_data.time)
        self.noisy_data = self.noisy_data.assign(
            green = fil.convolve1d(
                input = self.clean_data.green,
                weights = np.array(np.repeat(1, self.filter_size))
            ) /
            self.filter_size
        )
        self.noisy_data = self.noisy_data.assign(
            uv = fil.convolve1d(
                input = self.clean_data.uv,
                weights = np.array(np.repeat(1, self.filter_size))
            ) /
            self.filter_size
        )
        self.noisy_data = self.noisy_data.assign(
            orange = fil.convolve1d(
                input = self.clean_data.orange,
                weights = np.array(np.repeat(1, self.filter_size))
            ) /
            self.filter_size
        )

        # Add gaussian noise to the clean square wave
        np.random.seed(self.seed)

        self.noisy_data = self.noisy_data.assign(
            green =
                self.noisy_data.green +
                np.random.normal(
                    scale = self.noise_sigma,
                    size = len(self.noisy_data.green)
                )
        )

        np.random.seed(self.seed)

        self.noisy_data = self.noisy_data.assign(
            uv =
                self.noisy_data.uv +
                np.random.normal(
                    scale = self.noise_sigma,
                    size = len(self.noisy_data.uv)
                )
        )

        np.random.seed(self.seed)

        self.noisy_data = self.noisy_data.assign(
            orange =
                self.noisy_data.orange +
                np.random.normal(
                    scale = self.noise_sigma,
                    size = len(self.noisy_data.orange)
                )
        )

        self.cycle_data = pd.DataFrame({
            "start_time": [
                1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 21.0,
                23.0, 25.0, 27.0, 29.0, 31.0, 33.0, 35.0, 37.0, 39.0,
            ],
            "end_time":  [
                2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 22.0,
                24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0,
            ],
            "barcode_peak_median": [
                0.0, 0.0, 1.0, 1.0,  0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
            ],
            "control_peak_median": [
                0.8, 0.8, 0.0, 0.0, 0.8, 0.8, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.8, 0.8, 0.0, 0.0, 0.8, 0.8, 0.0, 0.0,
            ],
            "readout_peak_median": [
                0.0, 0.9, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0,
            ],
            "barcode": [
                False, False, True, True, False, False,
                True, True, True, True, True, False, False,
                True, True, False, False, True, True,
            ]
        })

        self.cycle_data = self.cycle_data.assign(
            barcode_peak_median =
                self.cycle_data.barcode_peak_median +
                self.pseudocount,
            control_peak_median =
                self.cycle_data.control_peak_median +
                self.pseudocount,
            readout_peak_median =
                self.cycle_data.readout_peak_median +
                self.pseudocount
        )

        self.normalized_cycle_data = self.cycle_data
        self.normalized_cycle_data = self.normalized_cycle_data.assign(
            readout_per_control =
                self.normalized_cycle_data.readout_peak_median /
                self.normalized_cycle_data.control_peak_median
        )

        self.sample_data = self.cycle_data.assign(
            cycle_nr = [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1, 1, 1, 1,
            ],
            sample_nr = [
                0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 1, 1, 1, 1, 1,
            ],
        )
        self.sample_data = self.sample_data.loc[
            self.sample_data.barcode == False
        ]
        self.sample_data = self.sample_data.drop(columns = "barcode")

        self.sample_data = self.sample_data.assign(
            readout_peak_z_score = [-1.,  1., -1.,  1., -1.,  1., -1.,  1.]
        )

        self.sample_data = self.sample_data[
            [
                "start_time",
                "end_time",
                "barcode_peak_median",
                "control_peak_median",
                "readout_peak_median",
                "readout_peak_z_score",
                "cycle_nr",
                "sample_nr",
            ]
        ]

        self.test_gen_map_content = (
            "9:CELLS\n10:SUBSTRATE\n11:FS\n12:FS\n"
            "13:Drug 1\n14:Drug 2\n15:Drug 3\n23:BCM\n24:BCM"
        )

        with tempfile.NamedTemporaryFile(mode = "w+t", suffix = ".txt") as \
            self.channel_file:

            self.channel_file.write(self.test_gen_map_content)
            self.channel_file.seek(0)
            self.channel_map = bd.ChannelMap(pl.Path(self.channel_file.name))

        self.plug_sequence = bd.PlugSequence(
            (
                bd.Sample(1, 2, "Drug 1 + Drug 2", [9, 10, 13, 14]),
                bd.Sample(1, 1, "Barcode", [11, 12, 23, 24]),
                bd.Sample(1, 2, "Drug 1 + Drug 3", [9, 10, 13, 15]),
                bd.Sample(1, 3, "End Cycle Barcode", [11, 12, 23, 24]),
            ),
            channel_map = self.channel_map,
        )

        self.labelled_sample_data = self.sample_data.assign(
            name = [
                "Drug 1 + Drug 2",
                "Drug 1 + Drug 2",
                "Drug 1 + Drug 3",
                "Drug 1 + Drug 3",
                "Drug 1 + Drug 2",
                "Drug 1 + Drug 2",
                "Drug 1 + Drug 3",
                "Drug 1 + Drug 3"
            ],
            compound_a = [
                "Drug 1",
                "Drug 1",
                "Drug 1",
                "Drug 1",
                "Drug 1",
                "Drug 1",
                "Drug 1",
                "Drug 1"
            ],
            compound_b = [
                "Drug 2",
                "Drug 2",
                "Drug 3",
                "Drug 3",
                "Drug 2",
                "Drug 2",
                "Drug 3",
                "Drug 3"
            ],
        )

        self.tmpdir = tempfile.mkdtemp()
        self.pmt_path = pl.Path(self.tmpdir, 'exp.txt')
        self.pmt_path.touch()
        self.config = config.PlugyConfig(
            input_dir = self.tmpdir,
            # workaround since there is only one simulated sample
            # readout_analysis_column is by default "readout_peak_z_score"
            # but z score calculation happens only for more than one sample
            # in sample_df
            readout_analysis_column = 'readout_peak_median'
        )


    @unittest.skip
    def test_plot_test_data(self):
        test_data_fig, test_data_ax = plt.subplots(1, 2, figsize = (40, 10))
        test_data_ax[0].plot(
            self.clean_data.time,
            self.clean_data.green,
            color = "green"
        )
        test_data_ax[0].plot(
            self.clean_data.time,
            self.clean_data.uv,
            color = "blue"
        )
        test_data_ax[0].plot(
            self.clean_data.time,
            self.clean_data.orange,
            color = "orange"
        )

        test_data_ax[1].plot(
            self.noisy_data.time,
            self.noisy_data.green,
            color = "green"
        )
        test_data_ax[1].plot(
            self.noisy_data.time,
            self.noisy_data.uv,
            color = "blue"
        )
        test_data_ax[1].plot(
            self.noisy_data.time,
            self.noisy_data.orange,
            color = "orange"
        )
        for i in range(2):
            test_data_ax[i].set_xlabel("Time [s]")
            test_data_ax[i].set_ylabel("PMT Output [V]")
        test_data_fig.tight_layout()
        test_data_fig.show()
        self.assertTrue(True)


    @unittest.skip
    def test_plot_detected_data(self):
        """
        Tests plotting of the plug data together with the pmt data
        """
        with unittest.mock.patch.object(
            target = pmt.PmtData,
            attribute = "read_txt",
            new = lambda _: self.noisy_data,
        ):

            # noinspection PyTypeChecker
            plug_data = plug.PlugData(
                pmt_data = pmt.PmtData(
                    input_file = pl.Path("MOCK"),
                    peak_min_distance = 0.03,
                    config = self.config,
                ),
                plug_sequence = None,
                channel_map = None,
                config = self.config,
            )

        plug_data_fig, plug_data_ax = plt.subplots(figsize = (40, 10))
        plug_data_ax = plug_data.plot_plug_pmt_data(axes = plug_data_ax)

        plug_data_fig.tight_layout()
        plug_data_fig.show()

        self.assertTrue(True)


    # noinspection DuplicatedCode
    def test_plug_detect_clean_data(self):
        """
        Tests detecting simple plugs from clean data
        """
        with unittest.mock.patch.object(
            target = pmt.PmtData,
            attribute = "read_txt",
            new = lambda _: self.clean_data,
        ):

            # noinspection PyTypeChecker
            plug_data = plug.PlugData(
                pmt_data = pmt.PmtData(
                    input_file = pl.Path("MOCK"),
                    peak_min_distance = 0.03,
                    config = self.config,
                ),
                has_controls = False,
                samples_per_cycle = 2,
                plug_sequence = None,
                channel_map = None,
                config = self.config,
            )

        pd_test.assert_frame_equal(
            self.cycle_data.round(),
            plug_data.plug_df[self.cycle_data.columns].round()
        )


    # noinspection DuplicatedCode
    def test_plug_detect_noisy_data(self):
        """
        Tests detecting plugs with a large amount of noise
        """
        with unittest.mock.patch.object(
            target = pmt.PmtData,
            attribute = "read_txt",
            new = lambda _: self.noisy_data
        ):

            # noinspection PyTypeChecker
            plug_data = plug.PlugData(
                pmt_data = pmt.PmtData(
                    input_file = pl.Path("MOCK"),
                    peak_min_distance = 0.03,
                    config = self.config,
                ),
                has_controls = False,
                samples_per_cycle = 2,
                plug_sequence = None,
                channel_map = None,
                config = self.config,
            )

        pd_test.assert_frame_equal(
            self.cycle_data.round(),
            plug_data.plug_df[self.cycle_data.columns].round()
        )


    # noinspection DuplicatedCode
    def test_plug_cycle_sample_calling(self):
        """
        Tests if cycles and sample numbers are properly detected
        """
        with unittest.mock.patch.object(
            target = pmt.PmtData,
            attribute = "read_txt",
            new = lambda _: self.noisy_data,
        ):

            # noinspection PyTypeChecker
            plug_data = plug.PlugData(
                pmt_data = pmt.PmtData(
                    input_file = pl.Path("MOCK"),
                    peak_min_distance = 0.03,
                    config = self.config,
                ),
                has_controls = False,
                samples_per_cycle = 2,
                plug_sequence = None,
                channel_map = None,
                min_end_cycle_barcodes = 3,
                n_bc_adjacent_discards = 0,
                config = self.config,
            )
            plug_data.detect_samples()

        pd_test.assert_frame_equal(
            self.sample_data.round(),
            plug_data.sample_df.iloc[:,0:8].round()
        )


    def test_plug_sample_labelling(self):
        """
        Tests if samples are properly labelled with the help of a
        bd.PlugSequence object
        """

        with unittest.mock.patch.object(
            target = pmt.PmtData,
            attribute = "read_txt",
            new = lambda _: self.noisy_data,
        ):

            # noinspection PyTypeChecker
            plug_data = plug.PlugData(
                pmt_data = pmt.PmtData(
                    input_file = pl.Path("MOCK"),
                    peak_min_distance = 0.03,
                    config = self.config,
                ),
                has_controls = False,
                plug_sequence = self.plug_sequence,
                channel_map = self.channel_map,
                min_end_cycle_barcodes = 3,
                n_bc_adjacent_discards = 0,
                config = self.config,
            )
            plug_data.detect_samples()

        pd_test.assert_frame_equal(
            self.labelled_sample_data.round(),
            plug_data.sample_df.round()
        )


    # noinspection DuplicatedCode
    def test_plug_detect_clean_data_cell_norm(self):
        """
        Tests detecting simple plugs from clean data
        """
        with unittest.mock.patch.object(
            target = pmt.PmtData,
            attribute = "read_txt",
            new = lambda _: self.clean_data,
        ):

            # noinspection PyTypeChecker
            plug_data = plug.PlugData(
                pmt_data = pmt.PmtData(
                    input_file = pl.Path("MOCK"),
                    peak_min_distance = 0.03,
                    config = self.config,
                ),
                has_controls = False,
                samples_per_cycle = 2,
                plug_sequence = None,
                channel_map = None,
                normalize_using_control=True,
                config = self.config,
            )

        print(self.normalized_cycle_data)
        print(plug_data.plug_df[self.normalized_cycle_data.columns])

        pd_test.assert_almost_equal(
            self.normalized_cycle_data,
            plug_data.plug_df[self.normalized_cycle_data.columns],
            atol = .01
        )


    # noinspection DuplicatedCode
    def test_plug_detect_noisy_data_cell_norm(self):
        """
        Tests detecting plugs with a large amount of noise
        """


        print(self.normalized_cycle_data)

        with unittest.mock.patch.object(
            target = pmt.PmtData,
            attribute = "read_txt",
            new = lambda _: self.noisy_data,
        ):

            # noinspection PyTypeChecker
            plug_data = plug.PlugData(
                pmt_data = pmt.PmtData(
                    input_file = pl.Path("MOCK"),
                    peak_min_distance = 0.03,
                    config = self.config,
                ),
                has_controls = False,
                samples_per_cycle = 2,
                plug_sequence = None,
                channel_map = None,
                normalize_using_control=True,
                config = self.config,
            )

        pd_test.assert_frame_equal(
            self.normalized_cycle_data.round(),
            plug_data.plug_df[self.normalized_cycle_data.columns].round()
        )


if __name__ == '__main__':

    unittest.main()
