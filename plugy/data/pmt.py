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
import re
import gzip
import pathlib as pl
import importlib as imp
from typing import Union
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import scipy.signal as sig

import matplotlib.ticker as mpl_ticker
import matplotlib.pyplot as plt
import seaborn as sns

from ..data.config import PlugyConfig
from .. import misc

module_logger = logging.getLogger(__name__)


@dataclass
class PmtData(object):

    input_file: pl.Path
    acquisition_rate: int = 300
    cut: tuple = (None, None)
    correct_acquisition_time: bool = True

    channels: dict = field(
        default_factory = lambda: {
            'barcode': ('uv', 3),
            'control': ('orange', 2),
            'readout': ('green', 1),
        }
    )
    ignore_channels: set = field(default_factory = set)
    fake_gains: dict = field(default_factory = dict)
    fake_gain_default: float = 1.0
    fake_gain_adaptive: bool = False
    barcode_raw_threshold: float = None
    channels: dict = field(
        default_factory = lambda: {
            'barcode': ('uv', 3),
            'control': ('orange', 2),
            'readout': ('green', 1),
        }
    )
    peak_min_threshold: float = 0.05
    peak_max_threshold: float = 2.0
    peak_min_distance: float = 0.03
    peak_min_prominence: float = 0
    peak_max_prominence: float = 10
    peak_min_width: float = 0.5
    peak_max_width: float = 1.5
    width_rel_height: float = 0.5
    merge_peaks_distance: float = 0.2
    merge_peaks_by: str = 'center'

    config: PlugyConfig = field(default_factory = PlugyConfig)


    def __post_init__(self):

        module_logger.info(
            f"Creating PmtData object from file {self.input_file.absolute()}"
        )
        module_logger.debug(f"Configuration:")
        for k, v in self.__dict__.items():
            module_logger.debug(f"{k}: {v}")

        self.data = self.read_txt()
        self.data = self.set_channel_values()
        self.data = self.cut_data()

        self._set_fake_gain_adaptive()
        self.apply_fake_gain()
        self._override_barcode()


    def reload(self):
        """
        Reloads the object from the module level.
        """

        modname = self.__class__.__module__
        mod = __import__(modname, fromlist = [modname.split('.')[0]])
        imp.reload(mod)
        new = getattr(mod, self.__class__.__name__)
        setattr(self, '__class__', new)


    def read_txt(self) -> pd.DataFrame:
        """
        Reads input_file
        :return: pd.DataFrame containing the PMT data of all channels
        """
        module_logger.info(f"Reading file {self.input_file.absolute()}")
        if self.input_file.exists():
            if self.input_file.suffix == ".gz":
                module_logger.info("Detected gzipped file")
                with gzip.open(self.input_file, "rt") as f:
                    header_info = self.parse_header(f)

            elif self.input_file.suffix == ".txt":
                module_logger.info("Detected uncompressed txt file")
                with self.input_file.open("rt") as f:
                    header_info = self.parse_header(f)

            else:
                raise NotImplementedError(f"Input file has to be either .txt or .txt.gz, {self.input_file.suffix} files are not implemented!")

            data_frame = pd.read_csv(self.input_file, sep = "\t", decimal = header_info["decimal_separator"], skiprows = header_info["end_of_header"], header = None).iloc[:, 1:]
            data_frame.columns = ["time", "green", "orange", "uv"]

            return data_frame

        else:
            raise FileNotFoundError(f"Input file ({self.input_file.absolute()}) does not exist! Check the path!")


    def parse_header(self, file) -> dict:
        """
        Parses the LabView header and extracts information in form of a dict
        "end_of_header" : Line number of the first data line
        "decimal_separator" : Type of decimal separator
        :param file: read opened text file object
        :return: dictionary containing end_of_header and decimal_separator
        """
        info = dict()
        info["end_of_header"] = self.find_data(file)
        info["decimal_separator"] = self.detect_decimal_separator(file)
        return info


    @staticmethod
    def detect_decimal_separator(file) -> str:
        """
        Parses the LabView header and extracts the decimal separator
        :param file: read opened text file object
        :return: String containing the decimal separator
        """
        file.seek(0)
        for line in file:
            if line.startswith("Decimal_Separator"):
                sep = line.split()[1]
                module_logger.debug(f"Detected decimal separator: {sep}")
                return sep

        module_logger.error("Automatic decimal separator detection failed, falling back to ',' as decimal separator.")
        return ","

    @staticmethod
    def find_data(file) -> int:
        """
        Finds the ending of the header in a multichannel acquisition output file.
        Identifies data by its leading \t
        :param file: File object
        :return: Line number of the first data line
        """
        file.seek(0)
        idx = -1
        for idx, line in enumerate(file):
            if re.match(pattern = r"\t\d", string = line) is not None:
                break

        module_logger.debug(f"Detected end of header in line {idx}")
        assert idx > -1, "No lines detected in input_file! Check the contents of the file!"
        assert idx < 50, f"Automatically detected header length exceeds 50 lines ({idx})"

        return idx


    def cut_data(self, **kwargs) -> pd.DataFrame:
        """
        Returns data between time range specified in cut
        :param kwargs: "cut": specify an upper and lower limit (lower, upper) other than the one in the object already
        :return: pd.DataFrame containing the data in the time range
        """
        module_logger.debug(f"Cutting {self.__repr__()}")

        if "cut" in kwargs.keys():
            cut = kwargs["cut"]
        else:
            cut = self.cut

        df = self.data

        try:
            if cut[0] >= cut[1]:
                raise AttributeError(f"Cut has to be specified like cut = (min, max) you specified {cut}")
        except TypeError:
            # in case of comparison with None
            pass

        if cut[0] is not None:
            module_logger.debug(f"Cutting data before t = {cut[0]}")
            df = df.loc[df.time >= cut[0]]

        if cut[1] is not None:
            module_logger.debug(f"Cutting data after t = {cut[1]}")
            df = df.loc[df.time <= cut[1]]

        return df


    def set_channel_values(self) -> pd.DataFrame:
        """
        Sets & corrects values in the multichannel acquisition data.
        :return: DataFrame with the corrected data
        """
        time_between_samplings = 1 / self.acquisition_rate

        df = self.data.copy()

        for channel in (
            set(self.config.channel_names.values()) &
            misc.to_set(self.ignore_channels)
        ):

            module_logger.info('Setting %s channel to 0.0' % channel)
            df[channel] = .0

        if self.correct_acquisition_time:

            module_logger.info("Correcting acquisition time")
            df = df.assign(
                time = np.linspace(
                    self.data.time[0],
                    self.data.time[0] +
                        time_between_samplings *
                        (len(df) - 1),
                    len(df)
                )
            )

        return df


    def _set_fake_gain_adaptive(self):
        # for channel in ["uv", "green", "orange"]:
        #     pass

        if self.fake_gain_adaptive:

            raise NotImplementedError


    def apply_fake_gain(self):
        """
        Multiplies the corresponding channels PMT output by a factor provided
        in ``fake_gains``.
        """
        df = self.data

        for channel in self.config.channel_names.values():

            fake_gain = (
                self.fake_gains[channel]
                    if channel in self.fake_gains else
                self.fake_gain_default
            )

            if fake_gain != 1.0:

                module_logger.info(
                    f"Applying fake gain for {channel} channel: {fake_gain}"
                )
                df[channel] = df[channel] * fake_gain

        self.data = df


    def plot_pmt_data(
            self,
            axes: plt.Axes,
            cut: tuple = (None, None),
            ylim: tuple = (None, None),
            n_x_ticks: Union[int, float] = None,
        ) -> plt.Axes:
        """
        Plots the raw PMT data to the specified axes object.

        :param axes:
            plt.Axes object to draw on
        :param cut:
            Tuple to specify upper and lower time bounds for the pmt
            data to be plotted (lower, upper)

        :return:
            The axes object with the plot
        """
        module_logger.debug('Plotting PMT data')
        df = self.cut_data(cut = cut)

        with sns.axes_style(
            'darkgrid',
            {
                'xtick.bottom': True,
                'xtick.major.size': 1.0,
                'xtick.minor.size': 0.5
            },
        ):

            for channel in ('green', 'orange', 'uv'):

                sns.lineplot(
                    x = df.time,
                    y = df[channel],
                    estimator = None,
                    ci = None,
                    sort = False,
                    color = self.config.colors[channel],
                    linewidth = 1.,
                    ax = axes,
                )

            if df.time.max() - df.time.min() > 1000:

                major_tick_freq = 100
                minor_tick_freq = 25

            else:

                major_tick_freq = 10
                minor_tick_freq = 1

            axes.set_xticks(
                range(
                    int(round(df.time.min())),
                    int(round(df.time.max())),
                    major_tick_freq
                ),
                minor = False,
            )

            axes.set_xticks(
                range(
                    int(round(df.time.min())),
                    int(round(df.time.max())),
                    minor_tick_freq
                ),
                minor = True,
            )

            axes.set_xlim(
                left = int(round(df.time.min())),
                right = int(round(df.time.max())),
            )

            axes.set_ylim(ylim)

            if n_x_ticks:

                major_locator = plt.MaxNLocator(
                    nbins = n_x_ticks,
                    prune = 'lower',
                )
                minor_locator = plt.MaxNLocator(
                    nbins = n_x_ticks * 10,
                    prune = 'lower',
                )
                minor_locator.MAXTICKS = 10000

                axes.xaxis.set_major_locator(major_locator)
                axes.xaxis.set_minor_locator(minor_locator)

            axes.tick_params(
                axis = 'x',
                direction = 'out',
                length = 8, width = 2,
                bottom = True,
                which = 'major',
            )
            axes.tick_params(
                axis = 'x',
                direction = 'out',
                length = 4, width = 1,
                bottom = True,
                which = 'minor',
            )

            axes.grid(b = False, which = "major", axis = 'x')
            #axes.grid(b = True, which = "minor", color = "k", linewidth = 0.5)

            axes.set_facecolor("white")
            axes.tick_params(labelsize = "xx-large")

            axes.set_xlabel(
                "Time [s]",
                horizontalalignment = "left",
                size = "xx-large",
                x = 0,
            )
            axes.set_ylabel("Fluorescence [AU]", size = "xx-large")

        return axes


    def plot_peak_sequence(
            self,
            axes: plt.Axes,
            cut: tuple = (None, None),
        ) -> plt.Axes:
        """
        Creates a compact plot of the detected peaks: x axis is the sequence
        of the peaks, y axis is the median intensity of the channels.
        """

        module_logger.debug('Plotting peak sequence')
        df = self.peak_df

        for channel in ('barcode', 'control', 'readout'):

            if (
                self.config.channel_names[channel] in
                self.config.ignore_channels
            ):

                continue

            sns.scatterplot(
                x = np.arange(1, len(df) + 1),
                y = df['%s_peak_median' % channel],
                color = self.config.channel_color(channel),
                ax = axes,
                s = 50,
            )

        axes.set_facecolor('white')
        axes.grid(b = True, which = 'major', color = 'k', linewidth = 1.0)
        axes.grid(b = True, which = 'minor', color = 'k', linewidth = 0.5)

        if len(df) > 100:

            axes.xaxis.set_major_locator(mpl_ticker.MultipleLocator(20))

        axes.xaxis.set_minor_locator(
            mpl_ticker.MultipleLocator(5 if len(df) > 50 else 2)
        )
        axes.tick_params(labelsize = 'xx-large')
        axes.set_xlabel('Plug sequence', size = 'x-large')
        axes.set_ylabel('Fluorescence [AU]', size = 'x-large')
        axes.set_xlim((-5, len(df) + 5))

        return axes


    def correct_crosstalk(self) -> pd.DataFrame:
        """
        Implements crosstalk compensation as in Federica's BraDiPluS package
        (peaksSelection.R).

        :return: pd.DataFrame with corrected orange channel
        """
        # df = self.data
        #
        # df = df.assign(orange = df.orange - (0.45 * df.green))
        #
        # return df
        raise NotImplementedError


    def _override_barcode(self):
        """
        Overrides values in the barcode channel to 1 if current barcode
        value > bc_override_threshold.
        """

        if self.barcode_raw_threshold:

            module_logger.warning(
                f"Setting barcode channel to 1 for "
                f"barcode values > {self.barcode_raw_threshold}"
            )

            barcode = self.config.channels["barcode"][0]

            self.data[barcode].loc[
                self.data[barcode] > self.barcode_raw_threshold
            ] = 1.


    def detect_peaks(self, merge = None):

        peak_df = pd.DataFrame()

        merge = self.merge_peaks_by if merge is None else merge

        for channel, (channel_color, _) in self.channels.items():

            module_logger.debug(
                f"Running peak detection for {channel} channel"
            )
            peaks, properties = sig.find_peaks(
                self.data[channel_color],
                height = (self.peak_min_threshold, self.peak_max_threshold),
                distance = round(
                    self.peak_min_distance *
                    self.acquisition_rate
                ),
                prominence = (
                    self.peak_min_prominence,
                    self.peak_max_prominence,
                ),
                width = (
                    self.peak_min_width * self.acquisition_rate,
                    self.peak_max_width * self.acquisition_rate,
                ),
                rel_height = self.width_rel_height,
            )

            properties = pd.DataFrame.from_dict(properties)
            properties = properties.assign(barcode = channel == 'barcode')
            peak_df = peak_df.append(properties)

        # Converting ips values to int for indexing later on
        peak_df.assign(
            right_ips = round(peak_df.right_ips),
            left_ips = round(peak_df.left_ips),
        )
        peak_df = peak_df.astype({'left_ips': 'int32', 'right_ips': 'int32'})

        self.peak_df = peak_df

        if merge:

            self._merge_peaks(by = merge)


    def _merge_peaks(self, by = None):
        """
        Merges peaks if merge_peaks_distance is > 0.
        :param peak_df: DataFrame with peaks as called by detect_peaks()
        :return: List containing plug data.
        """

        peak_list = list()

        by = by or self.merge_peaks_by
        by_center = by == 'center'

        if not by_center or self.merge_peaks_distance:

            module_logger.info(
                'Merging %splugs %s' % (
                    'overlapping ' if by != 'center' else '',
                    'with centers closer than %.02f seconds' % (
                        self.merge_peaks_distance
                    )
                        if by == 'center' else
                    '',
                )
            )
            i_merge_dist = self.acquisition_rate * self.merge_peaks_distance

            merge_df = self.peak_df

            if by_center:

                merge_df = merge_df.assign(
                    peak_center = (
                        self.peak_df.left_ips +
                        (self.peak_df.right_ips - self.peak_df.left_ips) / 2
                    )
                )

            by_column = 'peak_center' if by_center else 'left_ips'
            merge_df = merge_df.sort_values(by = by_column)
            merge_df = merge_df.reset_index(drop = True)
            peaks = merge_df[by_column]
            rights = merge_df['right_ips']

            # Count through array
            i = 0
            while i < len(peaks):
                # Count neighborhood
                j = 0
                max_right = rights[i]
                while True:
                    if (
                        i + j >= len(peaks) or
                        (
                            peaks[i + j] - peaks[i] > i_merge_dist
                                if by_center else
                            peaks[i + j] > max_right
                        )
                    ):
                        # Merge from the left edge of the first plug (i)
                        # to the right base of the last plug
                        # to merge (i + j - 1)

                        left = min(merge_df.left_ips[i:i + j])
                        right = max(merge_df.right_ips[i:i + j])
                        peak_list.append(self.quantify_interval(left, right))

                        # Skip to the next unmerged plug
                        i += j
                        break
                    else:
                        max_right = max(max_right, rights[i + j])
                        j += 1

        else:

            module_logger.info(
                'Creating plug list without merging close plugs!'
            )

            for row in self.peak_df.sort_values(by = "left_ips"):

                peak_list.append(
                    self.quantify_interval(
                        row.left_ips,
                        row.right_ips
                    )
                )

         # Build plug_df DataFrame
        module_logger.debug('Building peak_df DataFrame')
        channels = [
            f"{str(key)}_peak_median"
            for key in self.config.channels.keys()
        ]

        self.peak_df = pd.DataFrame(
            peak_list,
            columns = ['start_time', 'end_time'] + channels,
        )


    def quantify_interval(self, start_index, end_index):
        """
        Calculates median for each acquired channel between two data points

        :param start_index: Index of the first datapoint in pmt_data.data
        :param end_index: Index of the last datapoint in pmt_data.data

        :return: List with start_index end_index and the channels according
        to the order of config.channels
        """

        i, j = start_index, end_index

        result = list()
        result.append(i / self.acquisition_rate + self.data.time.iloc[0])
        result.append(j / self.acquisition_rate + self.data.time.iloc[0])

        for _, (channel_color, _) in self.channels.items():

            result.append(self.data[channel_color][i:j].median())

        return result


    def _detection_issues_message(self):

        return (
            f"You may want to try:\n"
            f"\t- Increase `peak_max_width` (currently {self.peak_max_width})"
            f" if plugs are too short to be detected\n"
            f"\t- Decrease `width_rel_height` (currently "
            f"{self.width_rel_height}) if plugs are wider than tall\n"
            f"\t- Cut the data using the `cut` parameter (currently "
            f"{self.cut}) to remove parts from the beginning and end "
            f"which might look like a sample but in fact is not."
        )