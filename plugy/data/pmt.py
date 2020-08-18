#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Author      Nicolas Peschke
# Date        19.09.2019
#
# This file is part of the `plugy` python module
#
# Copyright
# 2018-2020
# EMBL, Heidelberg University
#
# File author(s): Dénes Türei (turei.denes@gmail.com)
#                 Nicolas Peschke
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

import pandas as pd
import numpy as np

import matplotlib.patches as mpl_patch
import matplotlib.collections as mpl_coll
import matplotlib.pyplot as plt
import seaborn as sns

from dataclasses import dataclass
from ..data.config import PlugyConfig

module_logger = logging.getLogger("plugy.data.pmt")


@dataclass
class PmtData(object):
    input_file: pl.Path
    acquisition_rate: int = 300
    cut: tuple = (None, None)
    correct_acquisition_time: bool = True
    ignore_orange_channel: bool = False
    ignore_green_channel: bool = False
    ignore_uv_channel: bool = False
    auto_gain: bool = False
    digital_gain_uv: float = 1.0
    digital_gain_green: float = 1.0
    digital_gain_orange: float = 1.0
    bc_override_threshold: float = None
    config: PlugyConfig = PlugyConfig()

    def __post_init__(self):
        module_logger.info(f"Creating PmtData object from file {self.input_file.absolute()}")
        module_logger.debug(f"Configuration:")
        for k, v in self.__dict__.items():
            module_logger.debug(f"{k}: {v}")

        self.data = self.read_txt()
        self.data = self.set_channel_values()
        self.data = self.cut_data()

        if self.auto_gain:
            self.digital_gain_uv, self.digital_gain_green, self.digital_gain_orange = self.calculate_gain()

        self.data = self.digital_gain()

        if self.bc_override_threshold is not None:
            self.data = self._override_barcode()
        # self.data = self.correct_crosstalk()

    def read_txt(self) -> pd.DataFrame:
        """
        Reads input_file
        :return: pd.DataFrame containing the PMT data of all channels
        """
        module_logger.info("Reading input file")
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
        if self.ignore_green_channel:
            module_logger.info("Setting green channel to 0.0")
            df = df.assign(green = 0.0)

        if self.ignore_uv_channel:
            module_logger.info("Setting uv channel to 0.0")
            df = df.assign(uv = 0.0)

        if self.ignore_orange_channel:
            module_logger.info("Setting orange channel to 0.0")
            df = df.assign(orange = 0.0)

        if self.correct_acquisition_time:
            module_logger.info("Correcting acquisition time")
            df = df.assign(time = np.linspace(self.data.time[0], self.data.time[0] + time_between_samplings * (len(df) - 1), len(df)))

        return df

    def calculate_gain(self):
        # for channel in ["uv", "green", "orange"]:
        #     pass
        raise NotImplementedError

    def digital_gain(self) -> pd.DataFrame:
        """
        Multiplies the corresponding channels PMT output by the given float (digital_gain_*)
        """
        df = self.data
        if self.digital_gain_uv != 1.0:
            module_logger.info(f"Applying digital gain for uv channel ({self.digital_gain_uv})")
            df = df.assign(uv = lambda x: x.uv * self.digital_gain_uv)
        if self.digital_gain_green != 1.0:
            module_logger.info(f"Applying digital gain for green channel ({self.digital_gain_green})")
            df = df.assign(green = lambda x: x.green * self.digital_gain_green)
        if self.digital_gain_orange != 1.0:
            module_logger.info(f"Applying digital gain for orange channel ({self.digital_gain_orange})")
            df = df.assign(orange = lambda x: x.orange * self.digital_gain_orange)

        return df

    def plot_pmt_data(self, axes: plt.Axes, cut: tuple = (None, None)) -> plt.Axes:
        """
        Plots the raw PMT data to the specified axes object
        :param axes: plt.Axes object to draw on
        :param cut: Tuple to specify upper and lower time bounds for the pmt data to be plotted (lower, upper)
        :return: The axes object with the plot
        """
        module_logger.debug(f"Plotting PMT data")
        df = self.cut_data(cut = cut)

        with sns.axes_style("darkgrid", {"xtick.bottom": True,
                                         'xtick.major.size': 1.0,
                                         'xtick.minor.size': 0.5}):
            sns.lineplot(x = df.time, y = df.green, estimator = None, ci = None, sort = False, color = self.config.colors["green"], ax = axes)
            sns.lineplot(x = df.time, y = df.orange, estimator = None, ci = None, sort = False, color = self.config.colors["orange"], ax = axes)
            sns.lineplot(x = df.time, y = df.uv, estimator = None, ci = None, sort = False, color = self.config.colors["blue"], ax = axes)

            if df.time.max() - df.time.min() > 1000:
                major_tick_freq = 100
                minor_tick_freq = 25
            else:
                major_tick_freq = 10
                minor_tick_freq = 1

            axes.set_xticks(range(int(round(df.time.min())), int(round(df.time.max())), major_tick_freq), minor = False)
            axes.set_xticks(range(int(round(df.time.min())), int(round(df.time.max())), minor_tick_freq), minor = True)
            axes.set_xlim(left = int(round(df.time.min())), right = int(round(df.time.max())))

            axes.grid(b = True, which = "major", color = "k", linewidth = 1.0)
            axes.grid(b = True, which = "minor", color = "k", linewidth = 0.5)

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

    def correct_crosstalk(self) -> pd.DataFrame:
        """
        Implements crosstalk compensation as in Federica's BraDiPluS package (peaksSelection.R)
        :return: pd.DataFrame with corrected orange channel
        """
        # df = self.data
        #
        # df = df.assign(orange = df.orange - (0.45 * df.green))
        #
        # return df
        raise NotImplementedError

    def _override_barcode(self) -> pd.DataFrame:
        """
        Overrides values in the barcode channel to 1 if current barcode value > bc_override_threshold
        """
        module_logger.warning(f"Setting barcode channel to 1 for barcode values > {self.bc_override_threshold}")
        df = self.data

        barcode = self.config.channels["barcode"][0]

        df[barcode].loc[df[barcode] > self.bc_override_threshold] = 1

        return df
