"""
Author      Nicolas Peschke
Date        19.09.2019

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

import re
import gzip
import pathlib as pl

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from dataclasses import dataclass, field


@dataclass
class PmtData(object):
    input_file: pl.Path
    acquisition_rate: int = 300
    cut: tuple = (None, None)
    channels: dict = field(default_factory=lambda: {"barcode": ("blue", 3), "cells": ("orange", 2), "readout": ("green", 1)})
    colors: dict = field(default_factory=lambda: {"green": "#5D9731", "blue": "#3A73BA", "orange": "#F68026"})
    correct_acquisition_time: bool = True
    ignore_orange_channel: bool = False
    ignore_green_channel: bool = False
    ignore_uv_channel: bool = False
    digital_gain_uv: float = 1.0
    digital_gain_green: float = 1.0
    digital_gain_orange: float = 1.0

    def __post_init__(self):
        self.data = self.read_txt()
        self.data = self.set_channel_values()
        self.data = self.cut_data()
        self.data = self.digital_gain()

    def read_txt(self) -> pd.DataFrame:
        """
        Reads input_file
        :return: pd.DataFrame containing the PMT data of all channels
        """
        if self.input_file.exists():
            if self.input_file.suffix == ".gz":
                with gzip.open(self.input_file, "rt") as f:
                    end_of_header = self.find_data(f)

            elif self.input_file.suffix == ".txt":
                with self.input_file.open("rt") as f:
                    end_of_header = self.find_data(f)

            else:
                raise NotImplementedError(f"Input file has to be either .txt or .txt.gz, {self.input_file.suffix} files are not implemented!")

            data_frame = pd.read_csv(self.input_file, sep="\t", decimal=",", skiprows=end_of_header, header=None).iloc[:, 1:]
            data_frame.columns = ["time", "green", "orange", "uv"]

            return data_frame

        else:
            raise FileNotFoundError(f"Input file ({self.input_file.absolute()}) does not exist! Check the path!")

    @staticmethod
    def find_data(file) -> int:
        """
        Finds the ending of the header in a multichannel acquisition output file.
        Identifies data by its leading \t
        :param file: File object
        :return: Line number of the first data line
        """
        idx = -1
        for idx, line in enumerate(file):
            if re.match(pattern=r"\t\d", string=line) is not None:
                break

        assert idx > -1, "No lines detected in input_file! Check the contents of the file!"
        assert idx < 50, f"Automatically detected header length exceeds 50 lines ({idx})"

        return idx

    def cut_data(self) -> pd.DataFrame:
        """
        Returns data between time range specified in cut
        :return: pd.DataFrame containing the data in the time range
        """
        try:
            if self.cut[0] >= self.cut[1]:
                raise AttributeError(f"Cut has to be specified like cut=(min, max) you specified {self.cut}")
        except TypeError:
            # in case of comparison with None
            pass

        df = self.data
        if self.cut[0] is not None:
            df = df.loc[df.time >= self.cut[0]]

        if self.cut[1] is not None:
            df = df.loc[df.time <= self.cut[1]]

        return df

    def set_channel_values(self):
        """
        Sets & corrects values in the multichannel acquisition data.
        :return: DataFrame with the corrected data
        """
        time_between_samplings = 1 / self.acquisition_rate

        df = self.data.copy()
        if self.ignore_green_channel:
            df = df.assign(green=0.0)

        if self.ignore_uv_channel:
            df = df.assign(uv=0.0)

        if self.ignore_orange_channel:
            df = df.assign(orange=0.0)

        if self.correct_acquisition_time:
            df = df.assign(time=np.linspace(0, time_between_samplings * (len(df) - 1), len(df)))

        return df

    def digital_gain(self):
        """
        Multiplies the corresponding channels PMT output by the given float (digital_gain_*)
        """
        df = self.data
        df = df.assign(uv=lambda x: x.uv * self.digital_gain_uv,
                       green=lambda x: x.green * self.digital_gain_green,
                       orange=lambda x: x.orange * self.digital_gain_orange)

        return df

    def plot_raw_data(self, axes: plt.Axes):
        """
        Plots the raw PMT data to the specified axes object
        :param axes: plt.Axes object to draw on
        :return: The axes object with the plot
        """
        sns.lineplot(x=self.data.time, y=self.data.green, estimator=None, ci=None, sort=False, color=self.colors["green"], ax=axes)
        sns.lineplot(x=self.data.time, y=self.data.orange, estimator=None, ci=None, sort=False, color=self.colors["orange"], ax=axes)
        sns.lineplot(x=self.data.time, y=self.data.uv, estimator=None, ci=None, sort=False, color=self.colors["blue"], ax=axes)
        axes.set_xlabel("Time [s]")
        axes.set_ylabel("Fluorescence [AU]")

        return axes
