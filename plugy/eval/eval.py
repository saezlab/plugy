"""
Author      Nicolas Peschke
Date        20.05.2019
"""

import pathlib as pl
import numpy as np
import pandas as pd
import pickle

from dataclasses import dataclass, field
from ..main import Plugy


@dataclass
class EvalPlugy(Plugy):
    infile: pl.Path
    results_dir: pl.Path = pl.Path.cwd().joinpath("results")
    experiment_name: str = "evalPlugyObject"
    cut: tuple = (None, None)
    drugs: list = field(default_factory=list)
    signal_threshold: float = .02
    adaptive_signal_threshold: bool = True
    peak_minwidth: float = 5
    channels: dict = field(default_factory=lambda: {"barcode": ("blue", 3), "cells": ("orange", 2), "readout": ("green", 1)})
    colors: dict = field(default_factory=lambda: {"green": "#5D9731", "blue": "#3A73BA", "orange": "#F68026"})
    discard: tuple = (2, 1)
    x_ticks_density: float = 5
    gaussian_smoothing_sigma: float = 33
    adaptive_threshold_blocksize: float = 111
    adaptive_threshold_method: str = "gaussian"
    adaptive_threshold_sigma: float = 190
    adaptive_threshold_offset: float = 0.01
    merge_close_peaks: float = 50
    drug_sep: str = "&"
    direct_drug_combinations: bool = False
    barcode_intensity_correction: float = 1.0
    ignore_orange_channel: bool = False
    ignore_green_channel: bool = False
    ignore_uv_channel: bool = False
    acquisition_rate: int = 300
    correct_acquisition_time: bool = True
    figure_file_type: str = ".svg"

    def __post_init__(self):
        """
        Upon initialization of the EvalPlugy object, the infile is read and preprocessed
        """
        self.data = None
        self.filtered_peaks = pd.DataFrame()
        self.set_channels(channels=self.channels)
        self.name = self.infile.name

        # Create result directory if it does not already exist
        try:
            self.results_dir.mkdir(parents=False, exist_ok=False)
        except FileNotFoundError:
            pass
        except FileExistsError:
            pass

        # self.read()
        #
        # self.set_channel_values(correct_time=self.correct_acquisition_time, ignore_green=self.ignore_green_channel, ignore_orange=self.ignore_orange_channel, ignore_uv=self.ignore_uv_channel)
        # self.save_plugy(self.experiment_name)

    def main(self):
        """
        Similar to Plugy.main(), with added call to EvalPlugy.set_channel_values() to correct timing and ignore unused channels.
        """
        self.read()
        self.set_channel_values(correct_time=self.correct_acquisition_time, ignore_green=self.ignore_green_channel, ignore_orange=self.ignore_orange_channel, ignore_uv=self.ignore_uv_channel)
        self.strip()
        self.find_peaks()
        self.peaks_df()
        self.plot_peaks()
        self.plot_peaks(raw=True)

        self.sample_names()
        self.find_cycles()
        self.export()

        self.save_plugy(self.experiment_name)

    def set_channel_values(self, correct_time: bool = True, ignore_green: bool = False, ignore_orange: bool = False, ignore_uv: bool = False):
        """
        Sets & corrects values in the multichannel acquisition data.
        :param correct_time: If the time should be corrected from having 100 measurements at a single
                             timepoint to evenly spaced measurements using the acquisition rate.
        :param ignore_green: If all values of the green channel should be set to 0
        :param ignore_orange: If all values of the orange channel should be set to 0
        :param ignore_uv: If all values of the uv channel should be set to 0
        """
        time_between_samplings = 1 / self.acquisition_rate

        # Iterate through self.data and overwrite if iteration successful
        with np.nditer(self.data, op_flags=["readwrite"]) as data:
            for idx, value in enumerate(data):
                # Get the column index
                col = idx % 4

                # Change column values depending on parameters
                if correct_time and (col == 0):
                    value[...] = (idx / 4) * time_between_samplings

                if ignore_green and (col == 1):
                    value[...] = 0

                if ignore_orange and (col == 2):
                    value[...] = 0

                if ignore_uv and (col == 3):
                    value[...] = 0

    def strip(self):
        """
        Cuts away data acquired outside of the time interval specified with cut.
        """
        temp_df = pd.DataFrame(self.data)

        temp_df = temp_df.loc[temp_df[0] > self.cut[0]]
        temp_df = temp_df.loc[temp_df[0] < self.cut[1]]

        self.data = temp_df.values

    # def filter_barcodes(self, n_discards: int = 1):
    #     barcodes = self..barcode
    #
    #     discards = []
    #
    #     for idx in range(len(barcodes)):
    #         try:
    #             if barcodes[idx] or barcodes[idx - nDiscards] or barcodes[idx + nDiscards]:
    #                 discards.append(True)
    #
    #             else:
    #                 discards.append(False)
    #         except KeyError:
    #             discards.append(False)
    #
    #     peaksDf.discard = discards
    #
    #     filteredPeaks = peaksDf.loc[lambda df: df.discard == False, :]
    #     return filteredPeaks

    def save_plugy(self, export_filename: str):
        """
        Saves the EvalPlugy object as a serialized pickle in the results_dir
        :param export_filename: Filename for the object
        """
        with self.results_dir.joinpath(export_filename).open("wb") as p:
            pickle.dump(self, p)


# class Test:
#     def __init__(self):
#         self.data = np.ndarray([1])
#
#     def test(self):
#         self.data = np.ndarray([1, 2, 3])
