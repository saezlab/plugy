"""
Author      Nicolas Peschke
Date        20.05.2019
"""

import pathlib as pl
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from dataclasses import dataclass, field
from ..main import Plugy


@dataclass
class EvalPlugy(Plugy):
    infile: pl.Path
    results_dir: pl.Path = pl.Path.cwd().joinpath("results")
    cut: tuple = (None, None)
    drugs: list = field(default_factory=list)
    signal_threshold: float = .02
    adaptive_signal_threshold: bool = True
    peak_minwidth: float = 5
    plug_minlength: float = 0.5
    n_bc_adjacent_discards: int = 1
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
        Upon initialization of the EvalPlugy object, some additional attributes are collected and the results_dir is
        created if it does not exist already.
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

    def main(self):
        """
        Similar to Plugy.main(), with added call to EvalPlugy.set_channel_values()
        to correct timing and ignore unused channels.
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
        self.filter_peakdf(discard_adjacent_plugs=self.n_bc_adjacent_discards, plug_length_threshold=self.plug_minlength)

        self.save_plugy(self.name)

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

        # noinspection PyAttributeOutsideInit
        self.data = temp_df.values

    def save_plugy(self, export_filename: str):
        """
        Saves the EvalPlugy object as a serialized pickle in the results_dir
        :param export_filename: Filename for the object
        """
        with self.results_dir.joinpath(export_filename + ".p").open("wb") as p:
            pickle.dump(self, p)

    def filter_peakdf(self, discard_adjacent_plugs: int = 1, plug_length_threshold: float = 0.5):
        """
        Filters peakdf to remove barcodes, too short plugs and plugs that are adjacent to barcodes
        :param discard_adjacent_plugs: The number of plugs adjacent on both sides of the barcode to discard
        :param plug_length_threshold: The minimum length of a plug to keep in seconds
        :return: The filtered DataFrame, also sets self.filtered_peaks
        """
        discards = list()

        # barcodes = self.peakdf.barcodes
        for idx in range(len(self.peakdf.barcode)):
            try:
                if self.peakdf.barcode[idx] or self.peakdf.barcode[idx - discard_adjacent_plugs] or self.peakdf.barcode[idx + discard_adjacent_plugs] or self.peakdf.length[idx] < plug_length_threshold:
                    discards.append(True)
                else:
                    discards.append(False)

            except KeyError:
                if self.peakdf.length[idx] < plug_length_threshold:
                    discards.append(True)
                else:
                    discards.append(False)

        self.peakdf.discard = discards

        # noinspection PyAttributeOutsideInit
        self.filtered_peaks = self.peakdf.loc[lambda df: df.discard == False, :]
        return self.filtered_peaks

    def plot_fluorescence_hist(self, axes: plt.Axes):
        """
        Plots the distribution of plug green fluorescence intensity
        :param axes: plt.Axes object to draw on
        :return: The plt.Axes object with the plot
        """
        axes = sns.distplot(self.filtered_peaks.green, rug=True, color="green", ax=axes)
        axes.set_ylabel("Occurrences")
        axes.set_xlabel("Green Channel Fluorescence [AU]")
        axes.set_title("Plug Fluorescence Distribution")
        return axes


def load_plugy_object(path: pl.Path) -> EvalPlugy:
    """
    Loads EvalPlugy object from the specified path.
    :param path: Path as pathlib.Path object
    :return: The stored plugy.eval.EvalPlugy object
    """
    with path.open("rb") as plugy_file:
        return pickle.load(plugy_file)
