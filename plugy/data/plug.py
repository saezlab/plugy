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
import pickle
import importlib as imp

import pathlib as pl

import pandas as pd
import numpy as np
import scipy.signal as sig
import scipy.stats as stats

import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patch
import matplotlib.collections as mpl_coll
import seaborn as sns

from .. import misc
from ..data import pmt, bd
from ..data.config import PlugyConfig
from dataclasses import dataclass

module_logger = logging.getLogger("plugy.data.plug")


@dataclass
class PlugData(object):
    pmt_data: pmt.PmtData
    plug_sequence: bd.PlugSequence
    channel_map: bd.ChannelMap
    auto_detect_cycles: bool = True
    peak_min_threshold: float = 0.05
    peak_max_threshold: float = 2.0
    peak_min_distance: float = 0.03
    peak_min_prominence: float = 0
    peak_max_prominence: float = 10
    peak_min_width: float = 0.5
    peak_max_width: float = 1.5
    width_rel_height: float = 0.5
    merge_peaks_distance: float = 0.2
    n_bc_adjacent_discards: int = 1
    min_end_cycle_barcodes: int = 12
    normalize_using_control: bool = False
    normalize_using_media_control_lin_reg: bool = False
    config: PlugyConfig = PlugyConfig()


    def __post_init__(self):
        module_logger.info(f"Creating PlugData object")
        module_logger.debug(f"Configuration: {[f'{k}: {v}' for k, v in self.__dict__.items()]}")

        self.detect_plugs()


    def reload(self):
        """
        Reloads the object from the module level.
        """

        modname = self.__class__.__module__
        mod = __import__(modname, fromlist = [modname.split('.')[0]])
        imp.reload(mod)
        new = getattr(mod, self.__class__.__name__)
        setattr(self, '__class__', new)


    def detect_plugs(self):
        """
        Finds plugs using the scipy.signal.find_peaks() method. Merges the plugs afterwards if merge_peaks_distance is > 0
        :return: DataFrame containing the plug data and a DataFrame containing information about the peaks as called by sig.find_peaks
        """
        module_logger.info("Finding plugs")
        self._detect_peaks()
        self._merge_peaks()
        self._set_barcode()
        self._normalize_to_control()


    def detect_samples(self):

        self._call_sample_cycles()
        self._label_samples()
        self._create_sample_df()
        self._add_z_scores()
        self._media_lin_reg_norm()
        self._check_sample_df_column(self.config.readout_analysis_column)
        self._check_sample_df_column(self.config.readout_column)


    def _detect_peaks(self):
        """
        Detects peaks using scipy.signal.find_peaks().
        :return: Returns a DataFrame containing information about the peaks as called by sig.find_peaks
        """
        peak_df = pd.DataFrame()

        for channel, (channel_color, _) in self.config.channels.items():

            module_logger.debug(f"Running peak detection for {channel} channel")
            peaks, properties = sig.find_peaks(self.pmt_data.data[channel_color],
                                               height = (self.peak_min_threshold,
                                                       self.peak_max_threshold),

                                               distance = round(self.peak_min_distance *
                                                              self.pmt_data.acquisition_rate),

                                               prominence = (self.peak_min_prominence,
                                                           self.peak_max_prominence),

                                               width = (self.peak_min_width * self.pmt_data.acquisition_rate,
                                                      self.peak_max_width * self.pmt_data.acquisition_rate),

                                               rel_height = self.width_rel_height)

            properties = pd.DataFrame.from_dict(properties)
            properties = properties.assign(barcode = True if channel == "barcode" else False)
            peak_df = peak_df.append(properties)

        # Converting ips values to int for indexing later on
        peak_df.assign(right_ips = round(peak_df.right_ips), left_ips = round(peak_df.left_ips))
        peak_df = peak_df.astype({"left_ips": "int32", "right_ips": "int32"})

        self.peak_df = peak_df


    def _merge_peaks(self):
        """
        Merges peaks if merge_peaks_distance is > 0.
        :param peak_df: DataFrame with peaks as called by detect_peaks()
        :return: List containing plug data.
        """

        plug_list = list()

        if self.merge_peaks_distance > 0:

            module_logger.info(f"Merging plugs with closer centers than {self.merge_peaks_distance} seconds")
            merge_peaks_samples = self.pmt_data.acquisition_rate * self.merge_peaks_distance
            merge_df = self.peak_df.assign(
                plug_center = (
                    self.peak_df.left_ips +
                    (self.peak_df.right_ips - self.peak_df.left_ips) / 2
                )
            )
            merge_df = merge_df.sort_values(by = "plug_center")
            merge_df = merge_df.reset_index(drop = True)

            centers = merge_df.plug_center

            # Count through array
            i = 0
            while i < len(centers):
                # Count neighborhood
                j = 0
                while True:
                    if (i + j >= len(centers)) or (centers[i + j] - centers[i] > merge_peaks_samples):
                        # Merge from the left edge of the first plug (i) to the right base of the last plug to merge (i + j - 1)
                        plug_list.append(self._get_plug_data_from_index(merge_df.left_ips[i], merge_df.right_ips[i + j - 1]))

                        # Skip to the next unmerged plug
                        i += j
                        break
                    else:
                        j += 1

        else:

            module_logger.info("Creating plug list with without merging close plugs!")
            for row in self.peak_df.sort_values(by = "left_ips"):
                plug_list.append(self._get_plug_data_from_index(row.left_ips, row.right_ips))

         # Build plug_df DataFrame
        module_logger.debug("Building plug_df DataFrame")
        channels = [f"{str(key)}_peak_median" for key in self.config.channels.keys()]

        self.plug_df = pd.DataFrame(plug_list, columns = ["start_time", "end_time"] + channels)


    def _set_barcode(self):
        """
        Creates a new boolean column `barcode` in the `plug_df` which is
        `True` if the plug is a barcode.
        """

        # Call barcode plugs
        module_logger.debug("Calling barcode plugs")

        self.plug_df = self.plug_df.assign(
            barcode = (
                self.plug_df.barcode_peak_median >
                self.plug_df.control_peak_median *
                self.config.barcode_threshold
            )
        )


    def _normalize_to_control(self):
        """
        Normalizes the readout channel by dividing its value by a control
        channel.
        """

        if self.normalize_using_control:

            self.plug_df = self.plug_df.assign(
                readout_per_control = (
                    self.plug_df.readout_peak_median /
                    self.plug_df.control_peak_median
                )
            )


    def _get_plug_data_from_index(self, start_index, end_index):
        """
        Calculates median for each acquired channel between two data points
        :param start_index: Index of the first datapoint in pmt_data.data
        :param end_index: Index of the last datapoint in pmt_data.data
        :return: List with start_index end_index and the channels according to the order of config.channels
        """
        return_list = list()
        return_list.append(start_index / self.pmt_data.acquisition_rate + self.pmt_data.data.time.iloc[0])
        return_list.append(end_index / self.pmt_data.acquisition_rate + self.pmt_data.data.time.iloc[0])
        for _, (channel_color, _) in self.config.channels.items():
            return_list.append(self.pmt_data.data[channel_color][start_index:end_index].median())

        return return_list


    def _call_sample_cycles(self):
        """
        Finds cycles and labels individual samples
        :return: Tuple of pd.DataFrame containing sample data
        and the input plug_df updated with info which plugs were discarded
        """

        sample_df = self.plug_df

        # counters
        current_cycle = 0
        bc_peaks = 0
        sm_peaks = 0
        sample_in_cycle = -1
        # new vectors
        cycle = []
        sample = []
        discard = []

        for idx, bc in enumerate(self.plug_df.barcode):

            if bc:
                discard.append(True)
                cycle.append(current_cycle)
                sample.append(max(sample_in_cycle, 0))
                bc_peaks += 1

            else:
                # Checking cycle
                if bc_peaks > 0 or sample_in_cycle < 0:

                    if (
                        bc_peaks >= self.config.min_between_samples_barcodes or
                        sample_in_cycle < 0
                    ):

                        sample_in_cycle += 1

                        if (
                            bc_peaks >= self.min_end_cycle_barcodes and
                            sample_in_cycle > 0
                        ):

                            current_cycle += 1
                            sample_in_cycle = 0

                    bc_peaks = 0

                sample.append(sample_in_cycle)
                cycle.append(current_cycle)

                # Discarding barcode-adjacent plugs
                try:
                    if (
                        self.plug_df.barcode[idx - self.n_bc_adjacent_discards] or
                        self.plug_df.barcode[idx + self.n_bc_adjacent_discards]
                    ):
                        discard.append(True)
                    else:
                        discard.append(False)
                except KeyError:
                    discard.append(False)

        self.plug_df = self.plug_df.assign(cycle_nr = cycle, sample_nr = sample, discard = discard)



    def _add_z_scores(self):
        # Calculating z-score on filtered data and inserting it after readout_peak_median (index 5)

        if len(self.sample_df) > 1:

            self.sample_df.insert(
                loc = 5,
                column = "readout_peak_z_score",
                value = stats.zscore(self.sample_df.readout_peak_median),
            )

            if self.normalize_using_control:

                self.sample_df.insert(
                    loc = 6,
                    column = "readout_per_control_z_score",
                    value = stats.zscore(self.sample_df.readout_per_control),
                )

        else:

            module_logger.warning(f"Samples DataFrame contains {len(self.sample_df)} line(s), omitting z-score calculation!")


    def get_media_control_data(self) -> pd.DataFrame:
        """
        Returns a DataFrame with only media control plugs (both compounds FS)
        :return: pd.DataFrame with media control plugs
        """

        control_labels = misc.to_set(self.config.control_label)

        media_control_data = self.sample_df.loc[
            self.sample_df.compound_a.isin(control_labels) &
            self.sample_df.compound_b.isin(control_labels)
        ]

        return media_control_data


    def get_media_control_lin_reg(self, readout_column: str = ""):
        """
        Calculates a linear regression over time of all media control plugs
        The readout_peak_median column is used to calculate the regression

        :return: Tuple with slope, intercept, rvalue, pvalue and stderr of the regression
                 See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
                 for more information about the returned values
        """
        media_control = self.get_media_control_data()

        readout_column = readout_column or self.config.readout_column

        slope, intercept, rvalue, pvalue, stderr = stats.linregress(media_control.start_time, media_control[readout_column])

        return slope, intercept, rvalue, pvalue, stderr


    def plot_plug_pmt_data(self, axes: plt.Axes, cut: tuple = (None, None)) -> plt.Axes:
        """
        Plots pmt data and superimposes rectangles with the called plugs upon the plot
        :param axes: plt.Axes object to plot to
        :param cut: tuple with (start_time, end_time) to subset the plot to a certain time range
        :return: plt.Axes object with the plot
        """
        module_logger.info("Plotting detected peaks")
        axes = self.pmt_data.plot_pmt_data(axes, cut = cut)

        plug_df = self.plug_df
        sample_df = self.sample_df

        if cut[0] is not None:
            plug_df = plug_df.loc[plug_df.start_time >= cut[0]]
            sample_df = sample_df.loc[sample_df.start_time >= cut[0]]

        if cut[1] is not None:
            plug_df = plug_df.loc[plug_df.end_time <= cut[1]]
            sample_df = sample_df.loc[sample_df.end_time <= cut[1]]

        # Plotting light green rectangles that indicate the used plug length and plug height
        bc_patches = list()
        readout_patches = list()

        for plug in plug_df.itertuples():
            if plug.barcode:
                bc_patches.append(mpl_patch.Rectangle(xy = (plug.start_time, 0), width = plug.end_time - plug.start_time, height = plug.barcode_peak_median))
            # else:
            #     readout_patches.append(mpl_patch.Rectangle(xy = (plug.start_time, 0), width = plug.end_time - plug.start_time, height = plug.readout_peak_median))

        for plug in sample_df.itertuples():
            readout_patches.append(mpl_patch.Rectangle(xy = (plug.start_time, 0), width = plug.end_time - plug.start_time, height = plug.readout_peak_median))

        axes.add_collection(mpl_coll.PatchCollection(bc_patches, facecolors = self.config.colors["blue"], alpha = 0.4))
        axes.add_collection(mpl_coll.PatchCollection(readout_patches, facecolors = self.config.colors["green"], alpha = 0.4))

        return axes


    def highlight_plugs(self, axes: plt.Axes, below_peak: bool = True):

        if below_peak:

            return self.highlight_plugs_below_peak(axes)

        else:

            return self.highlight_plugs_vspan(axes)


    def highlight_plugs_below_peak(self, axes: plt.Axes):

        plug_patches = {
            'readout': [],
            'barcode': [],
        }

        for plug in self.plug_df.itertuples():

            plug_type = 'barcode' if plug.barcode else 'readout'
            patch = mpl_patch.Rectangle(
                xy = (plug.start_time, 0),
                width = plug.end_time - plug.start_time,
                height = getattr(plug, '%s_peak_median' % plug_type),
            )
            plug_patches[plug_type].append(patch)

        colors = {
            'readout': self.config.colors["green"],
            'barcode': self.config.colors["blue"],
        }

        for key, color in colors.items():

            axes.add_collection(
                mpl_coll.PatchCollection(
                    plug_patches[key],
                    facecolors = color,
                    alpha = .4,
                )
            )

        return axes


    def highlight_plugs_vspan(self, axes: plt.Axes):

        for plug in self.plug_df.itertuples():

            color = self.config.colors["blue" if plug.barcode else "green"]
            axes.axvspan(
                xmin = plug.start_time,
                xmax = plug.end_time,
                facecolor = color,
                alpha = .4,
            )

        return axes


    def highlight_samples(self, axes: plt.Axes):

        samples = self.plug_df.groupby(
            ['cycle_nr', 'sample_nr']
        ).agg(
            {
                'start_time': 'min',
                'end_time': 'max',
            }
        )

        ymax = axes.get_ylim()[1]

        for sample in samples.itertuples():

            axes.axvspan(
                xmin = sample.start_time,
                xmax = sample.end_time,
                facecolor = '#777777',
                alpha = .3,
            )
            axes.text(
                x = sample.start_time,
                y = ymax - .1,
                s = '%u/%u' % sample.Index,
                size = 'xx-large',
            )

        return axes


    def plot_cycle_pmt_data(self, axes: plt.Axes) -> plt.Axes:
        """
        Plots pmt data and superimposes filled rectangles for cycles with correct numbers of samples and
        unfilled rectangles for discarded cycles
        :param axes: plt.Axes object to plot to
        :return: plt.Axes object with the plot
        """
        module_logger.info("Plotting cycle data")
        axes = self.pmt_data.plot_pmt_data(axes = axes, cut = (None, None))

        used_cycle_patches = list()
        discarded_cycle_patches = list()
        patch_height = self.pmt_data.data[["green", "orange", "uv"]].max().max()

        for used_cycle in self.sample_df.groupby("cycle_nr"):
            cycle_start_time = used_cycle[1].start_time.min()
            cycle_end_time = used_cycle[1].end_time.max()
            used_cycle_patches.append(mpl_patch.Rectangle(xy = (cycle_start_time, 0), width = cycle_end_time - cycle_start_time, height = patch_height))

        for detected_cycle in self.plug_df.groupby("cycle_nr"):
            cycle_start_time = detected_cycle[1].start_time.min()
            cycle_end_time = detected_cycle[1].end_time.max()
            discarded_cycle_patches.append(mpl_patch.Rectangle(xy = (cycle_start_time, 0), width = cycle_end_time - cycle_start_time, height = patch_height))

            axes.text(x = (cycle_end_time - cycle_start_time) / 2 + cycle_start_time, y = 0.9 * patch_height, s = f"Cycle {detected_cycle[0]}", horizontalalignment = "center")

        axes.add_collection(mpl_coll.PatchCollection(used_cycle_patches, facecolors = "green", alpha = 0.4))
        axes.add_collection(mpl_coll.PatchCollection(discarded_cycle_patches, edgecolors = "red", facecolors = "none", alpha = 0.4))

        # for cycle in self.sample_df.cycle_nr.unique():
        #     cycle_start_time = self.sample_df.loc[self.sample_df.cycle_nr == cycle].start_time.min()
        #     cycle_end_time = self.sample_df.loc[self.sample_df.cycle_nr == cycle].end_time.max()

        return axes


    def _label_samples(self):
        """
        Labels sample_df with associated names and compounds according to the ChannelMap in the PlugSequence
        :param sample_df: pd.DataFrame with sample_nr column to associate names and compounds
        :return: pd.DataFrame with the added name, compound_a and b columns
        """

        # Label samples in case channel map and plug sequence are provided
        if (
            not isinstance(self.channel_map, bd.ChannelMap) or
            not isinstance(self.plug_sequence, bd.PlugSequence)
        ):

            module_logger.warning(
                "Channel map and/or plug sequence not properly specified, "
                "skipping labeling of samples!"
            )
            return

        labelled_df = self.plug_df
        sample_sequence = self.plug_sequence.get_samples(channel_map = self.channel_map)
        labelling_possible = True
        discarded_cycles = list()

        for cycle in labelled_df.groupby("cycle_nr"):

            # cycle[1] is the group DataFrame, cycle[0] is the current cycle_nr
            found_samples = len(cycle[1].sample_nr.unique())
            expected_samples = len(sample_sequence.sequence)

            if found_samples != expected_samples:

                log_msg = f"Cycle {cycle[0]} detected between {cycle[1].start_time.min()} - {cycle[1].end_time.max()} contains {'less' if found_samples < expected_samples else 'more'} samples ({found_samples}) than expected ({expected_samples})"
                if self.auto_detect_cycles:
                    module_logger.info(log_msg)
                    module_logger.info(f"Discarding Cycle {cycle[0]}")
                    labelled_df.loc[labelled_df.cycle_nr == cycle[0], "discard"] = True
                    discarded_cycles.append(cycle[0])
                else:
                    module_logger.critical(log_msg)
                    labelling_possible = False

        check_msg = "\n".join([f"Check:",
                               f"\tIncrease peak_max_width (currently {self.peak_max_width}) if plugs are too short to be detected",
                               f"\tDecrease width_rel_height (currently {self.width_rel_height}) if plugs are wider than tall",
                               f"\tCut of PMT data (currently {self.pmt_data.cut}) being precisely at the start of the first actual sample (not the barcode)"])

        if not labelling_possible:
            raise AssertionError("\n".join([f"Number of found and expected_samples have to match for all cycles", check_msg]))

        # discarded_cycles = labelled_df.cycle_nr.unique()
        assert len(discarded_cycles) < len(labelled_df.cycle_nr.unique()), "\n".join([f"Did not detect any cycle with the proper number of samples", check_msg])

        module_logger.info("Labelling samples with compound names")
        labelled_df["name"] = labelled_df.loc[~labelled_df.discard].sample_nr.apply(lambda nr: self.get_sample_name(nr, sample_sequence))
        labelled_df["compound_a"] = labelled_df.loc[~labelled_df.discard].sample_nr.apply(lambda nr: self.channel_map.get_compounds(sample_sequence.sequence[nr].open_valves)[0])
        labelled_df["compound_b"] = labelled_df.loc[~labelled_df.discard].sample_nr.apply(lambda nr: self.channel_map.get_compounds(sample_sequence.sequence[nr].open_valves)[1])

        self.plug_df = labelled_df


    def _create_sample_df(self):

        sample_df = self.plug_df.loc[self.plug_df.discard == False]
        self.sample_df = sample_df.drop(columns = ["discard", "barcode"])


    def get_sample_name(self, sample_nr: int, sample_sequence: bd.PlugSequence):
        """
        Returns a unified naming string for a sample.
        Concatenation of both compounds or single compound or cell control
        :param sample_nr: Sample number
        :param sample_sequence: bd.PlugSequence object to get open valves from
        """
        compounds = self.channel_map.get_compounds(sample_sequence.sequence[sample_nr].open_valves)
        compounds = [compound for compound in compounds if compound != "FS"]

        if len(compounds) == 0:
            return "Cell Control"
        else:
            return " + ".join(compounds)


    def plot_sample_cycles(self):
        """
        Creates a plot with pmt data for the individual samples and cycles.
        :return: plt.Figure and plt.Axes object with the plot
        """
        names = self.sample_df.name.unique()
        cycles = sorted(self.sample_df.cycle_nr.unique())

        sample_cycle_fig, sample_cycle_ax = plt.subplots(
            nrows = len(names),
            ncols = len(cycles),
            figsize = (7 * len(cycles), 5 * len(names)),
            squeeze = False,
        )

        y_max = self.sample_df.readout_peak_median.max() * 1.1

        for idx_y, name in enumerate(names):
            for idx_x, cycle in enumerate(cycles):
                module_logger.debug(f"Plotting sample {idx_y + 1} of {len(names)}, cycle {idx_x + 1} of {len(cycles)}")
                sample_cycle_ax[idx_y][idx_x] = self.plot_sample(name = name, cycle_nr = cycle, axes = sample_cycle_ax[idx_y][idx_x])
                sample_cycle_ax[idx_y][idx_x].set_ylim((0, y_max))

        sample_cycle_fig.tight_layout()
        return sample_cycle_fig, sample_cycle_ax


    def plot_sample(self, name: str, cycle_nr: int, axes: plt.Axes, offset: int = 10) -> plt.Axes:
        """
        Plots the PMT traces for a particular drug and cycle and
        :param name: Name of the drug combination/valve as listed in the PlugSequence
        :param cycle_nr: Number of the cycle
        :param axes: The plt.Axes object to draw on
        :param offset: How many seconds to plot left and right of the plugs
        :return: The plt.Axes object with the plot
        """
        peak_data = self.sample_df[(self.sample_df.cycle_nr == cycle_nr) & (self.sample_df.name == name)]
        if len(peak_data) == 0:
            axes.text(0.5, 0.5, "No Data")
        else:
            axes = self.plot_plug_pmt_data(axes = axes, cut = (peak_data.start_time.min() - offset, peak_data.end_time.max() + offset))

        axes.set_title(f"{name} | Cycle {cycle_nr}")
        return axes

        # start_time = peak_data.iloc[0].t0 - offset
        # end_time = peak_data.iloc[-1].t1 + offset
        #
        # plotting_data = pd.DataFrame(self.data)
        # plotting_data = plotting_data[(plotting_data[0] > start_time) & (plotting_data[0] < end_time)]
        #
        # sns.lineplot(x = plotting_data[0], y = plotting_data[1], estimator = None, ci = None, sort = False, color = self.colors["green"], ax = axes)
        # sns.lineplot(x = plotting_data[0], y = plotting_data[2], estimator = None, ci = None, sort = False, color = self.colors["orange"], ax = axes)
        # sns.lineplot(x = plotting_data[0], y = plotting_data[3], estimator = None, ci = None, sort = False, color = self.colors["blue"], ax = axes)
        #
        # # Plotting light green rectangles that indicate the used plug length and plug height
        # patches = list()
        # for plug in peak_data.itertuples():
        #     patches.append(mpl_patch.Rectangle(xy = (plug.t0, 0), width = plug.length, height = plug.green))
        # axes.add_collection(mpl_coll.PatchCollection(patches, facecolors = self.colors["green"], alpha = 0.4))
        #
        # axes.set_xlabel("Time [s]")
        # axes.set_ylabel("Fluorescence [AU]")
        # axes.set_title(f"{drug} Cycle {cycle}")

        # return axes


    # QC Plots
    def plot_media_control_evolution(self, axes: plt.Axes, by_sample = False) -> plt.Axes:
        """
        Plots a scatter plot with readout medians for the media control over
        the experiment time.
        
        :param axes: plt.Axes object to draw on
        :param by_sample: True to plot swarmplot by sample number
        :return: plt.Axes object with the plot
        """
        if self.normalize_using_control:
            readout_column = "readout_per_control"
        else:
            readout_column = "readout_peak_median"

        plot_data = self.get_media_control_data()

        if by_sample:
            axes = sns.swarmplot(x = "sample_nr", y = readout_column, data = plot_data, ax = axes, hue = "cycle_nr", dodge = True)
            axes.set_xlabel("Sample Number")
        else:
            slope, intercept, rvalue, _, _ = self.get_media_control_lin_reg(readout_column)
            axes = sns.scatterplot(x = "start_time", y = readout_column, data = plot_data, ax = axes)
            misc.plot_line(slope, intercept, axes)
            axes.text(0.1, 0.9, f"R²: {round(rvalue, 2)}", transform=axes.transAxes)
            axes.set_xlabel("Experiment Time [s]")

        axes.set_title("FS media control plug fluorescence")
        axes.set_ylabel(readout_column)
        return axes


    def add_length_column(self):
        """
        Creates a new column ``length`` in ``sample_df`` with the difference
        of plug start and end times.
        """

        self.sample_df = self.sample_df.assign(
            length = (
                self.sample_df.end_time -
                self.sample_df.start_time
            )
        )


    def plot_length_bias(self, col_wrap: int = 3) -> sns.FacetGrid:
        """
        Plots each plugs fluorescence over its length grouped by valve. Also fits a linear regression to show if there
        is a correlation between the readout and the plug length indicating non ideal mixing.
        
        :param col_wrap: After how many subplots the column should be wrapped.
        :return: sns.FacetGrid object with the subplots
        """

        self.add_length_column()

        df = self.sample_df

        df = df.groupby('name').filter(lambda x: x.shape[0] > 1)

        length_bias_plot = sns.lmplot(
            x = "length",
            y = "readout_peak_median",
            col = "name",
            data = df,
            col_wrap = col_wrap,
        )
        length_bias_plot.set_xlabels("Length")
        length_bias_plot.set_ylabels("Fluorescence [AU]")

        length_bias_plot.set(
            ylim = (0, df.readout_peak_median.max()),
            xlim = (df.length.min(), df.length.max()),
        )

        return length_bias_plot


    def plot_contamination(self, channel_x: str, channel_y: str, axes: plt.Axes, filtered: bool = False, hue: str = "start_time", normalize: bool = False) -> plt.Axes:
        """
        Plots contamination as a scatter plot
        :param channel_x: Name of the channel to be plotted on the x axis (e.g. readout_peak_median, bc_peak_median, control_peak_median)
        :param channel_y: Name of the channel to be plotted on the y axis (e.g. readout_peak_median, bc_peak_median, control_peak_median)
        :param filtered: True if sample_df should be used, False if raw plug_df should be used
        :param axes: plt.Axes object to draw on
        :param hue: Name of the column in plug_df that is used to color the dots
        :param normalize: True plug data should be scaled by its mean
        :return: plt.Axes object with the plot
        """

        if filtered:
            norm_df = self.sample_df
        else:
            norm_df = self.plug_df.loc[self.plug_df.cycle_nr.isin(self.sample_df.cycle_nr.unique())]

        if normalize:
            norm_df = norm_df.assign(norm_x = norm_df[channel_x] / norm_df[channel_x].mean(), norm_y = norm_df[channel_y] / norm_df[channel_y].mean())
            contamination_plot = sns.scatterplot(x = "norm_x", y = "norm_y", hue = hue, data = norm_df, ax = axes)
            axes.set_xlabel(channel_x + " [% of mean]")
            axes.set_ylabel(channel_y + " [% of mean]")

        else:
            contamination_plot = sns.scatterplot(x = channel_x, y = channel_y, hue = hue, data = norm_df, ax = axes)
            axes.set_xlabel(channel_x + " [AU]")
            axes.set_ylabel(channel_y + " [AU]")

        # if filtered:
        #     contamination_plot = sns.scatterplot(x = channel_x, y = channel_y, hue = hue, data = self.sample_df, ax = axes)
        # else:
        #     contamination_plot = sns.scatterplot(x = channel_x, y = channel_y, hue = hue, data = self.plug_df, ax = axes)
        return contamination_plot


    def plot_control_regression(self, axes: plt.Axes) -> plt.Axes:
        """
        Plots a scatter plot of control peak medians over experiment time and applies a linear regression to it
        :param axes: plt.Axes object to draw on
        :return: plt.Axes object with the plot
        """
        axes = sns.regplot(x = "start_time", y = "control_peak_median", data = self.sample_df, ax = axes)
        axes.set_title("Control Time Bias")
        axes.set_ylabel("Peak Median Fluorescence Intensity [AU]")
        axes.set_xlabel("Experiment Time [s]")
        return axes


    def plot_control_cycle_dist(self, axes: plt.Axes) -> plt.Axes:
        """
        Gathers control peak medians by cycle and plots a violin plot
        :param axes: plt.Axes object to draw on
        :return: plt.Axes object with the plot
        """
        axes = sns.violinplot(x = "cycle_nr", y = "control_peak_median", data = self.sample_df, ax = axes)
        axes.set_title("Control Intensity by Cycle")
        axes.set_ylabel("Peak Median Fluorescence Intensity [AU]")
        axes.set_xlabel("Cycle")
        return axes


    def plot_control_sample_dist(self, axes: plt.Axes) -> plt.Axes:
        """
        Gathers control peak medians by sample and plots a violin plot
        :param axes: plt.Axes object to draw on
        :return: plt.Axes object with the plot
        """
        axes = sns.violinplot(x = "name", y = "control_peak_median", data = self.sample_df, ax = axes)
        axes.set_title("Control Intensity by Sample")
        axes.set_ylabel("Peak Median Fluorescence Intensity [AU]")
        axes.set_xlabel("Sample")
        for tick in axes.get_xticklabels():
            tick.set_rotation(90)
        return axes


    def plot_control_readout_correlation(self, axes: plt.Axes) -> plt.Axes:
        """
        Correlates control and readout peak medians
        :param axes: plt.Axes object to draw on
        :return: plt.Axes object with the plot
        """
        # axes = sns.regplot(x = "control_peak_median", y = "readout_peak_median",  data = self.sample_df, ax = axes)
        axes = sns.scatterplot(x = "control_peak_median", y = "readout_peak_median", hue = "sample_nr", style = "cycle_nr", data = self.sample_df, ax = axes)
        axes.set_title("Readout - Control Correlation")
        axes.set_xlabel("Control Peak Median Fluorescence Intensity [AU]")
        axes.set_ylabel("Readout Peak Median Fluorescence Intensity [AU]")

        axes.set_ylim(self.sample_df.readout_peak_median.min() * 0.95, self.sample_df.readout_peak_median.max() * 1.05)
        axes.set_xlim(self.sample_df.control_peak_median.min() * 0.95, self.sample_df.control_peak_median.max() * 1.05)

        return axes


    def save(self, file_path: pl.Path):
        """
        Saves this PlugData object as pickle
        :param file_path: Path to the file to write to
        """
        with file_path.open("wb") as f:
            pickle.dump(self, f)


    def plot_compound_violins(self, column_to_plot: str= "readout_peak_z_score", by_cycle: bool = False) -> sns.FacetGrid:
        """
        Plots a violin plot per compound combination
        :param column_to_plot: Column to be plotted (e.g. readout_peak_z_score, readout_per_control_z_score).
        :param by_cycle: Produce separate plots for each cycle.
        :return: seaborn.FacetGrid object with the plot
        """

        self._check_sample_df_column(column_to_plot)

        height = 3 * (len(self.sample_df.cycle_nr.unique()) if by_cycle else 1) + 2
        aspect = round(len(self.sample_df.name.unique()) * 0.35) / height

        args = {'row': 'cycle_nr'} if by_cycle else {}

        grid = sns.catplot(x = 'name', y = column_to_plot, data = self.sample_df, kind = 'violin', height = height, aspect = aspect, **args)

        for ax in grid.axes.flat:

            ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)

        return grid


    def plot_compound_heatmap(
            self,
            column_to_plot: str,
            annotation_df: pd.DataFrame = None,
            annotation_column: str = "significant",
            by_cycle: bool = False,
            **kwargs
        ) -> sns.FacetGrid:
        """
        Plots a heatmap to visualize the different combinations
        :param column_to_plot: Name of the column to extract values from
        :param annotation_df: pd.DataFrame grouped by column, compound_a and
            compound_b for annotation
        :param annotation_column: Which column in annotation_df to use for
            the annotation
        :param by_cycle: Produce separate plots for each cycle.
        :return: seaborn.FacetGrid object with the plot
        """
        self._check_sample_df_column(column_to_plot)

        assert column_to_plot not in {"compound_a", "compound_b"},\
            f"You can not plot this coulumn on a heatmap: `{column_to_plot}`."

        if annotation_df is not None:

            annotation_df = annotation_df.reset_index()
            annotation_df = annotation_df.pivot(
                "compound_a",
                "compound_b",
                annotation_column,
            )
            annotation_df = (
                annotation_df.
                replace(True, "*").
                replace(False, "").
                replace(np.nan, "")
            )

        cycles = self.sample_df.cycle_nr.unique() if by_cycle else (None,)

        grid = sns.FacetGrid(
            data = self.sample_df,
            col = 'cycle_nr' if by_cycle else None,
            height = 5,
            aspect = .5 * len(cycles) if by_cycle else 1,
        )

        for i, cycle in enumerate(cycles):

            data = self.sample_df
            data = data if cycle is None else data[data.cycle_nr == cycle]
            data = data[[column_to_plot, "compound_a", "compound_b"]]
            data = data.groupby(["compound_a", "compound_b"]).mean()
            data = data.reset_index()
            data = data.pivot("compound_a", "compound_b", column_to_plot)
            data = data.reindex(
                data.isna().
                sum(axis = 1).
                sort_values(ascending = False).
                index.
                to_list()
            )
            data = data[
                data.isna().
                sum(axis = 0).
                sort_values(ascending = True).
                index.
                to_list()
            ]

            if annotation_df is not None:

                annotation_df = annotation_df.reindex_like(data)

            ax = grid.axes.flat[i]

            sns.heatmap(
                data,
                annot = annotation_df,
                fmt = "",
                ax = ax,
                **kwargs
            )

            cycle_str = ('\nCycle #%u' % cycle) if by_cycle else ''
            unit_str = 'z-score' if 'z_score' in column_to_plot else 'AU'
            ax.set_title(
                f"{column_to_plot} by combination [{unit_str}]{cycle_str}"
            )
            ax.set_ylabel("")
            ax.set_xlabel("")

            if plt.matplotlib.__version__ == '3.1.1':

                ylim = ax.get_ylim()
                ax.set_ylim(ylim[0] + .5, ylim[1] - .5)

        return grid


    def _check_sample_df_column(self, column: str):
        """
        Checks if column is in sample_df
        :param column: Column to check
        """
        try:
            assert column in self.sample_df.columns.to_list(), f"Column {column} not in the column names of sample_df ({self.sample_df.columns.to_list()}), specify a column from the column names!"
        except AssertionError:
            module_logger.critical(f"Column {column} not in the column names of sample_df ({self.sample_df.columns.to_list()}), specify a column from the column names!")
            raise


    def _media_lin_reg_norm(self):
        """
        Normalizes sample_df using media control regression

        :return: updated sample_df
        """

        if self.normalize_using_media_control_lin_reg:

            sample_df = self.sample_df

            if self.normalize_using_control:
                readout_column = "readout_per_control"
            else:
                readout_column = "readout_peak_median"

            slope, intercept, _, _, _ = self.get_media_control_lin_reg(readout_column=readout_column)

            sample_df = sample_df.assign(readout_media_norm=sample_df[readout_column] / (sample_df["start_time"] * slope + intercept))
            sample_df = sample_df.assign(readout_media_norm_z_score=stats.zscore(sample_df.readout_media_norm))

            self.sample_df = sample_df