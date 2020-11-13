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

import sys
import logging
import pathlib as pl
import importlib as imp
import traceback

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.multitest as statsmod

import matplotlib.pyplot as plt
import seaborn as sns

from .data.config import PlugyConfig
from .data.bd import ChannelMap, PlugSequence
from .data.pmt import PmtData
from .data.plug import PlugData
from . import misc as misc

from dataclasses import dataclass, field

module_logger = logging.getLogger("plugy.data.exp")


@dataclass
class PlugExperiment(object):
    config: PlugyConfig = field(default_factory = PlugyConfig)
    ignore_qc_result: bool = False

    def __post_init__(self):
        module_logger.info('Initializing PlugExperiment')
        module_logger.debug('Configuration:')
        for k, v in self.config.__dict__.items():
            module_logger.debug(f"{k}: {v}")

        self.ignore_qc_result = (
            self.ignore_qc_result or
            self.config.ignore_qc_result
        )

        self.main()


    def reload(self):
        """
        Reloads the object from the module level.
        """

        modname = self.__class__.__module__
        mod = __import__(modname, fromlist = [modname.split('.')[0]])
        imp.reload(mod)
        new = getattr(mod, self.__class__.__name__)
        setattr(self, '__class__', new)


    def main(self):

        if self.config.run or self.config.plugs or self.config.init:

            self.setup()
            self.load()

        if self.config.run or self.config.plugs:

            self.detect_plugs()

        if self.config.run or self.config.samples:

            self.detect_samples()

        if self.config.run:

            self.qc()
            self.drug_combination_analysis()
            self.close_figures()


    def setup(self):

        self.check_config()

        # if not self.config.result_base_dir.exists():
        #     self.config.result_base_dir.mkdir()

        # assert \
        #     not self.config.result_dir.exists(), \
        #     (
        #         f"Automatically generated result directory name already exists"
        #         f" {self.config.result_dir.name}, "
        #         f"please retry in a couple of seconds"
        #     )

        # self.config.result_dir.mkdir()

        sns.set_context(self.config.seaborn_context)
        sns.set_style(self.config.seaborn_style)


    def load(self):

        self.channel_map = ChannelMap(self.config.channel_file)
        self.plug_sequence = PlugSequence.from_csv_file(
            self.config.seq_file,
            allow_lt4_valves = self.config.allow_lt4_valves,
        )

        self.pmt_data = PmtData(
            input_file = self.config.pmt_file,
            acquisition_rate = self.config.acquisition_rate,
            cut = self.config.cut,
            correct_acquisition_time = self.config.correct_acquisition_time,
            channels = self.config.channels,
            ignore_channels = self.config.ignore_channels,
            fake_gains = self.config.fake_gains,
            fake_gain_default = self.config.fake_gain_default,
            fake_gain_adaptive = self.config.fake_gain_adaptive,
            barcode_raw_threshold=self.config.barcode_raw_threshold,
            peak_min_threshold = self.config.peak_min_threshold,
            peak_max_threshold = self.config.peak_max_threshold,
            peak_min_distance = self.config.peak_min_distance,
            peak_min_prominence = self.config.peak_min_prominence,
            peak_max_prominence = self.config.peak_max_prominence,
            peak_min_width = self.config.peak_min_width,
            peak_max_width = self.config.peak_max_width,
            width_rel_height = self.config.width_rel_height,
            merge_peaks_distance = self.config.merge_peaks_distance,
            merge_peaks_by = self.config.merge_peaks_by,
            config = self.config,
        )


    def detect_plugs(self):

        self.plug_data = PlugData(
            pmt_data = self.pmt_data,
            plug_sequence = self.plug_sequence,
            channel_map = self.channel_map,
            auto_detect_cycles = self.config.auto_detect_cycles,
            n_bc_adjacent_discards = self.config.n_bc_adjacent_discards,
            min_end_cycle_barcodes = self.config.min_end_cycle_barcodes,
            min_between_samples_barcodes =
                self.config.min_between_samples_barcodes,
            min_plugs_in_sample = self.config.min_plugs_in_sample,
            normalize_using_control = self.config.normalize_using_control,
            normalize_using_media_control_lin_reg =
                self.config.normalize_using_media_control_lin_reg,
            has_barcode = self.config.has_barcode,
            has_samples_cycles = self.config.has_samples_cycles,
            samples_per_cycle = self.config.samples_per_cycle,
            heatmap_second_scale = self.config.heatmap_second_scale,
            config = self.config,
        )


    def detect_samples(self):

        try:

            self.plug_data.detect_samples()

        except AssertionError:
            # In case labelling does not work because
            # the number of samples diverges from the expected number.
            # Plotting pmt data as a fallback

            module_logger.error(
                'Could not find any cycle with the expected number of '
                'samples. Potting the PMT data and exiting.'
            )

            self.plot_pmt_data()

            raise

        self.sample_data = self.get_sample_data()
        self.append_experiments()
        self.sample_statistics = self.calculate_statistics()


    def append_experiments(self, experiments: list = None):
        """
        Merges one or more other experiments by concatenating the plug and
        sample data frames. The time and cycle values will be modified in a
        way that the merged experiments follow each other without overlap.
        """

        experiments = (
            self.config.append
                if experiments is None else
            experiments
        )

        if not isinstance(experiments, (tuple, list)):

            experiments = tuple(experiments)

        for exp in experiments:

            if not isinstance(exp, tuple):

                exp = (exp,)

            self.plug_data.append(*exp)


    def plot_pmt_data(self):

        qc_dir = self.ensure_qc_dir()

        pmt_overview_fig, pmt_overview_ax = plt.subplots(figsize=(300, 10))
        self.pmt_data.plot_pmt_data(pmt_overview_ax)

        self.plug_data.highlight_plugs(
            axes = pmt_overview_ax,
            below_peak = False,
        )
        self.plug_data.highlight_samples(axes = pmt_overview_ax)
        self.plug_data.pmt_plot_add_thresholds(axes = pmt_overview_ax)

        if self.config.plot_git_caption:
            misc.add_git_hash_caption(pmt_overview_fig)

        pmt_overview_fig.tight_layout()
        png_path = qc_dir.joinpath(f"pmt_overview.png")
        pmt_overview_fig.savefig(png_path)

        module_logger.info(f"Plotted PMT data to {png_path}")


    def get_sample_data(self) -> pd.DataFrame:
        """
        Generates sample_data df that groups plugs for each sample and cycle calculating their median
        and how much their plug number diverges from the expected as produced by the BD using the PlugSequence
        :return: pd.DataFrame grouped by cycle_nr and sample_nr with the medians and additional plug count divergence
        """
        sample_data = self.plug_data.sample_df.groupby(["cycle_nr", "sample_nr"], as_index = False).median()
        divergence = self.get_plug_count_divergence()

        sample_data = sample_data.set_index(["cycle_nr", "sample_nr"]).join(divergence.set_index(["cycle_nr", "sample_nr"]))
        return sample_data


    def get_plug_count_divergence(self):
        """
        Compares generated and called number of plugs per sample and returns the difference
        :return: pd.Series with the divergence per sample
        """
        divergences = list()
        plug_sample_sequence = self.plug_sequence.get_samples(channel_map = self.channel_map).sequence
        for (cycle_nr, sample_nr), group in self.plug_data.sample_df.groupby(["cycle_nr", "sample_nr"]):
            divergence = (self.config.n_bc_adjacent_discards * 2) + len(group) - plug_sample_sequence[sample_nr].n_replicates
            divergences.append([cycle_nr, sample_nr, divergence])

        return pd.DataFrame(divergences, columns = ["cycle_nr", "sample_nr", "plug_count_divergence"])


    def get_contamination(self) -> pd.Series:
        """
        Normalizes filtered plug data to the means of the unfiltered data.
        :return: Series with the relative contamination of the plugs in sample_df
        """
        used_plugs = self.plug_data.plug_df.loc[self.plug_data.plug_df.cycle_nr.isin(self.plug_data.sample_df.cycle_nr.unique()) & self.plug_data.plug_df.barcode == 1]
        barcode_mean = used_plugs.loc[used_plugs.barcode].barcode_peak_median.mean()
        control_mean = self.plug_data.sample_df.control_peak_median.mean()
        norm_df = self.plug_data.sample_df.assign(norm_barcode = self.plug_data.sample_df.barcode_peak_median / barcode_mean,
                                                  norm_control = self.plug_data.sample_df.control_peak_median / control_mean)

        return norm_df.norm_barcode / norm_df.norm_control


    def plot_plug_count_hist(self, axes: plt.Axes):
        """
        Plots the distribution of plug number divergence from the expected
        number by sample

        :param axes: plt.Axes object to draw on

        :return: plt.Axes object with the plot
        """

        axes = sns.countplot(
            x = self.sample_data.plug_count_divergence,
            ax = axes,
        )
        axes.set_ylabel('Counts')
        axes.set_xlabel('Plug count divergence per sample')

        return axes


    def check_config(self):
        """
        Checks if pmt_file, seq_file and config_file exist as specified in the PlugyConfig
        """
        files_to_check = {"pmt_file": self.config.pmt_file, "seq_file": self.config.seq_file, "config_file": self.config.channel_file}
        errors = list()

        for name, file in files_to_check.items():
            try:
                # If file is per default None
                assert isinstance(file, pl.Path), f"{name} was not specified as pathlib.Path object (is of {type(file)}) in PlugyConfig but is mandatory for PlugExperiment"

                # If file exists
                try:
                    file.exists(), f"{name} specified in PlugyConfig {file.absolute()} does not exist but is mandatory for PlugExperiment"
                except AssertionError as error:
                    errors.append(error)

            except AssertionError as error:
                errors.append(error)

        if len(errors) > 0:
            for error in errors:
                module_logger.critical(error.args[0])

            raise AssertionError("One or more file paths are not properly specified, see the log for more information!")


    def qc(self):
        """
        Produces multiple QC plots and metrics to evaluate the technical quality of the PlugExperiment
        :return: True if quality is sufficient, False otherwise
        """
        qc_issues = []
        qc_dir = self.ensure_qc_dir()

        # Plotting media control readout over experiment time
        media_control_fig, media_control_ax = plt.subplots(ncols = 2, figsize = (20, 10))
        media_control_ax[0] = self.plug_data.plot_media_control_evolution(axes = media_control_ax[0])
        media_control_ax[1] = self.plug_data.plot_media_control_evolution(axes = media_control_ax[1], by_sample = True)

        media_control_fig.tight_layout()
        if self.config.plot_git_caption:
            misc.add_git_hash_caption(media_control_fig)
        media_control_fig.savefig(qc_dir.joinpath(f"fs_media_control.{self.config.figure_export_file_type}"))

        self.plot_pmt_data()

        # Plotting plug numbers
        plug_count_hist_fig, plug_count_hist_ax = plt.subplots()
        plug_count_hist_ax = self.plot_plug_count_hist(axes = plug_count_hist_ax)

        plug_count_hist_fig.tight_layout()
        if self.config.plot_git_caption:
            misc.add_git_hash_caption(plug_count_hist_fig)
        plug_count_hist_fig.savefig(qc_dir.joinpath(f"plug_count_hist.{self.config.figure_export_file_type}"))

        self.plot_length_bias()

        # Plotting contamination
        contamination = self.get_contamination()
        if contamination.mean() > self.config.contamination_threshold:
            msg = (
                f"Contamination above threshold ({contamination.mean()} > "
                f"{self.config.contamination_threshold})"
            )
            module_logger.warning(msg)
            qc_issues.append(msg)

        contamination_hist_fig, contamination_hist_ax = plt.subplots()
        sns.histplot(contamination, kde = True, ax = contamination_hist_ax)
        contamination_hist_ax.set_title("Relative barcode contamination")
        contamination_hist_ax.set_xlabel(r"Relative contamination $\left[\frac{\overline{barcode_{median}}}{\overline{control_{median}}}\right]$")
        contamination_hist_ax.set_ylabel("Counts")
        misc.add_git_hash_caption(contamination_hist_fig)
        contamination_hist_fig.tight_layout()
        contamination_hist_fig.savefig(qc_dir.joinpath(f"contamination_hist.{self.config.figure_export_file_type}"))

        contamination_fig, contamination_ax = plt.subplots(2, 3, sharex = "all", sharey = "all", figsize = (30, 20))
        for idx_y, channel in enumerate(["readout_peak_median", "control_peak_median"]):
            contamination_ax[idx_y][2] = self.plug_data.plot_contamination(channel_x = "barcode_peak_median", channel_y = channel, hue = "start_time", filtered = True, axes = contamination_ax[idx_y][2])
            contamination_ax[idx_y][2].set_title("Filtered")
            for idx_x, hue in enumerate(["start_time", "barcode"]):
                contamination_ax[idx_y][idx_x] = self.plug_data.plot_contamination(channel_x = "barcode_peak_median", channel_y = channel, hue = hue, filtered = False, axes = contamination_ax[idx_y][idx_x])
                contamination_ax[idx_y][idx_x].set_title("Unfiltered")

        if self.config.plot_git_caption:
            misc.add_git_hash_caption(contamination_fig)
        contamination_fig.tight_layout()
        contamination_fig.savefig(qc_dir.joinpath(f"contamination.{self.config.figure_export_file_type}"))

        # Plotting control
        control_fig = plt.figure(figsize = (40, 20), constrained_layout = False)
        control_fig_gs = control_fig.add_gridspec(nrows = 2, ncols = 4)
        control_ax_sample_dist = control_fig.add_subplot(control_fig_gs[0, :])
        control_ax_control_regression = control_fig.add_subplot(control_fig_gs[1, 0])
        control_ax_cycle_dist = control_fig.add_subplot(control_fig_gs[1, 1])
        control_ax_readout_correlation = control_fig.add_subplot(control_fig_gs[1, 2])
        control_ax_control_heatmap = control_fig.add_subplot(control_fig_gs[1, 3])

        control_ax_control_regression = self.plug_data.plot_control_regression(control_ax_control_regression)
        control_ax_cycle_dist = self.plug_data.plot_control_cycle_dist(control_ax_cycle_dist)
        control_ax_sample_dist = self.plug_data.plot_control_sample_dist(control_ax_sample_dist)
        control_ax_readout_correlation = self.plug_data.plot_control_readout_correlation(control_ax_readout_correlation)

        grid = self.plug_data.plot_compound_heatmap(
            column_to_plot = 'control_peak_median',
            vmin = self.config.drug_comb_heatmap_scale_min,
            vmax = self.config.drug_comb_heatmap_scale_max,
        )

        control_fig.axes[-1] = grid.axes.flat[0]

        control_fig.tight_layout()
        if self.config.plot_git_caption:
            misc.add_git_hash_caption(control_fig)
        control_fig.savefig(qc_dir.joinpath(f"control_fluorescence.{self.config.figure_export_file_type}"))

        self.plot_sample_cycles()

        qc_successful = not qc_issues
        qc_fail_msg = (
            "Quality control failed due to the following reasons: "
            f"{', '.join(qc_issues)}. "
            "See also the QC plots for more information. "
            "In case you still want to continue, you can "
            "set the `ignore_qc_result` config parameter to True."
        )

        if qc_issues:
            module_logger.critical(qc_fail_msg)
        else:
            module_logger.info("Quality control successful")

        if not self.ignore_qc_result:

            assert qc_successful, qc_fail_msg

        return qc_successful


    def ensure_qc_dir(self):

        qc_dir = self.config.result_dir.joinpath("qc")

        if not qc_dir.exists():

            qc_dir.mkdir()

        return qc_dir


    def plot_length_bias(self):

        # Plotting length bias
        try:
            qc_dir = self.ensure_qc_dir()
            length_bias_plot = self.plug_data.plot_length_bias(col_wrap = 8)
            if self.config.plot_git_caption:
                misc.add_git_hash_caption(length_bias_plot.fig)
            length_bias_plot.fig.tight_layout()
            length_bias_plot.fig.savefig(qc_dir.joinpath(f"length_bias.{self.config.figure_export_file_type}"))
        except:
            traceback.print_exc(file = sys.stdout)
            module_logger.error("Failed to plot length bias")


    def plot_sample_cycles(self):

        # Plotting PMT overview
        try:
            qc_dir = self.ensure_qc_dir()
            sample_cycle_fig, sample_cycle_ax = (
                self.plug_data.plot_sample_cycles()
            )
            if self.config.plot_git_caption:
                misc.add_git_hash_caption(sample_cycle_fig)

            sample_cycle_fig.savefig(
                qc_dir.joinpath("sample_cycle_overview.png")
            )
        except:
            traceback.print_exc(file = sys.stdout)
            module_logger.error("Failed to plot sample cycles")


    def calculate_statistics(self) -> pd.DataFrame:
        """
        Calculates statistics
        """
        module_logger.info("Calculating statistics")
        media_data = self.plug_data.get_media_control_data()
        compound_data = self.plug_data.sample_df[~self.plug_data.sample_df.isin(media_data)].dropna()

        group_columns = ["compound_a", "compound_b", "name"]
        sample_stats = compound_data.groupby(by = group_columns)[self.config.readout_analysis_column].agg([np.mean, np.std])

        p_values = list()
        for combination, values in compound_data.groupby(by = group_columns):
            p_values.append(stats.ranksums(x = values[self.config.readout_analysis_column],
                                           y = media_data[self.config.readout_analysis_column])[1])

        sample_stats = sample_stats.assign(pval = p_values)
        significance, p_adjusted, _, alpha_corr_bonferroni = statsmod.multipletests(pvals = sample_stats.reset_index().pval, alpha = self.config.alpha, method = "bonferroni")
        sample_stats = sample_stats.assign(p_adjusted = p_adjusted, significant = significance)

        # Renaming columns to avoid shadowing mean function
        sample_stats.columns = ["mean_z_score", "std_z_score", "pval", "p_adjusted", "significant"]

        # # Reindex based on sample_df from plug.PlugData object
        # sample_stats = sample_stats.reindex(self.plug_data.sample_df.name.unique())
        return sample_stats


    def drug_combination_analysis(self):
        """
        Analyzes drug combinations and produces result plots
        """
        module_logger.info("Running drug combination analysis")

        self.z_scores_violin_plot()
        self.z_scores_violin_plot(by_cycle = True)
        self.z_scores_heatmap()
        self.z_scores_heatmap(by_cycle = True)


    def z_scores_violin_plot(self, by_cycle: bool = False):

        # Overview violin plot with z-scores

        grid = self.plug_data.plot_compound_violins(column_to_plot=self.config.readout_analysis_column, by_cycle = by_cycle)

        ax = grid.axes[0][0]
        # Getting y coordinates for asterisk from axis dimensions
        y_max = ax.axis()[3]

        # Labelling significant samples
        statistics = self.sample_statistics.reset_index()
        statistics = statistics.drop(["compound_a", "compound_b"], axis = 1)
        statistics = statistics.set_index("name")
        statistics = statistics.reindex(self.plug_data.sample_df.name.unique())

        for idx, sample in enumerate(self.plug_data.sample_df.name.unique()):
            if sample != "Cell Control":
                if statistics.significant[idx]:
                    ax.annotate("*", xy = (idx, y_max), xycoords = "data", textcoords = "data", ha = "center")

        if not by_cycle:

            ax.set_title("Caspase activity z-scores", pad = 20)

        grid.fig.tight_layout()

        if self.config.plot_git_caption:
            misc.add_git_hash_caption(grid.fig)

        path = self.config.result_dir.joinpath(f"drug_comb_z_violins{'_by-cycle' if by_cycle else ''}.{self.config.figure_export_file_type}")
        module_logger.info(f"Saving violin plots to {path}")
        grid.savefig(path)


    def z_scores_heatmap(self, by_cycle: bool = False):

        # Overview heatmap of z-scores

        # Labelling significant samples
        statistics = self.sample_statistics.reset_index()
        statistics = statistics.drop("name", axis = 1)
        statistics = statistics.set_index(["compound_a", "compound_b"])
        statistics = statistics.significant

        grid = self.plug_data.plot_compound_heatmap(
            column_to_plot = self.config.readout_analysis_column,
            annotation_df = statistics,
            vmin = self.config.drug_comb_heatmap_scale_min,
            vmax = self.config.drug_comb_heatmap_scale_max,
            by_cycle = by_cycle,
        )

        grid.fig.tight_layout()

        if self.config.plot_git_caption:

            misc.add_git_hash_caption(grid.fig)

        path = self.config.result_dir.joinpath(
            f"drug_comb_z_heatmap{'_by-cycle' if by_cycle else ''}."
            f"{self.config.figure_export_file_type}"
        )
        module_logger.info(f"Saving heatmap(s) to {path}")
        grid.savefig(path)


    def close_figures(self):

        plt.close('all')