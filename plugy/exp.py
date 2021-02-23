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
import collections
import importlib as imp
import traceback
import typing

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


@dataclass(init = False)
class PlugExperiment(object):
    config: PlugyConfig = field(default_factory = PlugyConfig)
    ignore_qc_result: bool = None
    run: bool = None
    init: bool = None
    plugs: bool = None
    samples: bool = None
    qc: bool = None
    analysis: bool = None
    font_scale: typing.Union[float, int] = 2
    scatter_dot_size: typing.Union[float, int] = 10


    def __init__(
            self,
            config: PlugyConfig = None,
            ignore_qc_result: bool = None,
            run: bool = None,
            init: bool = None,
            plugs: bool = None,
            qc: bool = None,
            analysis: bool = None,
            **kwargs
        ):

        self.config = config if config else PlugyConfig(**kwargs)

        for attr in (
            'run', 'init', 'plugs', 'qc', 'analysis', 'ignore_qc_result'
        ):

            from_call = locals()[attr]
            value = (
                getattr(self.config, attr)
                    if from_call is None else
                from_call
            )

            setattr(self, attr, value)


    def __post_init__(self):
        module_logger.info('Initializing PlugExperiment')
        module_logger.debug('Configuration:')
        for k, v in self.config.__dict__.items():
            module_logger.debug(f"{k}: {v}")

        self._set_workflow_param()

        if self.run:

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


    def main(
            self,
            init: bool = None,
            plugs: bool = None,
            samples: bool = None,
            qc: bool = None,
            analysis: bool = None,
        ):

        steps = collections.OrderedDict()

        for attr in ('init', 'plugs', 'samples', 'qc', 'analysis'):

            call = locals()[attr]
            instance = getattr(self, attr)

            steps[attr] = call if call is not None else instance

        if not any(steps.values()):

            if self.config.has_samples_cycles:

                steps['qc'] = True
                steps['analysis'] = True

            elif self.config.has_barcode:

                steps['samples'] = True

            else:

                steps['plugs'] = True

        if steps['analysis'] or steps['qc']:

            steps['samples'] = True

        if steps['samples']:

            steps['plugs'] = True

        if steps['plugs']:

            steps['init'] = True

        module_logger.debug('Main: %s' % misc.dict_str(steps))

        for step, enabled in steps.items():

            if enabled:

                getattr(self, '_%s' % step)()


    def _init(self):

        self.seaborn_setup()
        self.load()


    def _plugs(self):

        self.detect_plugs()


    def _samples(self):

        self.detect_samples()


    def _qc(self):

        self.quality_control()


    def _analysis(self):

        self.drug_combination_analysis()
        self.close_figures()


    def _set_workflow_param(self):

        for attr in ('ignore_qc_result', 'run', 'init', 'plugs', 'samples'):

            here = getattr(self, attr)
            from_config = getattr(self.config, attr)

            setattr(
                self,
                attr,
                here if here is not None else from_config
            )


    def seaborn_setup(self, font_scale = None):


        sns.set_context(
            self.config.seaborn_context,
            font_scale = font_scale or self.font_scale,
            rc = self.config.seaborn_context_dict,
        )
        sns.set_style(
            self.config.seaborn_style,
            rc = self.config.seaborn_style_dict,
        )


    def load(self):

        self.load_channel_map()
        self.load_plug_sequence()
        self.load_pmt_data()


    def load_channel_map(self):

        self.channel_map = (

            ChannelMap(self.config.channel_file)

            if self.config.channel_file else

            None

        )


    def load_plug_sequence(self):

        self.plug_sequence = (

            PlugSequence.from_csv_file(
                self.config.seq_file,
                allow_lt4_valves = self.config.allow_lt4_valves,
            )

            if self.config.seq_file else

            None

        )


    def load_pmt_data(self):

        self.pmt_data = (

            PmtData(
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

            if self.config.pmt_file else

            None

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
            heatmap_override_scale = self.config.heatmap_override_scale,
            heatmap_override_second_scale =
                self.config.heatmap_override_second_scale,
            palette = self.config.palette,
            font_scale = self.font_scale,
            scatter_dot_size = self.scatter_dot_size,
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

        return None

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
        plt.clf()

        module_logger.info(f"Plotted PMT data to {png_path}")


    def plot_plug_sequence(self):
        """
        Creates a compact plot of the detected peaks: x axis is the sequence
        of the peaks, y axis is the median intensity of the channels.
        """

        qc_dir = self.ensure_qc_dir()

        width = len(self.pmt_data.peak_df) / 20 + 3
        fig, ax = plt.subplots(figsize=(width, 3))
        self.pmt_data.plot_peak_sequence(ax)
        fig.tight_layout()

        path = qc_dir.joinpath(
            f"plug_sequence.{self.config.figure_export_file_type}"
        )
        fig.savefig(path)
        plt.clf()

        module_logger.info(f"Plotted plug sequence to {path}")


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
        norm_df = self.plug_data.sample_df.assign(
            norm_barcode = (
                self.plug_data.sample_df.barcode_peak_median /
                barcode_mean
            ),
            norm_control = (
                self.plug_data.sample_df.control_peak_median /
                control_mean
            )
        )

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
            color = self.config.palette[0],
        )
        axes.set_ylabel('Number of samples')
        axes.set_xlabel('Plug count deviation')

        return axes


    def quality_control(self):
        """
        Produces multiple QC plots and metrics to evaluate the technical
        quality of the PlugExperiment

        :return:
            True if quality is sufficient, False otherwise
        """

        self.seaborn_setup()

        qc_issues = []
        qc_dir = self.ensure_qc_dir()

        # Plotting media control readout over experiment time
        media_control_fig, media_control_ax = plt.subplots(
            ncols = 2,
            figsize = (18, 9),
        )
        media_control_ax[0] = (
            self.plug_data.plot_media_control_evolution(
                axes = media_control_ax[0],
            )
        )
        media_control_ax[1] = (
            self.plug_data.plot_media_control_evolution(
                axes = media_control_ax[1],
                by_sample = True,
            )
        )

        media_control_fig.tight_layout()

        if self.config.plot_git_caption:
            misc.add_git_hash_caption(media_control_fig)

        media_control_fig.savefig(
            qc_dir.joinpath(
                f"fs_media_control.{self.config.figure_export_file_type}"
            )
        )
        plt.clf()

        self.plot_pmt_data()

        # Plotting plug numbers
        plug_count_hist_fig, plug_count_hist_ax = plt.subplots()
        plug_count_hist_ax = self.plot_plug_count_hist(
            axes = plug_count_hist_ax,
        )

        plug_count_hist_fig.tight_layout()
        if self.config.plot_git_caption:
            misc.add_git_hash_caption(plug_count_hist_fig)
        plug_count_hist_fig.savefig(
            qc_dir.joinpath(
                f"plug_count_hist.{self.config.figure_export_file_type}"
            )
        )
        plt.clf()

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

        contamination_hist_fig, contamination_hist_ax = plt.subplots(
            figsize = (8, 6.2),
        )
        sns.histplot(
            contamination,
            ax = contamination_hist_ax,
            color = self.config.palette[0],
            stat = 'density',
        )
        sns.kdeplot(
            contamination,
            ax = contamination_hist_ax,
            color = self.config.palette[1],
            linewidth = 2,
        )
        contamination_hist_ax.set_title("Relative barcode contamination")
        contamination_hist_ax.set_xlabel(
            "Barcode signal in sample plugs\n"
            r"$\left[\frac{\overline{barcode_{median}}}"
            r"{\overline{control_{median}}}\right]$"
        )
        contamination_hist_ax.set_ylabel("Number of sample plugs")
        misc.add_git_hash_caption(contamination_hist_fig)
        contamination_hist_fig.tight_layout()
        contamination_hist_fig.savefig(
            qc_dir.joinpath(
                f"contamination_hist.{self.config.figure_export_file_type}"
            )
        )
        plt.clf()

        contamination_fig, contamination_ax = plt.subplots(
            2, 3,
            sharex = "all",
            sharey = "all",
            figsize = (18, 12),
        )

        for idx_y, channel in enumerate(
            ["readout_peak_median", "control_peak_median"]
        ):
            contamination_ax[idx_y][2] = self.plug_data.plot_contamination(
                channel_x = "barcode_peak_median",
                channel_y = channel,
                hue = "start_time",
                filtered = True,
                axes = contamination_ax[idx_y][2],
            )
            contamination_ax[idx_y][2].set_title("Filtered")
            contamination_ax[idx_y][2].get_legend().set_title('Time [s]')

            for idx_x, hue in enumerate(["start_time", "barcode"]):

                contamination_ax[idx_y][idx_x] = (
                    self.plug_data.plot_contamination(
                        channel_x = "barcode_peak_median",
                        channel_y = channel,
                        hue = hue,
                        filtered = False,
                        axes = contamination_ax[idx_y][idx_x],
                    )
                )

                contamination_ax[idx_y][idx_x].set_title("Unfiltered")

                leg_title = 'Time [s]' if hue == 'start_time' else 'Plug type'
                legend = contamination_ax[idx_y][idx_x].get_legend()
                legend.set_title(leg_title)

                if hue == 'barcode':
                    for t, l in zip(legend.texts, ['Sample', 'Barcode']):
                        t.set_text(l)

        if self.config.plot_git_caption:

            misc.add_git_hash_caption(contamination_fig)

        contamination_fig.tight_layout()
        contamination_fig.savefig(
            qc_dir.joinpath(
                f"contamination.{self.config.figure_export_file_type}"
            )
        )
        plt.clf()

        self.plot_control()
        self.plot_sample_cycles()

        plt.close('all')

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
            length_bias_plot.fig.savefig(
                qc_dir.joinpath(
                    f"length_bias.{self.config.figure_export_file_type}"
                )
            )
            plt.clf()
        except:
            traceback.print_exc(file = sys.stdout)
            module_logger.error("Failed to plot length bias")


    def plot_control(self):

        self.seaborn_setup(font_scale = self.config.font_scale * .7)

        qc_dir = self.ensure_qc_dir()

        # Plotting control
        control_fig = plt.figure(
            figsize = (20, 10),
            constrained_layout = False,
        )
        control_fig_gs = control_fig.add_gridspec(nrows = 2, ncols = 4)
        control_ax_sample_dist = control_fig.add_subplot(control_fig_gs[0, :])
        control_ax_control_regression = (
            control_fig.add_subplot(control_fig_gs[1, 0])
        )
        control_ax_cycle_dist = control_fig.add_subplot(control_fig_gs[1, 1])
        control_ax_readout_correlation = (
            control_fig.add_subplot(control_fig_gs[1, 2])
        )

        control_ax_control_regression = (
            self.plug_data.plot_control_regression(
                control_ax_control_regression
            )
        )
        control_ax_cycle_dist = self.plug_data.plot_control_cycle_dist(
            control_ax_cycle_dist
        )
        control_ax_sample_dist = self.plug_data.plot_control_sample_dist(
            control_ax_sample_dist
        )
        control_ax_readout_correlation = (
            self.plug_data.plot_control_readout_correlation(
                control_ax_readout_correlation
            )
        )

        grid = self.plug_data.plot_compound_heatmap(
            column_to_plot = 'control_peak_median',
        )

        control_fig.axes[-1] = grid.axes.flat[0]

        control_fig.tight_layout()
        if self.config.plot_git_caption:
            misc.add_git_hash_caption(control_fig)
        control_fig.savefig(
            qc_dir.joinpath(
                f"control_fluorescence.{self.config.figure_export_file_type}"
            )
        )
        plt.clf()

        self.seaborn_setup()


    def plot_sample_cycles(self):

        self.seaborn_setup(font_scale = self.font_scale * .7)

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
            plt.clf()
        except:
            traceback.print_exc(file = sys.stdout)
            module_logger.error("Failed to plot sample cycles")

        self.seaborn_setup()


    def calculate_statistics(self) -> pd.DataFrame:
        """
        Calculates statistics
        """
        module_logger.info("Calculating statistics")
        media_data = self.plug_data.get_media_control_data()
        compound_data = self.plug_data.sample_df[
            ~self.plug_data.sample_df.isin(media_data)
        ].dropna()

        group_columns = ["compound_a", "compound_b", "name"]
        sample_stats = compound_data.groupby(by = group_columns)[
            self.config.readout_analysis_column
        ].agg([np.mean, np.std])

        p_values = list()

        for combination, values in compound_data.groupby(by = group_columns):

            p_values.append(
                stats.ranksums(
                    x = values[self.config.readout_analysis_column],
                    y = media_data[self.config.readout_analysis_column]
                )[1]
            )

        sample_stats = sample_stats.assign(pval = p_values)
        significance, p_adjusted, _, alpha_corr_bonferroni = (
            statsmod.multipletests(
                pvals = sample_stats.reset_index().pval,
                alpha = self.config.alpha,
                method = "bonferroni",
            )
        )
        sample_stats = sample_stats.assign(
            p_adjusted = p_adjusted,
            significant = significance,
        )

        # Renaming columns to avoid shadowing mean function
        sample_stats.columns = [
            "mean_z_score",
            "std_z_score",
            "pval",
            "p_adjusted",
            "significant",
        ]

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

        grid = self.plug_data.plot_compound_violins(
            column_to_plot = self.config.readout_analysis_column,
            by_cycle = by_cycle,
        )

        ax = grid.axes[0][0]
        # Getting y coordinates for asterisk from axis dimensions
        y_max = ax.axis()[3]

        # Labelling significant samples
        statistics = self.sample_statistics.reset_index()
        statistics = statistics.drop(["compound_a", "compound_b"], axis = 1)
        statistics = statistics.set_index("name")
        statistics = statistics.reindex(
            self.plug_data.sample_df.name.unique()
        )

        for idx, sample in enumerate(self.plug_data.sample_df.name.unique()):
            if sample != "Cell Control":
                if statistics.significant[idx]:
                    ax.annotate(
                        "*",
                        xy = (idx, y_max),
                        xycoords = "data",
                        textcoords = "data",
                        ha = "center",
                    )

        if not by_cycle:

            ax.set_title("Caspase activity z-scores", pad = 20)

        grid.fig.tight_layout()

        if self.config.plot_git_caption:
            misc.add_git_hash_caption(grid.fig)

        path = self.config.result_dir.joinpath(
            f"drug_comb_z_violins{'_by-cycle' if by_cycle else ''}."
            f"{self.config.figure_export_file_type}"
        )
        module_logger.info(f"Saving violin plots to {path}")
        grid.savefig(path)
        plt.clf()


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
        plt.clf()


    def close_figures(self):

        plt.close('all')