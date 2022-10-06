#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# This file is part of the `plugy` python module
#
# Copyright
# 2018-2022
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

from typing import Iterable, Optional
from numbers import Number

import sys
import os
import logging
import collections
import importlib as imp
import traceback
import typing
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# to output text instead of paths
plt.rcParams['svg.fonttype'] = 'none'

from .data.config import PlugyConfig
from .data.bd import ChannelMap, PlugSequence
from .data.pmt import PmtData
from .data.plug import PlugData
from . import misc as misc


module_logger = logging.getLogger(__name__)


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

        module_logger.info('Workflow steps: %s' % misc.dict_str(steps))

        for step, enabled in steps.items():

            if enabled:

                module_logger.info('Executing step `%s`.' % step)
                getattr(self, '_%s' % step)()
                module_logger.info('Finished step `%s`.' % step)

            else:

                module_logger.info('Skipping step `%s`.' % step)


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
        self.stats_table_tsv()


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
            has_controls = self.config.has_controls,
            heatmap_second_scale = self.config.heatmap_second_scale,
            heatmap_override_scale = self.config.heatmap_override_scale,
            heatmap_override_second_scale =
                self.config.heatmap_override_second_scale,
            palette = self.config.palette,
            font_scale = self.font_scale,
            scatter_dot_size = self.scatter_dot_size,
            short_labels = self.config.short_labels,
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
        self.sample_statistics = self.stats()


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


    def plot_pmt_data(
            self,
            width: Number = 300,
            xmin: Optional[Number] = None,
            xmax: Optional[Number] = None,
            ymax: Optional[Number] = None,
            label: Optional[str] = None,
            **kwargs
        ):

        qc_dir = self.ensure_qc_dir()

        pmt_overview_fig, pmt_overview_ax = plt.subplots(
            figsize = (width, 10)
        )
        kwargs['n_x_ticks'] = kwargs.get('n_x_ticks', width / 2)
        self.pmt_data.plot_pmt_data(
            pmt_overview_ax,
            **kwargs,
        )


        pmt_overview_ax.set_xlim(xmin, xmax)
        pmt_overview_ax.set_ylim(0.0, ymax)

        self.plug_data.highlight_plugs(
            axes = pmt_overview_ax,
            below_peak = False,
        )
        self.plug_data.highlight_samples(axes = pmt_overview_ax)
        self.plug_data.pmt_plot_add_thresholds(axes = pmt_overview_ax)

        if self.config.plot_git_caption:

            misc.add_git_hash_caption(pmt_overview_fig)

        pmt_overview_fig.tight_layout()
        fmt = (
            'png'
                if self.config.pmt_overview_force_png else
            self.config.figure_export_file_type
        )
        label = f'_{label}' if label else ''
        path = qc_dir.joinpath(f"pmt_overview{label}.{fmt}")
        pmt_overview_fig.savefig(path)
        plt.clf()

        module_logger.info(f"Plotted PMT data to `{path}`.")


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

        # 1: drift of the medium only control
        self.plot_medium_control_trends()

        # 2: raw PMT data
        self.plot_pmt_data()

        # 3: per sample plug counts
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

        # 4: plug lengths (sizes)
        self.plot_length_bias()

        # 5: barcode dye contamination of sample plugs
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

        if self.config.figure_titles:

            contamination_hist_ax.set_title("Relative barcode contamination")

        xlab = "Barcode to control ratio"

        if self.config.figure_export_file_type == 'png':

            xlab += (
                r"\n$\left[\frac{\overline{barcode_{median}}}"
                r"{\overline{control_{median}}}\right]$"
            )

        contamination_hist_ax.set_xlabel(xlab)

        contamination_hist_ax.set_ylabel("Number of sample plugs")

        if self.config.plot_git_caption:

            misc.add_git_hash_caption(contamination_hist_fig)

        contamination_hist_fig.tight_layout()

        path = qc_dir.joinpath(
            f"contamination_hist.{self.config.figure_export_file_type}"
        )
        contamination_hist_fig.savefig(path)
        plt.clf()

        module_logger.info(f"Saving contamination histogram to `{path}`.")

        # 6: channel vs. channel scatterplot matrix
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

        path = qc_dir.joinpath(
            f"contamination.{self.config.figure_export_file_type}"
        )
        contamination_fig.savefig(path)
        plt.clf()

        module_logger.info(f"Saving contamination scatter plots to `{path}`.")

        # 7: composite single channel insights figures
        self.univar_overviews()

        # 8: raw data in sample vs. cycle grid
        self.plot_samples_cycles()

        # 9: grid of plug length (size) vs. channels
        self.length_grid()

        # 10: density plots of plug lengths (sizes)
        self.length_density()
        self.length_boxplot()

        # 11: density plots of plug volumes
        self.volume_density()
        self.volume_boxplot()

        # 12: violin plots of standard deviations of samples:
        # readout, readout:control and medium corrected readout:control
        self.sample_sd_violin()

        # 13: printing coefficients of variance into the log
        self.report_cv()

        # 14: heatmap matrix
        self.heatmap_matrix()

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


    def plot_medium_control_trends(self):

        if not self.plug_data.has_medium_control:

            module_logger.info(
                'Skipping medium control trends figure, looks like '
                'this experiment has no medium only samples.'
            )
            return

        qc_dir = self.ensure_qc_dir()

        # Plotting media control readout over experiment time
        media_control_fig, media_control_ax = plt.subplots(
            ncols = 2,
            figsize = (14, 7),
        )
        media_control_ax[0] = (
            self.plug_data.plot_medium_control_trends(
                ax = media_control_ax[0],
            )
        )
        media_control_ax[1] = (
            self.plug_data.plot_medium_control_trends(
                ax = media_control_ax[1],
                by_sample = True,
            )
        )

        media_control_fig.tight_layout()

        if self.config.plot_git_caption:
            misc.add_git_hash_caption(media_control_fig)

        path = qc_dir.joinpath(
            f"fs_medium_control.{self.config.figure_export_file_type}"
        )

        media_control_fig.savefig(path)
        plt.clf()

        module_logger.info('Medium control trends plotted to %s' % path)


    def plot_length_bias(self):
        """
        Creates a figure of readout signal vs. plug length by sample.
        """

        try:
            qc_dir = self.ensure_qc_dir()
            length_bias_plot = self.plug_data.plot_length_bias(col_wrap = 8)
            if self.config.plot_git_caption:
                misc.add_git_hash_caption(length_bias_plot.fig)
            length_bias_plot.fig.tight_layout()
            path = qc_dir.joinpath(
                f'length_bias.{self.config.figure_export_file_type}'
            )
            length_bias_plot.fig.savefig(path)
            plt.clf()
            module_logger.info('Plug length bias plotted to %s' % path)
        except:
            traceback.print_exc(file = sys.stdout)
            module_logger.error('Failed to plot length bias')


    def heatmap_matrix(self, **kwargs):
        """
        Creates a composite figure to overview the medians of conditions
        by cycle vs. channel or derived variable.
        """

        grid = self.plug_data.heatmap_matrix(**kwargs)

        self._plot_base(grid, 'heatmap_matrix', 'heatmap matrix')


    def univar_overview(
            self,
            var: str = None,
            var2: str = 'readout_peak_median',
            var3: str = 'control_peak_median',
        ):
        """
        Creates a composite figure to investigate one variable and discover
        potential bias or drift across time, cycles, samples, or in relation
        to another variable. It is especially suitable to see if the control
        channel shows a bias with other variables or correlation with the
        readout channel.

        Args:
            var:
                A variable name in the plug or sample data frame.
            var2:
                Another variable: it will be used only for one panel,
                a scatter plot ``var`` vs. ``var2``.
        """

        self.seaborn_setup(font_scale = self.config.font_scale * .7)

        qc_dir = self.ensure_qc_dir()

        fig = self.plug_data.univar_overview(
            var = var,
            var2 = var2,
            var3 = var3,
        )
        fig.tight_layout()

        if self.config.plot_git_caption:

            misc.add_git_hash_caption(fig)

        path = qc_dir.joinpath(
            f'{var}_overview.{self.config.figure_export_file_type}'
        )
        fig.savefig(path)
        plt.clf()

        module_logger.info(f"Saving overview plot of `{var}` to `{path}`.")

        self.seaborn_setup()


    def univar_overviews(
            self,
            variables: Iterable[str] | None = None,
            var2: str | None = None,
        ):
        """
        Calls ``univar_overview`` for each channel and the derived variables.
        It means creating a composite figure for each of these variables.
        """

        default_vars = misc.CHANNEL_VARS
        variables = misc.to_tuple(variables) or default_vars

        for var in variables:

            self.univar_overview(var = var)


    def plot_samples_cycles(
            self,
            label: Optional[str] = None,
            ymax: Optional[Number] = None,
            **kwargs
        ):
        """
        Creates a plot with raw data for the individual samples and cycles.

        Args:
            label:
                A custom label to be included in the file name. Useful if
                plotting a subset of samples.
        """

        self.seaborn_setup(font_scale = self.font_scale * .7)

        # Plotting PMT overview
        try:

            qc_dir = self.ensure_qc_dir()

            sample_cycle_fig, sample_cycle_ax = (
                self.plug_data.plot_samples_cycles(
                    ylim = (None, ymax),
                    **kwargs
                )
            )

            if self.config.plot_git_caption:

                misc.add_git_hash_caption(sample_cycle_fig)

            ftype = (
                self.config.figure_export_file_type
                    if 0 < len(kwargs.get('samples', ())) < 9 else
                'png'
            )

            label = f'_{label}' if label else ''
            fname = f'sample_cycle_overview{label}.{ftype}'
            path = qc_dir.joinpath(fname)
            sample_cycle_fig.savefig(path)
            module_logger.info(f'Saved figure to `{path}`')

            plt.clf()

        except:

            traceback.print_exc(file = sys.stdout)
            module_logger.error('Failed to plot samples vs. cycles.')

        self.seaborn_setup()


    def stats(
            self,
            cycle: int | None = None,
            extra_cols: list[str] = None,
        ) -> pd.DataFrame:
        """
        Calculates statistics for each sample: means, standard deviations,
        Wilcoxon tests, their adjusted p-values and significances.

        Args:
            cycle:
                Use only this experiment cycle. If None, all cycles will be
                used.
            extra_cols:
                Keep also these columns, calculate their means and standard
                deviations.

        Returns:
            A data frame with the sample level statistics, with compounds
            in row multi index.
        """

        return self.plug_data.stats(cycle = cycle, extra_cols = extra_cols)


    def stats_table(
            self,
            by_cycle: bool = True,
            exp_summary: bool = True,
            extra_cols: bool | list[str] = True,
            sort_by: list[str] | None = None,
            multi_index: bool = True,
        ) -> pd.DataFrame:
        """
        A large table of statistics.

        Args:
            by_cycle:
                Include statistics for each cycle.
            exp_summary:
                Include statistics for the entire experiment.
            extra_cols:
                Include extra columns, besides the defaults. Alternatively,
                a list of column names that overrides the extra columns.
            sort_by:
                Sort the conditions by these column names. For descending
                order prefix the column name with a hat (^). Cycle will
                always be the first column to sort by.
            multi_index:
                Keep conditions in multi index. If `False`, the multi index
                will be turned into regular columns.

        Return:
            A data frame of the statistics with conditions in row multi index.
        """

        args = locals()
        args.pop('self')

        return self.plug_data.stats_table(**args)


    def _stats_table_export(
            self,
            path: str | None = None,
            ftype: str = 'tsv',
            by_cycle: bool = True,
            exp_summary: bool = True,
            extra_cols: bool | list[str] = True,
            sort_by: list[str] | None = None,
            multi_index: bool = True,
            **kwargs
        ):
        """
        Export a table of statistics.

        Args:
            path:
                Path or file name. Default directory is *results*, default
                file name is *results/<input-name>_stats-table.<ftype>*.
            ftype:
                File type, such as "csv", "tsv", "xlsx", or any other type
                supported by ``pandas``.
            kwargs:
                Passed to pandas export (``to_...``) functions.
        """

        table = self.stats_table(
            by_cycle = by_cycle,
            exp_summary = exp_summary,
            extra_cols = extra_cols,
            sort_by = sort_by,
            multi_index = multi_index,
        )

        defaults = dict(
            fname_suffix = 'stats-table',
            sheet_name = f'stats__{self.pmt_data.input_file.stem}',
            label = 'large stats table',
        )
        defaults.update(kwargs)

        self._export(df = table, path = path, ftype = ftype, **defaults)


    def stats_table_tsv(
            self,
            path: str | None = None,
            by_cycle: bool = True,
            exp_summary: bool = True,
            extra_cols: bool | list[str] = True,
            sort_by: list[str] | None = None,
            **kwargs
        ):
        """
        Export a table of statistics to TSV.

        Args:
            path:
                Path or file name. Default directory is *results*, default
                file name is *results/<input-name>_stats-table.tsv*.
            kwargs:
                Passed to ``pandas.DataFrame.to_csv``.

        The rest of the arguments are described at ``stats_table``.
        """

        args = locals()
        args.pop('self')
        args.pop('kwargs')
        args['ftype'] = 'tsv'

        self._stats_table_export(**args, **kwargs)


    def stats_table_excel(
            self,
            path: str | None = None,
            by_cycle: bool = True,
            exp_summary: bool = True,
            extra_cols: bool | list[str] = True,
            sort_by: list[str] | None = None,
            multi_index: bool = True,
            **kwargs
        ):
        """
        Export a table of statistics to XLSX.

        Args:
            path:
                Path or file name. Default directory is *results*, default
                file name is *results/<input-name>_stats-table.xlsx*.
            kwargs:
                Passed to ``pandas.DataFrame.to_excel``.

        The rest of the arguments are described at ``stats_table``.
        """

        args = locals()
        args.pop('self')
        args.pop('kwargs')
        args['ftype'] = 'xlsx'

        self._stats_table_export(**args, **kwargs)


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

        self.seaborn_setup(font_scale = self.font_scale * .7)

        cycles = (
            self.plug_data.sample_df.cycle_nr.unique()
                if by_cycle else
            [None]
        )

        fig, ax = plt.subplots(
            nrows = (
                self.plug_data.sample_df.cycle_nr.nunique()
                    if by_cycle else
                1
            ),
            figsize = (
                max(
                    4,
                    round(len(self.plug_data.sample_df.name.unique()) * 0.35)
                ),
                3 * (len(cycles) if by_cycle else 1) + 2
            ),
            sharex = True,
        )

        for i, cycle in enumerate(cycles):

            this_ax = ax if len(cycles) == 1 else ax.flat[i]

            self.plug_data.plot_compound_violins(
                ax = this_ax,
                column_to_plot = self.config.readout_analysis_column,
                cycle = cycle,
            )

        this_ax = ax if len(cycles) == 1 else ax.flat[0]
        # Getting y coordinates for asterisk from axis dimensions
        y_max = this_ax.axis()[3]

        # Labelling significant samples
        statistics = self.sample_statistics.reset_index()
        statistics = statistics.drop(["compound_a", "compound_b"], axis = 1)
        statistics = statistics.set_index("name")
        statistics = statistics.reindex(
            self.plug_data.sample_df.name.unique()
        )

        for idx, sample in enumerate(self.plug_data.sample_df.name.unique()):

            if sample != "Cell Control":

                this_ax.annotate(
                    statistics.stars[idx],
                    xy = (idx, y_max),
                    xycoords = "data",
                    textcoords = "data",
                    ha = "center",
                )

        if not by_cycle:

            this_ax.set_title("Caspase activity z-scores", pad = 20)

        fig.tight_layout()

        if self.config.plot_git_caption:

            misc.add_git_hash_caption(fig)

        path = self.config.result_dir.joinpath(
            f"drug_comb_z_violins{'_by-cycle' if by_cycle else ''}."
            f"{self.config.figure_export_file_type}"
        )
        module_logger.info(f"Saving violin plots to {path}")
        fig.savefig(path)
        plt.clf()
        self.seaborn_setup()


    def z_scores_heatmap(self, by_cycle: bool = False):
        """
        Drug vs. drug heatmap showing the median readouts for each drug
        combination.
        """

        # Overview heatmap of z-scores

        # Labelling significant samples
        statistics = self.sample_statistics.reset_index()
        statistics = statistics.drop("name", axis = 1)
        statistics = statistics.set_index(["compound_a", "compound_b"])
        statistics = statistics.stars

        grid = self.plug_data.samples_heatmap(
            var = self.config.readout_analysis_column,
            annotation_df = statistics,
            by_cycle = by_cycle,
        )

        self._plot_base(
            grid,
            f"drug_comb_z_heatmap{'_by-cycle' if by_cycle else ''}",
            'drug combination heatmap',
            qc_dir = False,
        )


    def length_grid(self):
        """
        A pairs grid figure between plug lengths, fluorescent channels
        and the ratio of the readout and control channels.
        """

        grid = self.plug_data.length_grid()

        self._plot_base(grid, 'lengths_grid', 'plug length grid')


    def length_density(self):
        """
        Density plots and histograms about plug lengths. Three panels are
        created, with barcode plugs, sample plugs and both togther.
        """

        grid = self.plug_data.size_density()

        self._plot_base(grid, 'lengths_density', 'plug length density plots')


    def length_boxplot(self):
        """
        Boxplots about plug lengths. Three panels are created, with barcode
        plugs, sample plugs and both togther.
        """

        grid = self.plug_data.size_density(boxplot = True)

        self._plot_base(grid, 'lengths_boxplot', 'length boxplots')


    def volume_density(self, flow_rate: float | None = None):
        """
        Density plots and histograms about plug volumes. Three panels are
        created, with barcode plugs, sample plugs and both togther.

        Args:
            flow_rate (float): The flow rate used at the data acquisition
                in microlitres per hour.
        """

        grid = self.plug_data.size_density(
            volume = True,
            flow_rate = flow_rate,
        )

        self._plot_base(grid, 'volumes_density', 'plug volume density plots')


    def volume_boxplot(self, flow_rate: float | None = None):
        """
        Boxplots about plug volumes. Three panels are created, with barcode
        plugs, sample plugs and both togther.

        Args:
            flow_rate (float): The flow rate used at the data acquisition
                in microlitres per hour.
        """

        grid = self.plug_data.size_density(
            boxplot = True,
            volume = True,
            flow_rate = flow_rate,
        )

        self._plot_base(grid, 'volumes_boxplot', 'plug volume boxplots')


    def sample_sd_violin(self):

        grid = self.plug_data.sample_sd_violin()

        self._plot_base(grid, 'sample_sd_violin', 'sample SD violin plots')


    def report_cv(self):

        cv = self.plug_data.mean_cv()

        for cycle, ccv in cv.items():

            for var, vcv in ccv.items():

                module_logger.info(
                    'Coefficient of variance, cycle %s, %s: '
                    '%.01f (%.01f-%.01f).' % (
                        str(cycle),
                        var,
                        vcv['mean'],
                        vcv['ci_low'],
                        vcv['ci_high'],
                    )
                )


    def _plot_base(
            self,
            fig,
            fname,
            log_msg = None,
            qc_dir = True,
        ):
        """
        Many plotting functions contain mostly repeated code. Here we attempt
        to contain these common parts in one function.

        Args:
            fig: A matplotlib Figure object, or some higher level object
                carrying a Figure object (such as a grid) in its `fig`
                attribute.
            fname (str): The file name, without extension.
            log_msg (str): How to call this figure in the log messages.
            qc_dir (bool): Save the plots in the `qc` subdirectory instead of
                directly into the `results` directory.
        """

        parent = None

        if not isinstance(fig, mpl.figure.Figure):

            if hasattr(fig, 'fig'):

                parent = fig
                fig = parent.fig

            else:

                raise ValueError(
                    'Don\'t know how to process a `%s` object.' % type(fig)
                )

        fig.tight_layout()

        if self.config.plot_git_caption:

            misc.add_git_hash_caption(fig)

        dir_path = self.ensure_qc_dir() if qc_dir else self.config.result_dir
        path = dir_path.joinpath(
            f"{fname}.{self.config.figure_export_file_type}"
        )

        log_msg = log_msg or fname
        module_logger.info(
            f"Saving {log_msg} figure to {path}"
        )

        save = parent.savefig if hasattr(parent, 'savefig') else fig.savefig

        save(path)

        plt.clf()


    def close_figures(self):

        plt.close('all')


    def pmt_data_to_excel(
            self,
            path = None,
            sheet_name = 'PMT_data',
            **kwargs,
        ):

        self._to_excel(
            df = self.pmt_data.data,
            path = path,
            sheet_name = sheet_name,
            fname_suffix = 'pmt',
            label = 'PMT data',
            **kwargs,
        )


    def raw_plugs_to_excel(
            self,
            path = None,
            sheet_name = 'Plugs_raw',
            **kwargs,
        ):

        self._to_excel(
            df = self.plug_data.plug_df,
            path = path,
            sheet_name = sheet_name,
            fname_suffix = 'plugs_raw',
            label = 'raw plug data',
            **kwargs,
        )


    def plugs_to_excel(self, fname = None, sheet_name = 'Plugs', **kwargs):

        self._to_excel(
            df = self.plug_data.sample_df,
            path = path,
            sheet_name = sheet_name,
            fname_suffix = 'plugs',
            label = 'plug data',
            **kwargs,
        )


    def samples_to_excel(
            self,
            path = None,
            sheet_name = 'Samples',
            **kwargs,
        ):

        self._to_excel(
            df = self.sample_data,
            path = path,
            sheet_name = sheet_name,
            fname_suffix = 'samples',
            label = 'sample data',
            **kwargs,
        )


    def stats_to_excel(
            self,
            path = None,
            sheet_name = 'Statistics',
            **kwargs,
        ):

        if not hasattr(self, 'sample_statistics'):

            self.sample_statistics = self.stats()

        self._to_excel(
            df = self.sample_statistics,
            path = path,
            sheet_name = sheet_name,
            fname_suffix = 'stats',
            label = 'sample statistics',
            **kwargs,
        )


    def _export(
            self,
            df,
            path = None,
            fname_suffix = None,
            ftype: str = 'tsv',
            label = 'data',
            sheet_name = 'Sheet1',
            **kwargs,
        ):

        methods = {
            'tsv': 'csv',
            'xlsx': 'excel',
            'tex': 'latex',
            'md': 'markdown',
        }

        ftype = ftype.lower()
        path = path or self.config.result_dir

        if os.path.isdir(path):

            fname_suffix = fname_suffix or 'untitled'
            fname = f'{self.pmt_data.input_file.stem}_{fname_suffix}.{ftype}'
            path = os.path.join(path, fname)

        if not os.path.dirname(path):

            path = os.path.join(self.config.result_dir, path)


        if 'index' not in kwargs:

            kwargs['index'] = isinstance(df.index, pd.MultiIndex)

        if ftype == 'tsv':

            kwargs['sep'] = '\t'

        if ftype == 'xlsx':

            kwargs['sheet_name'] = sheet_name

        method = f'to_{methods.get(ftype, ftype)}'

        getattr(df, method)(path, **kwargs)

        module_logger.info(f'Exported {label} to `{path}`.')


    def _to_excel(
            self,
            df,
            path = None,
            fname_suffix = None,
            label = 'data',
            sheet_name = 'Sheet1',
            **kwargs,
        ):

        args = locals()
        args.pop('self')
        args.pop('kwargs')

        self._export(ftype = 'xlsx', **args, **kwargs)


    def _to_tsv(
            self,
            df,
            path = None,
            fname_suffix = None,
            label = 'data',
            **kwargs,
        ):

        args = locals()
        args.pop('self')
        args.pop('kwargs')

        self._export(ftype = 'tsv', **args, **kwargs)


    def _to_csv(
            self,
            df,
            path = None,
            fname_suffix = None,
            label = 'data',
            **kwargs,
        ):

        args = locals()
        args.pop('self')
        args.pop('kwargs')

        self._export(ftype = 'csv', **args, **kwargs)
