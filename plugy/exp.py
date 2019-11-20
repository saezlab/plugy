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
import logging
import pathlib as pl

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .data.config import PlugyConfig
from .data.bd import ChannelMap, PlugSequence
from .data.pmt import PmtData
from .data.plug import PlugData
import lib.helpers.helpers as helpers

from dataclasses import dataclass

module_logger = logging.getLogger("plugy.data.exp")


@dataclass
class PlugExperiment(object):
    config: PlugyConfig = PlugyConfig()
    ignore_qc_result = False

    def __post_init__(self):
        module_logger.info(f"Initializing PlugExperiment using the following configuration")
        module_logger.info("\n".join([f"{k}: {v}" for k, v in self.config.__dict__.items()]))

        self.check_config()
        self.channel_map = ChannelMap(self.config.channel_file)
        self.plug_sequence = PlugSequence.from_csv_file(self.config.seq_file)

        self.pmt_data = PmtData(input_file=self.config.pmt_file,
                                acquisition_rate=self.config.acquisition_rate,
                                cut=self.config.cut,
                                correct_acquisition_time=self.config.correct_acquisition_time,
                                ignore_orange_channel=False,
                                ignore_green_channel=False,
                                ignore_uv_channel=False,
                                digital_gain_uv=self.config.digital_gain_uv,
                                digital_gain_green=self.config.digital_gain_green,
                                digital_gain_orange=self.config.digital_gain_orange,
                                config=self.config)

        self.plug_data = PlugData(pmt_data=self.pmt_data,
                                  plug_sequence=self.plug_sequence,
                                  channel_map=self.channel_map,
                                  peak_min_threshold=self.config.peak_min_threshold,
                                  peak_max_threshold=self.config.peak_max_threshold,
                                  peak_min_distance=self.config.peak_min_distance,
                                  peak_min_prominence=self.config.peak_min_prominence,
                                  peak_max_prominence=self.config.peak_max_prominence,
                                  peak_min_width=self.config.peak_min_width,
                                  peak_max_width=self.config.peak_max_width,
                                  width_rel_height=self.config.width_rel_height,
                                  merge_peaks_distance=self.config.merge_peaks_distance,
                                  n_bc_adjacent_discards=self.config.n_bc_adjacent_discards,
                                  min_end_cycle_barcodes=self.config.min_end_cycle_barcodes,
                                  config=self.config)

        sns.set_context(self.config.seaborn_context)
        sns.set_style(self.config.seaborn_style)

        self.sample_data = self.get_sample_data()

        if not self.config.result_dir.exists():
            self.config.result_dir.mkdir()

        self.qc()
        self.drug_combination_analysis()

    def get_sample_data(self) -> pd.DataFrame:
        """
        Generates sample_data df that groups plugs for each sample and cycle calculating their median
        and how much their plug number diverges from the expected as produced by the BD using the PlugSequence
        :return: pd.DataFrame grouped by cycle_nr and sample_nr with the medians and additional plug count divergence
        """
        sample_data = self.plug_data.sample_df.groupby(["cycle_nr", "sample_nr"], as_index=False).median()
        divergence = self.get_plug_count_divergence()

        sample_data = sample_data.set_index(["cycle_nr", "sample_nr"]).join(divergence.set_index(["cycle_nr", "sample_nr"]))
        return sample_data

    def get_plug_count_divergence(self):
        """
        Compares generated and called number of plugs per sample and returns the difference
        :return: pd.Series with the divergence per sample
        """
        divergences = list()
        plug_sample_sequence = self.plug_sequence.get_samples(channel_map=self.channel_map).sequence
        for (cycle_nr, sample_nr), group in self.plug_data.sample_df.groupby(["cycle_nr", "sample_nr"]):
            divergence = (self.config.n_bc_adjacent_discards * 2) + len(group) - plug_sample_sequence[sample_nr].n_replicates
            divergences.append([cycle_nr, sample_nr, divergence])

        return pd.DataFrame(divergences, columns=["cycle_nr", "sample_nr", "plug_count_divergence"])

    def get_contamination(self) -> pd.Series:
        """
        Normalizes filtered plug data to the means of the unfiltered data.
        :return: Series with the relative contamination of the plugs in sample_df
        """
        barcode_mean = self.plug_data.plug_df.loc[self.plug_data.plug_df.barcode].barcode_peak_median.mean()
        control_mean = self.plug_data.sample_df.control_peak_median.mean()
        norm_df = self.plug_data.sample_df.assign(norm_barcode=self.plug_data.sample_df.barcode_peak_median / barcode_mean,
                                                  norm_control=self.plug_data.sample_df.control_peak_median / control_mean)

        return norm_df.norm_barcode / norm_df.norm_control

    def plot_plug_count_hist(self, axes: plt.Axes):
        """
        Plots the distribution of plug number divergence from the expected number by sample
        :param axes: plt.Axes object to draw on
        :return: plt.Axes object with the plot
        """
        axes = sns.countplot(self.sample_data.plug_count_divergence, ax=axes)
        axes.set_ylabel("Counts")
        axes.set_xlabel("Plug count divergence per sample")

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
        qc_successful = True
        qc_dir = self.config.result_dir.joinpath("qc")

        if not qc_dir.exists():
            qc_dir.mkdir()

        # Plotting PMT overview
        pmt_overview_fig, pmt_overview_ax = plt.subplots(figsize=(150, 10))
        pmt_overview_ax = self.pmt_data.plot_pmt_data(axes=pmt_overview_ax)

        pmt_overview_fig.tight_layout()
        if self.config.plot_git_caption:
            helpers.addGitHashCaption(pmt_overview_fig)
        pmt_overview_fig.savefig(qc_dir.joinpath(f"pmt_overview.png"))

        # Plotting plug numbers
        plug_count_hist_fig, plug_count_hist_ax = plt.subplots()
        plug_count_hist_ax = self.plot_plug_count_hist(axes=plug_count_hist_ax)

        plug_count_hist_fig.tight_layout()
        if self.config.plot_git_caption:
            helpers.addGitHashCaption(plug_count_hist_fig)
        plug_count_hist_fig.savefig(qc_dir.joinpath(f"plug_count_hist.{self.config.figure_export_file_type}"))

        # Plotting length bias
        length_bias_plot = self.plug_data.plot_length_bias(col_wrap=8)
        if self.config.plot_git_caption:
            helpers.addGitHashCaption(length_bias_plot.fig)
        length_bias_plot.fig.tight_layout()
        length_bias_plot.fig.savefig(qc_dir.joinpath(f"length_bias.{self.config.figure_export_file_type}"))

        # Plotting contamination
        contamination = self.get_contamination()
        if contamination.mean() > self.config.contamination_threshold:
            qc_successful = False

        contamination_hist_fig, contamination_hist_ax = plt.subplots()
        sns.distplot(contamination, kde=True, ax=contamination_hist_ax)
        contamination_hist_ax.set_title("Relative barcode contamination")
        contamination_hist_ax.set_xlabel(r"Relative contamination $\left[\frac{\overline{barcode_{median}}}{\overline{control_{median}}}\right]$")
        contamination_hist_ax.set_ylabel("Counts")
        helpers.addGitHashCaption(contamination_hist_fig)
        contamination_hist_fig.tight_layout()
        contamination_hist_fig.savefig(qc_dir.joinpath(f"contamination_hist.{self.config.figure_export_file_type}"))

        contamination_fig, contamination_ax = plt.subplots(2, 3, sharex="all", sharey="all", figsize=(30, 20))
        for idx_y, channel in enumerate(["readout_peak_median", "control_peak_median"]):
            contamination_ax[idx_y][2] = self.plug_data.plot_contamination(channel_x="barcode_peak_median", channel_y=channel, hue="start_time", filtered=True, axes=contamination_ax[idx_y][2])
            contamination_ax[idx_y][2].set_title("Filtered")
            for idx_x, hue in enumerate(["start_time", "barcode"]):
                contamination_ax[idx_y][idx_x] = self.plug_data.plot_contamination(channel_x="barcode_peak_median", channel_y=channel, hue=hue, filtered=False, axes=contamination_ax[idx_y][idx_x])
                contamination_ax[idx_y][idx_x].set_title("Unfiltered")

        if self.config.plot_git_caption:
            helpers.addGitHashCaption(contamination_fig)
        contamination_fig.tight_layout()
        contamination_fig.savefig(qc_dir.joinpath(f"contamination.{self.config.figure_export_file_type}"))

        # Plotting control
        control_fig = plt.figure(figsize=(40, 20), constrained_layout=False)
        control_fig_gs = control_fig.add_gridspec(nrows=2, ncols=4)
        control_ax_sample_dist = control_fig.add_subplot(control_fig_gs[0, :])
        control_ax_control_regression = control_fig.add_subplot(control_fig_gs[1, 0])
        control_ax_cycle_dist = control_fig.add_subplot(control_fig_gs[1, 1])
        control_ax_readout_correlation = control_fig.add_subplot(control_fig_gs[1, 2])
        control_ax_control_heatmap = control_fig.add_subplot(control_fig_gs[1, 3])

        control_ax_control_regression = self.plug_data.plot_control_regression(control_ax_control_regression)
        control_ax_cycle_dist = self.plug_data.plot_control_cycle_dist(control_ax_cycle_dist)
        control_ax_sample_dist = self.plug_data.plot_control_sample_dist(control_ax_sample_dist)
        control_ax_readout_correlation = self.plug_data.plot_control_readout_correlation(control_ax_readout_correlation)
        control_ax_control_heatmap = self.plug_data.plot_control_channel_heatmap(control_ax_control_heatmap)

        control_fig.tight_layout()
        if self.config.plot_git_caption:
            helpers.addGitHashCaption(control_fig)
        control_fig.savefig(qc_dir.joinpath(f"control_fluorescence.{self.config.figure_export_file_type}"))

        # Plotting PMT overview
        sample_cycle_fig, sample_cycle_ax = self.plug_data.plot_sample_cycles()
        if self.config.plot_git_caption:
            helpers.addGitHashCaption(sample_cycle_fig)
        sample_cycle_fig.savefig(qc_dir.joinpath("sample_cycle_overview.png"))

        if qc_successful:
            module_logger.info("Quality control successful")
        else:
            module_logger.critical("Quality control failed, check logs and QC plots for more in depth information. In case you still want to continue, you can set ignore_qc_result to True")

        if not self.ignore_qc_result:
            assert qc_successful, "Quality control failed, check logs and QC plots for more in depth information. In case you still want to continue, you can set ignore_qc_result to True"

        return qc_successful

    def drug_combination_analysis(self):
        """
        Analyzes drug combinations and produces result plots
        """
        module_logger.info("Running drug combination analysis")

        # Overview violin plot with z-scores
        drug_z_violin_fig, drug_z_violin_ax = plt.subplots(figsize=(round(len(self.plug_data.sample_df.name.unique()) * 0.8), 10))
        drug_z_violin_ax = self.plug_data.plot_readout_z_violins(axes=drug_z_violin_ax)
        drug_z_violin_ax.set_title("Caspase activity z-scores")
        drug_z_violin_fig.tight_layout()
        if self.config.plot_git_caption:
            helpers.addGitHashCaption(drug_z_violin_fig)
        drug_z_violin_fig.savefig(self.config.result_dir.joinpath(f"drug_comb_z_violins.{self.config.figure_export_file_type}"))

        # Overview heatmap of z-scores
        drug_z_hm_fig, drug_z_hm_ax = plt.subplots()
        drug_z_hm_ax = self.plug_data.plot_readout_z_heatmap(axes=drug_z_hm_ax)
        drug_z_hm_ax.set_title("Caspase activity z-scores")
        drug_z_hm_fig.tight_layout()
        if self.config.plot_git_caption:
            helpers.addGitHashCaption(drug_z_hm_fig)
        drug_z_hm_fig.savefig(self.config.result_dir.joinpath(f"drug_comb_z_heatmap.{self.config.figure_export_file_type}"))
