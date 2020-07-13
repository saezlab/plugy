#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Author      Nicolas Peschke
# Date        26.09.2019
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

import time
import pathlib as pl

from dataclasses import dataclass, field


@dataclass
class PlugyConfig(object):
    # File Paths
    pmt_file: pl.Path = None
    seq_file: pl.Path = None
    channel_file: pl.Path = None
    result_base_dir: pl.Path = pl.Path.cwd().joinpath("results")
    result_dir_prefix: str = "run"

    # General config
    figure_export_file_type: str = "svg"
    colors: dict = field(default_factory = lambda: {"green": "#5D9731", "blue": "#3A73BA", "orange": "#F68026"})
    run: bool = True

    # PMT configuration
    channels: dict = field(default_factory = lambda: {"barcode": ("uv", 3), "control": ("orange", 2), "readout": ("green", 1)})
    acquisition_rate: int = 300
    cut: tuple = (None, None)
    correct_acquisition_time: bool = True
    ignore_orange_channel: bool = False
    ignore_green_channel: bool = False
    ignore_uv_channel: bool = False
    digital_gain_uv: float = 1.0
    digital_gain_green: float = 1.0
    digital_gain_orange: float = 1.0
    bc_override_threshold: float = None

    # Plug Calling
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

    # Analysis
    normalize_using_control: bool = False
    normalize_using_media_control_lin_reg: bool = True
    readout_column: str = "readout_peak_median"
    readout_analysis_column: str = "readout_peak_z_score"

    # Plotting config
    seaborn_context: str = "notebook"
    seaborn_style: str = "darkgrid"
    plot_git_caption: bool = True

    drug_comb_heatmap_scale_max: float = None
    drug_comb_heatmap_scale_min: float = None

    # QC
    contamination_threshold: float = 0.03

    # Statistics
    alpha: float = 0.05

    def __post_init__(self):
        # Creating result dir for each individual run

        current_time = time.strftime("%Y%m%d_%H_%M_%S")

        self.result_dir = self.result_base_dir.joinpath(f"{self.result_dir_prefix}_{current_time}")

        if not self.result_base_dir.exists():
            self.result_base_dir.mkdir()

        # assert not self.result_dir.exists(), f"Automatically generated result directory name already exists {self.result_dir.name}, please retry in a couple of seconds"

        if not self.result_dir.exists():
            self.result_dir.mkdir()


