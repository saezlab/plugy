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

import os
import sys
import time
import pathlib as pl
import typing
import logging
import hashlib

from dataclasses import dataclass, field


@dataclass
class PlugyConfig(object):

    # File Paths
    pmt_file: pl.Path = None
    seq_file: pl.Path = None
    channel_file: pl.Path = None
    result_base_dir: pl.Path = pl.Path.cwd().joinpath("results")
    result_subdirs: bool = False
    timestamp_result_subdirs: bool = False

    # General config
    name: str = None
    figure_export_file_type: str = "svg"
    colors: dict = field(default_factory = lambda: {"green": "#5D9731", "blue": "#3A73BA", "orange": "#F68026"})
    run: bool = True
    init: bool = True
    plugs: bool = True
    control_label: typing.Union[set, str] = field(
        default_factory = lambda: {"FS", "neg Ctr"}
    )
    ignore_qc_result: bool = False
    log_to_stdout: bool = True

    # sequence settings
    allow_lt4_valves: bool = False

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
    min_between_samples_barcodes: int = 2

    # if False, plug detection stops after the identification of the plugs
    # and no sample and cycle numbers will be assigned to the plugs
    has_samples_cycles: bool = True
    # the number of samples within one cycle; no need to provide if sequence
    # file is available; otherwise it helps in evaluating barcode detection
    # methods even if the sequence is unknoen
    samples_per_cycle: int = None
    # the name of a method for identifying barcode plugs; the simplest method,
    # which works with not too noisy data, is `blue_highest` which means the
    # plugs where the channel of the blue value is the highest are barcode
    # plugs;
    barcode_method: str = 'blue_highest'
    # a scaling factor for the `blue_highest` method: the blue channel must
    # be at least this times higher than the control channel for barcode
    # plugs
    # former name: barcode_threshold
    blue_highest_times: float = 1.0
    # change the value of the `blue_highest_times` parameter to find the one
    # giving the best result
    blue_highest_adaptive: bool = False

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

        self._set_name()

        current_time = time.strftime("%Y%m%d_%H%M")

        self.result_dir = self.result_base_dir

        if self.result_subdirs:

            subdir = (
                f"{os.path.splitext(self.pmt_file.name)[0]}_{current_time}"
                    if self.timestamp_result_subdirs else
                self.name
            )

            self.result_dir = self.result_dir.joinpath(
                self.result_dir,
                subdir,
            )

        os.makedirs(str(self.result_dir), exist_ok = True)

        self.start_logging()


    def _set_name(self):

        self.identity = hashlib.md5(
            ';'.join(
                (
                    str(self.pmt_file),
                    str(self.seq_file),
                    str(self.channel_file),
                )
            ).encode('utf-8')
        ).hexdigest()[:7]

        self.name = (
            self.name or
            '%s-%s' % (
                os.path.splitext(self.pmt_file.name)[0],
                self.identity,
            )
        )


    def start_logging(self):
        """
        Starts logging to STDOUT and to a log file in the result directory
        :param config: PlugyConfig object to retrieve the result_dir from
        :return: None
        """

        logging.shutdown()

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt = "%d.%m.%y %H:%M:%S",
        )

        logger = logging.getLogger()
        logger.handlers = []
        logger.setLevel(logging.DEBUG)

        if self.log_to_stdout:

            stream_handler = logging.StreamHandler(stream=sys.stdout)
            stream_handler.setLevel(logging.INFO)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(
            self.result_dir.joinpath("plugy_run.log"),
            mode="a"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
