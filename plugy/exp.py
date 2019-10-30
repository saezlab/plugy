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

from .data.config import PlugyConfig
from .data.bd import ChannelMap, PlugSequence
from .data.pmt import PmtData
from .data.plug import PlugData

from dataclasses import dataclass

module_logger = logging.getLogger("plugy.data.exp")


@dataclass
class PlugExperiment(object):
    config: PlugyConfig = PlugyConfig()

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
