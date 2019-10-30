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
from dataclasses import dataclass

module_logger = logging.getLogger("plugy.data.exp")


@dataclass
class PlugExperiment(object):
    config: PlugyConfig = PlugyConfig()
    
    def __post_init__(self):
        self.check_config()

    def check_config(self):
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
