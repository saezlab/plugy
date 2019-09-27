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

from dataclasses import dataclass

module_logger = logging.getLogger("plugy.data.bd")


@dataclass
class ChannelMap(object):
    input_file: pl.Path

    def __post_init__(self):
        module_logger.info(f"Creating ChannelMap object from file {self.input_file.absolute()}")
        module_logger.debug(f"Configuration:")
        for k, v in self.__dict__.items():
            module_logger.debug(f"{k}: {v}")

        self.map = self.read_input_file()

    def read_input_file(self):
        module_logger.info(f"Reading file")
        mapping = dict()
        with self.input_file.open("r") as f:
            for line in f:
                k, v = line.strip().split(":")

                try:
                    k = int(k)
                except ValueError:
                    raise ValueError(f"Channel has to be an int you specified {k}")

                if k not in range(9, 25):
                    raise ValueError(f"Channel out of BD range (9-24) you specified channel {k}")

                mapping[k] = v

        return mapping


class PlugSequence(object):
    pass
