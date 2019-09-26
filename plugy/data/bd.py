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
import pathlib as pl

from dataclasses import dataclass


@dataclass
class ChannelMapping(object):
    input_file: pl.Path

    def __post_init__(self):
        self.mapping = self.read_input_file()

    def read_input_file(self):
        with self.input_file.open("r") as f:
            pass



class PlugSequence(object):
    pass
