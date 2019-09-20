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

import re
import gzip
import pathlib as pl

import pandas as pd

from dataclasses import dataclass

import unittest


@dataclass
class PmtData(object):
    input_file: pl.Path

    def read_txt(self):
        with gzip.open(self.input_file, "rt") as f:
            for idx, line in enumerate(f):
                if re.match(pattern=r"\t\d", string=line) is not None:
                    break

            # data = np.genfromtxt(f, delimiter="\t", skip_header=idx)
            f.seek(0)
            data_frame = pd.read_csv(f, sep="\t", decimal=",", skiprows=idx, header=None).iloc[:, 1:]
            data_frame.columns = ["time", "green", "orange", "uv"]

        return data_frame
