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

import pandas as pd
import scipy.signal as sig

from ..data import pmt, bd
from dataclasses import dataclass

module_logger = logging.getLogger("plugy.data.plug")


@dataclass
class PlugData(object):
    pmt_data: pmt.PmtData
    plug_sequence: bd.PlugSequence
    channel_map: bd.ChannelMap
    peak_min_threshold: float = 0.05
    peak_max_threshold: float = 2.0
    peak_min_distance: float = 0.03
    peak_min_prominence: float = 0
    peak_max_prominence: float = 10
    peak_min_width: float = 0.5
    peak_max_width: float = 1.5
    prominence_rel_wlen: float = 3
    width_rel_height: float = 0.5
    peak_min_plateau_size: float = 0.5
    peak_max_plateau_size: float = 1.5

    def __post_init__(self):
        self.plug_df, self.peak_data = self.find_plugs()

    def find_plugs(self):
        peaks, properties = sig.find_peaks(self.pmt_data.data.orange,
                                           height=(self.peak_min_threshold, self.peak_max_threshold),
                                           distance=round(self.peak_min_distance * self.pmt_data.acquisition_rate),
                                           prominence=(self.peak_min_prominence, self.peak_max_prominence),
                                           width=(self.peak_min_width * self.pmt_data.acquisition_rate, self.peak_max_width * self.pmt_data.acquisition_rate),
                                           wlen=round(self.prominence_rel_wlen * self.pmt_data.acquisition_rate),
                                           rel_height=self.width_rel_height,
                                           plateau_size=(self.peak_min_plateau_size * self.pmt_data.acquisition_rate, self.peak_max_plateau_size * self.pmt_data.acquisition_rate))

        peak_df = pd.DataFrame.from_dict(properties)
        return pd.DataFrame(), peak_df
