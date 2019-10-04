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
from ..data.config import PlugyConfig
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
    merge_peaks_distance: float = 0.2
    config: PlugyConfig = PlugyConfig()

    def __post_init__(self):
        self.plug_df, self.peak_data = self.find_plugs()

    def find_plugs(self):
        """

        :return:
        """
        peak_df = pd.DataFrame()
        for channel, (channel_color, _) in self.config.channels.items():
            peaks, properties = sig.find_peaks(self.pmt_data.data[channel_color],
                                               height=(self.peak_min_threshold,
                                                       self.peak_max_threshold),

                                               distance=round(self.peak_min_distance *
                                                              self.pmt_data.acquisition_rate),

                                               prominence=(self.peak_min_prominence,
                                                           self.peak_max_prominence),

                                               width=(self.peak_min_width * self.pmt_data.acquisition_rate,
                                                      self.peak_max_width * self.pmt_data.acquisition_rate),

                                               wlen=round(self.prominence_rel_wlen * self.pmt_data.acquisition_rate),

                                               rel_height=self.width_rel_height,

                                               plateau_size=(self.peak_min_plateau_size * self.pmt_data.acquisition_rate,
                                                             self.peak_max_plateau_size * self.pmt_data.acquisition_rate))

            properties = pd.DataFrame.from_dict(properties)
            properties = properties.assign(barcode=True if channel == "barcode" else False)
            peak_df = peak_df.append(properties)

        plug_list = list()
        if self.merge_peaks_distance > 0:
            merge_peaks_samples = self.pmt_data.acquisition_rate * self.merge_peaks_distance
            merge_df = peak_df.assign(plug_center=peak_df.right_edges - (peak_df.plateau_sizes / 2))
            merge_df = merge_df.sort_values(by="plug_center")
            merge_df = merge_df.reset_index(drop=True)


            centers = merge_df.plug_center

            # Count through array
            i = 0
            while i < len(centers):
                # Count neighborhood
                j = 0
                while True:
                    # if i + j >= len(centers):
                    #     plug_list.append(self.get_plug_data_from_index(merge_df.left_edges[i], merge_df.right_bases[i + j - 1]))
                    #     i += j
                    #     break
                    # else:
                    if (i + j >= len(centers)) or (centers[i + j] - centers[i] > merge_peaks_samples):
                        plug_list.append(self.get_plug_data_from_index(merge_df.left_edges[i], merge_df.right_bases[i + j - 1]))
                        i += j
                        break
                    else:
                        j += 1



            # for idx, center in enumerate(centers):
            #     i = 0
            #     while True:
            #         if centers[idx + i] - center > merge_peaks_samples:
            #
            #         break

        else:
            for row in peak_df.sort_values(by="left_edges"):
                plug_list.append(self.get_plug_data_from_index(row.left_edges, row.right_bases))

        # Build plug_df DataFrame
        channels = [f"{str(key)}_peak_median" for key in self.config.channels.keys()]
        plug_df = pd.DataFrame(plug_list, columns=["start_time", "end_time"] + channels)

        # Call barcode plugs
        plug_df = plug_df.assign(barcode=plug_df.barcode_peak_median > plug_df.readout_peak_median)
        return plug_df, peak_df

    def get_plug_data_from_index(self, start_index, end_index):
        return_list = list()
        return_list.append(start_index / self.pmt_data.acquisition_rate)
        return_list.append(end_index / self.pmt_data.acquisition_rate)
        for _, (channel_color, _) in self.config.channels.items():
            return_list.append(self.pmt_data.data[channel_color][start_index:end_index].median())

        # return_list = return_list.append(self.pmt_data.data[self.config.channels["barcode"][0]][start_index:end_index].median()
        #  = self.pmt_data.data[self.config.channels["barcode"][0]][start_index:end_index].median()
        return return_list
