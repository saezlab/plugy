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

import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patch
import matplotlib.collections as mpl_coll


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
    width_rel_height: float = 0.5
    merge_peaks_distance: float = 0.2
    n_bc_adjacent_discards: int = 1
    config: PlugyConfig = PlugyConfig()

    def __post_init__(self):
        module_logger.info(f"Creating PlugData object")
        module_logger.debug(f"Configuration: {[f'{k}: {v}' for k, v in self.__dict__.items()]}")

        self.plug_df, self.peak_data = self.find_plugs()
        # self.samples_df = self.find_sample_cycles()p

    def find_plugs(self):
        """
        Finds plugs using the scipy.signal.find_peaks() method. Merges the plugs afterwards if merge_peaks_distance is > 0
        :return: DataFrame containing the plug data and a DataFrame containing information about the peaks as called by sig.find_peaks
        """
        module_logger.info("Finding plugs")
        peak_df = self.detect_peaks()

        plug_list = self.merge_peaks(peak_df)

        # Build plug_df DataFrame
        module_logger.debug("Building plug_df DataFrame")
        channels = [f"{str(key)}_peak_median" for key in self.config.channels.keys()]
        plug_df = pd.DataFrame(plug_list, columns=["start_time", "end_time"] + channels)

        # Call barcode plugs
        module_logger.debug("Calling barcode plugs")
        plug_df = plug_df.assign(barcode=(plug_df.barcode_peak_median > plug_df.readout_peak_median) | (plug_df.barcode_peak_median > plug_df.control_peak_median))

        return plug_df, peak_df

    def detect_peaks(self):
        """
        Detects peaks using scipy.signal.find_peaks().
        :return: Returns a DataFrame containing information about the peaks as called by sig.find_peaks
        """
        peak_df = pd.DataFrame()
        for channel, (channel_color, _) in self.config.channels.items():
            module_logger.debug(f"Running peak detection for {channel} channel")
            peaks, properties = sig.find_peaks(self.pmt_data.data[channel_color],
                                               height=(self.peak_min_threshold,
                                                       self.peak_max_threshold),

                                               distance=round(self.peak_min_distance *
                                                              self.pmt_data.acquisition_rate),

                                               prominence=(self.peak_min_prominence,
                                                           self.peak_max_prominence),

                                               width=(self.peak_min_width * self.pmt_data.acquisition_rate,
                                                      self.peak_max_width * self.pmt_data.acquisition_rate),

                                               rel_height=self.width_rel_height)

            properties = pd.DataFrame.from_dict(properties)
            properties = properties.assign(barcode=True if channel == "barcode" else False)
            peak_df = peak_df.append(properties)

        # Converting ips values to int for indexing later on
        peak_df.assign(right_ips=round(peak_df.right_ips), left_ips=round(peak_df.left_ips))
        peak_df = peak_df.astype({"left_ips": "int32", "right_ips": "int32"})
        return peak_df

    def merge_peaks(self, peak_df):
        """
        Merges peaks if merge_peaks_distance is > 0.
        :param peak_df: DataFrame with peaks as called by detect_peaks()
        :return: List containing plug data.
        """
        plug_list = list()
        if self.merge_peaks_distance > 0:
            module_logger.info(f"Merging plugs with closer centers than {self.merge_peaks_distance} seconds")
            merge_peaks_samples = self.pmt_data.acquisition_rate * self.merge_peaks_distance
            merge_df = peak_df.assign(plug_center=peak_df.left_ips + ((peak_df.right_ips - peak_df.left_ips) / 2))
            merge_df = merge_df.sort_values(by="plug_center")
            merge_df = merge_df.reset_index(drop=True)

            centers = merge_df.plug_center

            # Count through array
            i = 0
            while i < len(centers):
                # Count neighborhood
                j = 0
                while True:
                    if (i + j >= len(centers)) or (centers[i + j] - centers[i] > merge_peaks_samples):
                        # Merge from the left edge of the first plug (i) to the right base of the last plug to merge (i + j - 1)
                        plug_list.append(self.get_plug_data_from_index(merge_df.left_ips[i], merge_df.right_ips[i + j - 1]))

                        # Skip to the next unmerged plug
                        i += j
                        break
                    else:
                        j += 1

        else:
            module_logger.info("Creating plug list with without merging close plugs!")
            for row in peak_df.sort_values(by="left_ips"):
                plug_list.append(self.get_plug_data_from_index(row.left_ips, row.right_ips))
        return plug_list

    def get_plug_data_from_index(self, start_index, end_index):
        """
        Calculates median for each acquired channel between two data points
        :param start_index: Index of the first datapoint in pmt_data.data
        :param end_index: Index of the last datapoint in pmt_data.data
        :return: List with start_index end_index and the channels according to the order of config.channels
        """
        return_list = list()
        return_list.append(start_index / self.pmt_data.acquisition_rate)
        return_list.append(end_index / self.pmt_data.acquisition_rate)
        for _, (channel_color, _) in self.config.channels.items():
            return_list.append(self.pmt_data.data[channel_color][start_index:end_index].median())

        return return_list

    def find_sample_cycles(self):
        """
        Finds cycles and labels individual samples
        :return: DataFrame containing sample data
        """
        samples_df = self.plug_df

        # counters
        current_cycle = 0
        bc_peaks = 0
        sm_peaks = 0
        sample_in_cycle = 0
        # new vectors
        cycle = []
        sample = []
        discard = []

        for idx, bc in enumerate(samples_df.barcode):
            if bc:
                discard.append(True)
                cycle.append()
                bc_peaks += 1
            elif samples_df.barcode[idx - self.n_bc_adjacent_discards] or samples_df[idx + self.n_bc_adjacent_discards]:
                discard.append(True)

            else:
                pass

        return samples_df

    def plot_plug_pmt_data(self, axes: plt.Axes, cut: tuple = None) -> plt.Axes:
        """
        Plots pmt data and superimposes rectangles with the called plugs upon the plot
        :param axes: plt.Axes object to plot to
        :param cut: tuple with (start_time, end_time) to subset the plot to a certain time range
        :return: plt.Axes object with the plot
        """
        axes = self.pmt_data.plot_raw_data(axes)
        module_logger.info("Plotting detected peaks")

        if cut is not None:
            raise NotImplementedError("Plotting specific ranges is not yet implemented!")

        # Plotting light green rectangles that indicate the used plug length and plug height
        patches = list()
        for plug in self.plug_df.itertuples():
            patches.append(mpl_patch.Rectangle(xy=(plug.start_time, 0), width=plug.end_time - plug.start_time, height=plug.readout_peak_median))
        axes.add_collection(mpl_coll.PatchCollection(patches, facecolors=self.config.colors["green"], alpha=0.4))

        return axes
