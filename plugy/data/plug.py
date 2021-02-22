#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Author      Nicolas Peschke
# Date        19.09.2019
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

import logging
import pickle
import importlib as imp
import warnings
import collections
import itertools
import functools
import inspect
import typing

import pathlib as pl

import tqdm

import pandas as pd
import numpy as np
import scipy.stats as stats
import skimage.filters

import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patch
import matplotlib.collections as mpl_coll
import seaborn as sns

from .. import misc
from ..data import pmt, bd
from ..data.config import PlugyConfig
from dataclasses import dataclass, field

module_logger = logging.getLogger("plugy.data.plug")


@dataclass
class PlugData(object):

    pmt_data: pmt.PmtData
    plug_sequence: bd.PlugSequence
    channel_map: bd.ChannelMap
    auto_detect_cycles: bool = True
    merge_peaks_distance: float = 0.2
    n_bc_adjacent_discards: int = 1
    min_end_cycle_barcodes: int = 12
    min_between_samples_barcodes: int = 2
    min_plugs_in_sample: int = 1
    normalize_using_control: bool = False
    normalize_using_media_control_lin_reg: bool = False

    has_barcode: bool = True
    has_samples_cycles: bool = True
    samples_per_cycle: int = None

    heatmap_second_scale: str = 'pos_ctrl'
    heatmap_override_scale: tuple = None
    heatmap_override_second_scale: tuple = None

    palette: tuple = None
    font_scale: typing.Union[float, int] = 2
    scatter_dot_size: typing.Union[float, int] = 10

    config: PlugyConfig = field(default_factory = PlugyConfig)


    def __post_init__(self):

        module_logger.info(f"Creating PlugData object")
        module_logger.debug(
            f"Configuration: "
            f"{[f'{k}: {v}' for k, v in self.__dict__.items()]}"
        )

        self.detect_plugs()


    def reload(self):
        """
        Reloads the object from the module level.
        """

        modname = self.__class__.__module__
        mod = __import__(modname, fromlist = [modname.split('.')[0]])
        imp.reload(mod)
        new = getattr(mod, self.__class__.__name__)
        setattr(self, '__class__', new)


    def detect_plugs(self):
        """
        Finds plugs using the scipy.signal.find_peaks() method. Merges the
        plugs afterwards if merge_peaks_distance is > 0

        :return: DataFrame containing the plug data and a DataFrame
        containing information about the peaks as called by sig.find_peaks
        """
        module_logger.info("Finding plugs")
        self._detect_peaks()
        self._normalize_to_control()
        self._set_sample_param()
        self._set_barcoding()
        self._set_sample_cycle()


    def detect_samples(self):

        self._set_sample_param()
        self._count_samples_by_cycle()
        self._discard_cycles()
        self._label_samples()
        self._create_sample_df()
        self._add_z_scores()
        self._media_lin_reg_norm()
        self._check_sample_df_column(
            self.config.readout_analysis_column
        )
        self._check_sample_df_column(self.config.readout_column)


    def _detect_peaks(self):
        """
        Detects peaks using scipy.signal.find_peaks().
        :return: Returns a DataFrame containing information about
            the peaks
        as called by sig.find_peaks
        """

        self.pmt_data.detect_peaks()
        self.plug_df = self.pmt_data.peak_df


    def _set_barcoding(self, **kwargs):


        def param_range(val):

            if isinstance(val, (tuple, list)):

                if len(val) == 2:

                    step = (val[1] - val[0]) / 20 * .9999999
                    val = tuple(val) + (step,)

                return np.arange(*val)

            else:

                return (val,)


        config = self.config
        param = config.barcoding_param_defaults.get(
            config.barcoding_method,
            {},
        )
        param.update(config.barcoding_param)
        param.update(kwargs)

        param = dict(
            (key, param_range(val))
            for key, val in param.items()
        )
        self._barcoding_param = collections.namedtuple(
            'BarcodeParam',
            sorted(param.keys())
        )
        self._barcode_result = collections.namedtuple(
            'BarcodeResult',
            [
                'sample_mismatch',
                'sample_freq_var',
            ]
        )
        self._barcode_eval = {}

        n_param = functools.reduce(
            lambda i, j: i * len(j),
            param.values(),
            1
        )
        pbar_desc = 'Adjusting barcode detection%s'

        with tqdm.tqdm(
            total = n_param,
            dynamic_ncols = True,
            desc = pbar_desc % '',
        ) as tq:

            for _values in itertools.product(*param.values()):

                _param = dict(zip(param.keys(), _values))
                self._barcoding_param_last = self._barcoding_param(
                    *(_param[f] for f in self._barcoding_param._fields)
                )
                tq.set_description(
                    pbar_desc % (
                        ' [%s]' % misc.ntuple_str(self._barcoding_param_last)
                    )
                )
                self._set_barcoding_base(**_param)
                self._evaluate_barcoding()
                tq.update()

        self._select_best_barcoding()
        self._set_barcoding_base(**self._barcode_best_param._asdict())

        self._sample_cycle_message()


    def _sample_cycle_message(self):

        if self.has_samples_cycles:

            module_logger.info(
                'Found %u cycles, %u with the expected number of samples. '
                'Sample count deviations: %s. '
                'Best barcode detection parameters: %s.' % (
                    len(self._sample_count_anomaly),
                    list(self._sample_count_anomaly.values()).count(0),
                    misc.dict_str(self._sample_count_anomaly),
                    misc.ntuple_str(self._barcode_best_param),
                )
            )


    def _set_barcoding_base(
            self,
            barcoding_method = None,
            **kwargs,
        ):
        """
        Creates a new boolean column `barcode` in the `plug_df` which is
        `True` if the plug is a barcode.
        """

        if not self.has_barcode:

            module_logger.debug(
                '`has_barcode` is False, skipping barcode identification.'
            )
            return

        config = self.config
        method_name = barcoding_method or config.barcoding_method
        method = '_set_barcoding_%s' % method_name
        param = config.barcoding_param_defaults.get(method_name, {})
        param.update(config.barcoding_param)
        param.update(kwargs)

        if hasattr(self, method):

            module_logger.debug('Barcode detection method: %s' % method)
            module_logger.debug(
                'Barcode detection parameters: %s' % misc.dict_str(param)
            )
            getattr(self, method)(**param)
            self._evaluate_barcoding()

        else:

            module_logger.error('No such method: `%s`' % method)


    def _set_barcoding_simple(self, times: float = None):

        self.plug_df = self.plug_df.assign(
            barcode = (
                self.plug_df.barcode_peak_median >
                self.plug_df.control_peak_median * times
            )
        )


    def _set_barcoding_adaptive(self, **kwargs):

        self._barcoding_thresholds = {}

        method_name = kwargs.pop('thresholding_method')
        method = getattr(skimage.filters, 'threshold_%s' % method_name)
        method_argnames = set(inspect.signature(method).parameters.keys())

        adaptive_method = (
            kwargs['adaptive_method']
                if 'adaptive_method' in kwargs else
            'simple'
        )

        channels = {}
        channel_names = {'barcode', 'control'}

        # adaptive thresholds on blue and orange
        for channel in channel_names:

            param = dict(
                (
                    key if key0 not in channel_names else key1,
                    val
                )
                for key, val, key0, key1 in
                (
                    ([key, val] + key.split('_', maxsplit = 1) + [None])[:4]
                    for key, val in kwargs.items()
                )
                if (
                    (
                        key in method_argnames and
                        '%s_%s' % (channel, key) not in kwargs
                    ) or (
                        key1 in method_argnames and
                        key0 == channel
                    )
                )
            )

            module_logger.debug(
                'Calling `%s` with parameters %s' % (
                    method_name,
                    misc.dict_str(param),
                )
            )

            channels[channel] = self.plug_df[
                '%s_peak_median' % channel
            ].to_numpy()
            shape = (1, channels[channel].shape[0])
            channels[channel].shape = shape

            threshold = method(channels[channel], **param)
            self._barcoding_thresholds[channel] = threshold.flatten()

        # setting the barcode based on the plugs' values and the blue or
        # the orange adaptive thresholds: either the blue is above threshold
        # or the orange is below the threshold
        if adaptive_method == 'simple':

            barcode_threshold =  self._barcoding_thresholds['barcode']
            control_threshold = self._barcoding_thresholds['control']
            self.plug_df['barcode'] = np.logical_or(
                channels['barcode'] > barcode_threshold,
                channels['control'] < control_threshold * .9,
            ).flatten()

        if adaptive_method in {'higher', 'slope'}:

            threshold_barcode_norm = (
                self._barcoding_thresholds['barcode'] /
                self._barcoding_thresholds['barcode'].max()
            )
            threshold_control_norm = (
                self._barcoding_thresholds['control'] /
                self._barcoding_thresholds['control'].max()
            )

        if adaptive_method == 'higher':

            factor = (
                kwargs['higher_threshold_factor']
                    if 'higher_threshold_factor' in kwargs else
                1.
            )
            self.plug_df['barcode'] = (
                threshold_barcode_norm >
                threshold_control_norm * factor
            ).flatten()

        if adaptive_method == 'slope':

            barcode_slope = np.diff(threshold_barcode_norm.flatten())
            barcode_slope = np.concatenate([0], barcode_slope)
            control_slope = np.diff(threshold_control_norm.flatten())
            control_slope = np.concatenate([0], control_slope)

        # adaptive threshold on blue:orange ratio
        param = dict(
            (key, val)
            for key, val in kwargs.items()
            if key in method_argnames
        )
        ratio = (
            self.plug_df.barcode_peak_median.to_numpy() /
            self.plug_df.control_peak_median.to_numpy()
        )
        shape = (1, max(ratio.shape))
        ratio.shape = shape
        ratio = ratio / ratio.max() * 10
        threshold = method(ratio, **param)
        self._barcoding_thresholds['ratio'] = threshold.flatten()
        self._barcoding_thresholds['_ratio'] = ratio.flatten()

        if adaptive_method == 'ratio':

            self.plug_df['barcode'] = (ratio > threshold).flatten()

        #param = self.config.adaptive_param.copy()
        #param.update(kwargs)

        #barcode_control = (
            #self.plug_df.barcode_peak_median /
            #self.plug_df.control_peak_median
        #).to_numpy()
        #barcode_control = self.plug_df.barcode_peak_median.to_numpy().copy()

        #shape = (1, barcode_control.shape[0])
        #barcode_control.shape = shape

        #threshold = skimage.filters.threshold_local(
            #barcode_control,
            #**param,
        #)

        #barcode = self.plug_df.barcode_peak_median > threshold.flatten()
        #barcode = barcode.to_numpy()
        #barcode.shape = shape

        #barcode = skimage.morphology.opening(
            #barcode,
            #selem = skimage.morphology.square(2),
        #)

        #self.plug_df['barcode'] = barcode.flatten()


    def _normalize_to_control(self):
        """
        Normalizes the readout channel by dividing its value by a control
        channel.
        """

        if self.normalize_using_control:

            self.plug_df = self.plug_df.assign(
                readout_per_control = (
                    self.plug_df.readout_peak_median /
                    self.plug_df.control_peak_median
                )
            )


    def quantify_interval(self, start_index, end_index):
        """
        Calculates median for each acquired channel between two data points

        :param start_index: Index of the first datapoint in pmt_data.data
        :param end_index: Index of the last datapoint in pmt_data.data

        :return: List with start_index end_index and the channels according
        to the order of config.channels
        """

        return self.pmt_data.quantify_interval(start_index, end_index)


    def _set_sample_cycle(self, debug = False):
        """
        Enumerates samples and cycles. Adds new columns to the
        :py:attr:`plug_df`: `cycle_nr`, `sample_nr` and `discard`.
        """

        if not self.has_samples_cycles:

            self.plug_df['cycle_nr'] = 0
            self.plug_df['sample_nr'] = 0
            self.plug_df['discard'] = False
            return

        # counters
        current_cycle = 0
        bc_peaks = 0
        sm_peaks = 0
        sample_in_cycle = -1
        # new vectors
        cycle = []
        sample = []
        discard = []
        # short names
        bc_bw_samples = self.min_between_samples_barcodes
        bc_adj_discards = self.n_bc_adjacent_discards
        bc_cycle_end = self.min_end_cycle_barcodes
        sm_min_len = self.min_plugs_in_sample

        if debug:

            print(
                'bc_bw_samples=%u, bc_adj_discards=%u '
                'bc_cycle_end=%u, sm_min_len=%u' % (
                    bc_bw_samples,
                    bc_adj_discards,
                    bc_cycle_end,
                    sm_min_len,
                )
            )

        for idx, bc in enumerate(self.plug_df.barcode):

            if debug:

                print(
                    'plug=%u, '
                    'current_cycle=%u, sample_in_cycle=%u, '
                    'sm_peaks=%u, bc_peaks=%u, barcode=%s' % (
                        idx,
                        current_cycle,
                        sample_in_cycle,
                        sm_peaks,
                        bc_peaks,
                        str(bc),
                    )
                )

            if bc:

                discard.append(True)
                cycle.append(current_cycle)
                sample.append(max(sample_in_cycle, 0))
                bc_peaks += 1

            else:

                # step the cycle counter
                # if this is the first plug of a sample
                if bc_peaks > 0 or sample_in_cycle < 0:

                    if (
                        (
                            bc_peaks >= bc_bw_samples and
                            sm_peaks >= sm_min_len
                        )
                        or
                        (
                            bc_peaks >= self.min_end_cycle_barcodes and
                            (current_cycle or sm_peaks)
                        )
                        or
                        sample_in_cycle < 0
                    ):

                        sample_in_cycle += 1

                        if (
                            bc_peaks >= self.min_end_cycle_barcodes and
                            # I hope this won't mess up the behaviour
                            # in any other scenario
                            # was like this before:
                            # sample_in_cycle > 0
                            # and possibly this will be better:
                            (
                                current_cycle or
                                sm_peaks
                            )
                        ):

                            current_cycle += 1
                            sample_in_cycle = 0

                        sm_peaks = 0

                    bc_peaks = 0

                sample.append(sample_in_cycle)
                cycle.append(current_cycle)

                # stepping the within sample plug counter
                sm_peaks += 1

                # Discarding barcode-adjacent plugs
                try:
                    if (
                        self.plug_df.barcode[idx - bc_adj_discards] or
                        self.plug_df.barcode[idx + bc_adj_discards]
                    ):
                        discard.append(True)
                    else:
                        discard.append(False)
                except KeyError:
                    discard.append(False)

        self.plug_df = self.plug_df.assign(
            cycle_nr = cycle,
            sample_nr = sample,
            discard = discard,
        )


    def _add_z_scores(self):
        # Calculating z-score on filtered data and inserting it after
        # readout_peak_median (index 5)

        if len(self.sample_df) > 1:

            self.sample_df.insert(
                loc = 5,
                column = "readout_peak_z_score",
                value = stats.zscore(self.sample_df.readout_peak_median),
            )

            if self.normalize_using_control:

                self.sample_df.insert(
                    loc = 6,
                    column = "readout_per_control_z_score",
                    value = stats.zscore(self.sample_df.readout_per_control),
                )

        else:

            module_logger.warning(
                f"Samples DataFrame contains {len(self.sample_df)} line(s), "
                f"omitting z-score calculation!"
            )


    def get_media_control_data(self) -> pd.DataFrame:
        """
        Returns a DataFrame with only media control plugs (both compounds FS)
        :return: pd.DataFrame with media control plugs
        """

        control_labels = misc.to_set(self.config.control_label)

        media_control_data = self.sample_df.loc[
            self.sample_df.compound_a.isin(control_labels) &
            self.sample_df.compound_b.isin(control_labels)
        ]

        return media_control_data


    def get_media_control_lin_reg(self, readout_column: str = ""):
        """
        Calculates a linear regression over time of all media control plugs
        The readout_peak_median column is used to calculate the regression

        :return: Tuple with slope, intercept, rvalue, pvalue and stderr of
        the regression. See https://docs.scipy.org/doc/scipy/reference/
        generated/scipy.stats.linregress.html for more information about
        the returned values.
        """
        media_control = self.get_media_control_data()

        readout_column = readout_column or self.config.readout_column

        slope, intercept, rvalue, pvalue, stderr = stats.linregress(
            media_control.start_time,
            media_control[readout_column],
        )

        return slope, intercept, rvalue, pvalue, stderr


    def seaborn_setup(self):

        sns.set_context(
            self.config.seaborn_context,
            font_scale = self.config.font_scale,
            rc = self.config.seaborn_context_dict,
        )
        sns.set_style(
            self.config.seaborn_style,
            rc = self.config.seaborn_style_dict,
        )


    def plot_plug_pmt_data(
            self,
            axes: plt.Axes,
            cut: tuple = (None, None),
        ) -> plt.Axes:
        """
        Plots pmt data and superimposes rectangles with the called plugs upon
        the plot

        :param axes: plt.Axes object to plot to
        :param cut: tuple with (start_time, end_time) to subset the plot to
        a certain time range

        :return: plt.Axes object with the plot
        """

        axes = self.pmt_data.plot_pmt_data(axes, cut = cut)

        plug_df = self.plug_df
        sample_df = self.sample_df

        if cut[0] is not None:
            plug_df = plug_df.loc[plug_df.start_time >= cut[0]]
            sample_df = sample_df.loc[sample_df.start_time >= cut[0]]

        if cut[1] is not None:
            plug_df = plug_df.loc[plug_df.end_time <= cut[1]]
            sample_df = sample_df.loc[sample_df.end_time <= cut[1]]

        # Plotting light green rectangles that indicate
        # the used plug length and plug height
        bc_patches = list()
        readout_patches = list()

        for plug in plug_df.itertuples():

            if plug.barcode:

                bc_patches.append(
                    mpl_patch.Rectangle(
                        xy = (plug.start_time, 0),
                        width = plug.end_time - plug.start_time,
                        height = plug.barcode_peak_median,
                    )
                )


        for plug in sample_df.itertuples():

            readout_patches.append(
                mpl_patch.Rectangle(
                    xy = (plug.start_time, 0),
                    width = plug.end_time - plug.start_time,
                    height = plug.readout_peak_median,
                )
            )

        axes.add_collection(
            mpl_coll.PatchCollection(
                bc_patches,
                facecolors = self.config.colors['uv'],
                alpha = 0.4,
            )
        )
        axes.add_collection(
            mpl_coll.PatchCollection(
                readout_patches,
                facecolors = self.config.colors['green'],
                alpha = 0.4,
            )
        )

        return axes


    def highlight_plugs(self, axes: plt.Axes, below_peak: bool = False):

        if below_peak:

            return self.highlight_plugs_below_peak(axes)

        else:

            return self.highlight_plugs_vspan(axes)


    def highlight_plugs_below_peak(self, axes: plt.Axes):

        plug_patches = {
            'readout': [],
            'barcode': [],
        }

        for plug in self.plug_df.itertuples():

            plug_type = 'barcode' if plug.barcode else 'readout'
            patch = mpl_patch.Rectangle(
                xy = (plug.start_time, 0),
                width = plug.end_time - plug.start_time,
                height = getattr(plug, '%s_peak_median' % plug_type),
            )
            plug_patches[plug_type].append(patch)

        colors = {
            'readout': self.config.colors['green'],
            'barcode': self.config.colors['blue'],
        }

        for key, color in colors.items():

            axes.add_collection(
                mpl_coll.PatchCollection(
                    plug_patches[key],
                    facecolors = color,
                    alpha = .4,
                )
            )

        return axes


    def highlight_plugs_vspan(self, axes: plt.Axes):

        for plug in self.plug_df.itertuples():

            color = self.config.colors['uv' if plug.barcode else 'green']
            axes.axvspan(
                xmin = plug.start_time,
                xmax = plug.end_time,
                facecolor = color,
                alpha = .4,
            )

        return axes


    def highlight_samples(self, axes: plt.Axes):

        if 'sample_nr' not in self.plug_df.columns:

            return axes

        samples = self.plug_df.groupby(
            ['cycle_nr', 'sample_nr']
        ).agg(
            {
                'start_time': 'min',
                'end_time': 'max',
            }
        )

        ymax = axes.get_ylim()[1]

        for sample in samples.itertuples():

            axes.axvspan(
                xmin = sample.start_time,
                xmax = sample.end_time,
                facecolor = '#777777',
                alpha = .3,
            )
            axes.text(
                x = sample.start_time,
                y = ymax - .1,
                s = '%u/%u' % (sample.Index[0] + 1, sample.Index[1] + 1),
                size = 'xx-large',
            )

        return axes


    def pmt_plot_add_thresholds(self, axes: plt.Axes):

        if not hasattr(self, '_barcoding_thresholds'):

            return

        time = (self.plug_df.start_time + self.plug_df.end_time) / 2

        for channel, threshold in self._barcoding_thresholds.items():

            if channel[0] == '_':

                continue

            color = self.config.channel_color(channel)
            print('color: ', color, 'channel: ', channel)
            sns.lineplot(
                x = time,
                y = threshold / threshold.max() * axes.get_ylim()[1] * .95,
                color = color,
                style = True,
                dashes = [(2,2)],
                linewidth = 1.,
                legend = False,
                ax = axes,
            )

        ratio = self._barcoding_thresholds['_ratio']
        ratio = ratio / ratio.max() * axes.get_ylim()[1] * .95

        sns.scatterplot(
            x = time,
            y = ratio,
            color = '#CC00CC',
            ax = axes,
        )


    def plot_cycle_pmt_data(self, axes: plt.Axes) -> plt.Axes:
        """
        Plots pmt data and superimposes filled rectangles for cycles with
        correct numbers of samples and
        unfilled rectangles for discarded cycles

        :param axes: plt.Axes object to plot to
        :return: plt.Axes object with the plot
        """

        module_logger.info("Plotting cycle data")
        axes = self.pmt_data.plot_pmt_data(axes = axes, cut = (None, None))

        used_cycle_patches = list()
        discarded_cycle_patches = list()
        patch_height = (
            self.pmt_data.data[["green", "orange", "uv"]].max().max()
        )

        for used_cycle in self.sample_df.groupby("cycle_nr"):

            cycle_start_time = used_cycle[1].start_time.min()
            cycle_end_time = used_cycle[1].end_time.max()
            used_cycle_patches.append(
                mpl_patch.Rectangle(
                    xy = (cycle_start_time, 0),
                    width = cycle_end_time - cycle_start_time,
                    height = patch_height,
                )
            )

        for detected_cycle in self.plug_df.groupby("cycle_nr"):

            cycle_start_time = detected_cycle[1].start_time.min()
            cycle_end_time = detected_cycle[1].end_time.max()
            discarded_cycle_patches.append(
                mpl_patch.Rectangle(
                    xy = (cycle_start_time, 0),
                    width = cycle_end_time - cycle_start_time,
                    height = patch_height,
                )
            )

            axes.text(
                x = (
                    (cycle_end_time - cycle_start_time) / 2 +
                    cycle_start_time
                ),
                y = 0.9 * patch_height,
                s = f"Cycle {detected_cycle[0]}",
                horizontalalignment = "center",
            )

        axes.add_collection(
            mpl_coll.PatchCollection(
                used_cycle_patches,
                facecolors = "green",
                alpha = 0.4,
            )
        )
        axes.add_collection(
            mpl_coll.PatchCollection(
                discarded_cycle_patches,
                edgecolors = "red",
                facecolors = "none",
                alpha = 0.4,
            )
        )

        return axes


    def _set_sample_param(self):
        """
        Sets up the parameters for counting and possibly labeling samples.
        """

        # Label samples in case channel map and plug sequence are provided
        self._has_sequence = (
            isinstance(self.channel_map, bd.ChannelMap) and
            isinstance(self.plug_sequence, bd.PlugSequence)
        )

        self.sample_sequence = (
            self.plug_sequence.get_samples(channel_map = self.channel_map)
                if self._has_sequence else
            None
        )
        self.expected_samples = (
            len(self.sample_sequence.sequence)
                if self._has_sequence else
            self.samples_per_cycle
        )


    def _ensure_sample_param(self):

        if not hasattr(self, 'expected_samples'):

            self._set_sample_param()


    def _evaluate_barcoding(self):

        self._ensure_sample_param()
        self._count_samples_by_cycle()
        self._evaluate_barcoding_base()


    def _evaluate_barcoding_base(self):

        if self.has_samples_cycles:

            sample_mismatch = sum(
                abs(a)
                for a in self._sample_count_anomaly.values()
            )
            sample_freq_var = 0

            for cycle_nr in self.plug_df.cycle_nr.unique():

                this_cycle = self.plug_df[self.plug_df.cycle_nr == cycle_nr]
                start_times = this_cycle.groupby('sample_nr').min('start_time')
                sample_freq_var += start_times['start_time'].std()

            result = self._barcode_result(
                sample_mismatch = sample_mismatch,
                sample_freq_var = sample_freq_var,
            )

        else:

            result = (0, 0)

        self._barcode_eval[self._barcoding_param_last] = result


    def _select_best_barcoding(self):

        self._barcode_best_param = min(
            self._barcode_eval,
            key = self._barcode_eval.get,
        )


    def _adjust_sample_detection(self):
        """
        Labels sample_df with associated names and compounds according to the
        ChannelMap in the PlugSequence

        :param sample_df: pd.DataFrame with sample_nr column to associate
        names and compounds

        :return: pd.DataFrame with the added name, compound_a and b columns
        """

        if not self._has_sequence or self.config.adaptive:

            return

        blue_highest = self.config.blue_highest
        n_valid_cycles = collections.defaultdict(list)

        while True:

            self._count_samples_by_cycle()
            self._get_valid_cycles()
            n_valid_cycles[len(self.valid_cycles)].append(blue_highest)

            if (
                any(
                    0 < abs(d) < 4
                    for d in self._sample_count_anomaly.values()
                ) and
                blue_highest < 1.5
            ):

                blue_highest += .05
                self._set_barcode(blue_highest = blue_highest)
                self._set_sample_cycle()

            else:

                break

        blue_highest = n_valid_cycles[max(n_valid_cycles.keys())][0]
        self._set_barcode(blue_highest = blue_highest)
        self._set_sample_cycle()
        self._count_samples_by_cycle()

        module_logger.info(
            f"Adjusted `blue_highest` "
            f"to {blue_highest}."
        )


    def _count_samples_by_cycle(self, update = True):

        if not self.expected_samples:

            return

        if update:

            self._set_sample_cycle()

        self._samples_by_cycle = dict(
            (
                cycle_nr,
                cycle.sample_nr.nunique()
            )
            for cycle_nr, cycle in self.plug_df.groupby('cycle_nr')
        )

        self._sample_count_anomaly = dict(
            (
                cycle_nr,
                n_samples - self.expected_samples
            )
            for cycle_nr, n_samples in self._samples_by_cycle.items()
        )


    def _discard_cycles(self):

        for cycle_nr, cycle in self.plug_df.groupby('cycle_nr'):

            diff = self._sample_count_anomaly[cycle_nr]

            if diff:

                log_msg = (
                    f"Cycle {cycle_nr} detected between "
                    f"{cycle.start_time.min()}-{cycle.end_time.max()} "
                    f"contains {self._samples_by_cycle[cycle_nr]} samples, "
                    f"{'less' if diff < 0 else 'more'} than "
                    f"expected ({self.expected_samples})."
                )
                module_logger.info(log_msg)
                module_logger.info(f"Discarding cycle {cycle_nr}.")


                with warnings.catch_warnings():

                    warnings.simplefilter('ignore')
                    self.plug_df.discard[
                        self.plug_df.cycle_nr == cycle_nr
                    ] = True

        self._get_valid_cycles()
        self.n_cycles = len(self._samples_by_cycle)

        message = (
            self.pmt_data._detection_issues_message()
                if hasattr(self, 'pmt_data') else
            ''
        )

        if not self.valid_cycles:

            module_logger.critical(
                f"None of the {self.n_cycles} cycles is valid."
            )

            if message:

                module_logger.info(message)

        elif len(self.valid_cycles) != self.n_cycles:

            module_logger.info(
                f"Out of {self.n_cycles} only "
                f"{len(self.valid_cycles)} are valid."
            )

            if not self.auto_detect_cycles:

                module_logger.critical(
                    f"All cycles must have the expected number of samples."
                )
                self.valid_cycles = ()

            if message:

                module_logger.info(message)

            if not self.auto_detect_cycles:

                self.valid


    def _get_valid_cycles(self):

        self.valid_cycles = [
            cycle_nr
            for cycle_nr, diff in self._sample_count_anomaly.items()
            if not diff
        ]


    def _label_samples(self):

        assert self.valid_cycles, 'No valid cycles available.'

        module_logger.info('Labelling samples with compound names')

        df = self.plug_df
        seq = self.sample_sequence

        df['name'] = df.loc[~df.discard].sample_nr.apply(
            lambda nr: self.get_sample_name(nr, seq)
        )
        df['compound_a'] = df.loc[~df.discard].sample_nr.apply(
            lambda nr: self.channel_map.get_compounds(
                seq.sequence[nr].open_valves
            )[0]
        )
        df['compound_b'] = df.loc[~df.discard].sample_nr.apply(
            lambda nr: self.channel_map.get_compounds(
                seq.sequence[nr].open_valves
            )[1]
        )


    def _create_sample_df(self):

        sample_df = self.plug_df.loc[self.plug_df.discard == False]
        self.sample_df = sample_df.drop(columns = ['discard', 'barcode'])


    def get_sample_name(
            self,
            sample_nr: int,
            sample_sequence: bd.PlugSequence,
        ):
        """
        Returns a unified naming string for a sample.
        Concatenation of both compounds or single compound or cell control
        :param sample_nr: Sample number
        :param sample_sequence: bd.PlugSequence object to get open valves from
        """
        compounds = self.channel_map.get_compounds(
            sample_sequence.sequence[sample_nr].open_valves
        )
        compounds = [compound for compound in compounds if compound != "FS"]
        compounds = ' + '.join(compounds) or 'Cell Control'

        return compounds


    def plot_sample_cycles(self):
        """
        Creates a plot with pmt data for the individual samples and cycles.
        :return: plt.Figure and plt.Axes object with the plot
        """
        names = self.sample_df.name.unique()
        cycles = sorted(self.sample_df.cycle_nr.unique())

        sample_cycle_fig, sample_cycle_ax = plt.subplots(
            nrows = len(names),
            ncols = len(cycles),
            figsize = (7 * len(cycles), 5 * len(names)),
            squeeze = False,
        )

        y_max = self.sample_df.readout_peak_median.max() * 1.1

        for idx_y, name in enumerate(names):
            for idx_x, cycle in enumerate(cycles):
                module_logger.debug(
                    f"Plotting sample {idx_y + 1} of {len(names)}, "
                    f"cycle {idx_x + 1} of {len(cycles)}"
                )
                sample_cycle_ax[idx_y][idx_x] = self.plot_sample(
                    name = name,
                    cycle_nr = cycle,
                    axes = sample_cycle_ax[idx_y][idx_x],
                )
                sample_cycle_ax[idx_y][idx_x].set_ylim((0, y_max))

        sample_cycle_fig.tight_layout()

        return sample_cycle_fig, sample_cycle_ax


    def plot_sample(
            self,
            name: str,
            cycle_nr: int,
            axes: plt.Axes,
            offset: int = 10,
        ) -> plt.Axes:
        """
        Plots the PMT traces for a particular drug and cycle and

        :param name: Name of the drug combination/valve as listed in the
        PlugSequence
        :param cycle_nr: Number of the cycle
        :param axes: The plt.Axes object to draw on
        :param offset: How many seconds to plot left and right of the plugs

        :return: The plt.Axes object with the plot
        """
        peak_data = self.sample_df[(self.sample_df.cycle_nr == cycle_nr) & (self.sample_df.name == name)]
        if len(peak_data) == 0:
            axes.text(0.5, 0.5, "No Data")
        else:
            axes = self.plot_plug_pmt_data(axes = axes, cut = (peak_data.start_time.min() - offset, peak_data.end_time.max() + offset))

        axes.set_title(f"{name} | Cycle {cycle_nr}")
        return axes

        # start_time = peak_data.iloc[0].t0 - offset
        # end_time = peak_data.iloc[-1].t1 + offset
        #
        # plotting_data = pd.DataFrame(self.data)
        # plotting_data = plotting_data[(plotting_data[0] > start_time) & (plotting_data[0] < end_time)]
        #
        # sns.lineplot(x = plotting_data[0], y = plotting_data[1], estimator = None, ci = None, sort = False, color = self.colors["green"], ax = axes)
        # sns.lineplot(x = plotting_data[0], y = plotting_data[2], estimator = None, ci = None, sort = False, color = self.colors["orange"], ax = axes)
        # sns.lineplot(x = plotting_data[0], y = plotting_data[3], estimator = None, ci = None, sort = False, color = self.colors["blue"], ax = axes)
        #
        # # Plotting light green rectangles that indicate the used plug length and plug height
        # patches = list()
        # for plug in peak_data.itertuples():
        #     patches.append(mpl_patch.Rectangle(xy = (plug.t0, 0), width = plug.length, height = plug.green))
        # axes.add_collection(mpl_coll.PatchCollection(patches, facecolors = self.colors["green"], alpha = 0.4))
        #
        # axes.set_xlabel("Time [s]")
        # axes.set_ylabel("Fluorescence [AU]")
        # axes.set_title(f"{drug} Cycle {cycle}")

        # return axes


    # QC Plots
    def plot_media_control_evolution(
        self,
        axes: plt.Axes,
        by_sample = False,
    ) -> plt.Axes:
        """
        Plots a scatter plot with readout medians for the media control
        over the experiment time.

        :param axes: plt.Axes object to draw on
        :param by_sample: True to plot swarmplot by sample number
        :return: plt.Axes object with the plot
        """

        self.seaborn_setup()

        if self.normalize_using_control:
            readout_column = 'readout_per_control'
            ylab = 'Readout normalized to control'
        else:
            readout_column = 'readout_peak_median'
            ylab = 'Readout median'

        plot_data = self.get_media_control_data()

        if by_sample:

            axes = sns.swarmplot(
                x = 'sample_nr',
                y = readout_column,
                data = plot_data,
                ax = axes,
                hue = 'cycle_nr',
                dodge = True,
                palette = list(self.palette),
                size = self.scatter_dot_size,
            )
            axes.set_xlabel('Sample Number')
            axes.get_legend().set_title('Cycle')

        else:

            slope, intercept, rvalue, _, _ = (
                self.get_media_control_lin_reg(readout_column)
            )
            axes = sns.scatterplot(
                x = 'start_time',
                y = readout_column,
                data = plot_data,
                ax = axes,
                color = self.palette[0],
                s = self.scatter_dot_size * 10,
            )
            misc.plot_line(slope, intercept, axes)
            label = axes.text(
                0.1,
                0.9,
                f'R²: {round(rvalue, 2)}',
                transform = axes.transAxes,
            )
            label.set_bbox(dict(facecolor = 'white', alpha = 0.8))
            axes.set_xlabel('Experiment Time [s]')

        axes.set_title('FS media control plug fluorescence')
        axes.set_ylabel(ylab)
        return axes


    def add_length_column(self):
        """
        Creates a new column ``length`` in ``sample_df`` with the difference
        of plug start and end times.
        """

        self.sample_df = self.sample_df.assign(
            length = (
                self.sample_df.end_time -
                self.sample_df.start_time
            )
        )


    def plot_length_bias(self, col_wrap: int = 3) -> sns.FacetGrid:
        """
        Plots each plugs fluorescence over its length grouped by valve. Also fits a linear regression to show if there
        is a correlation between the readout and the plug length indicating non ideal mixing.
        
        :param col_wrap: After how many subplots the column should be wrapped.
        :return: sns.FacetGrid object with the subplots
        """

        self.add_length_column()

        df = self.sample_df

        df = df.groupby('name').filter(lambda x: x.shape[0] > 1)

        length_bias_plot = sns.lmplot(
            x = "length",
            y = "readout_peak_median",
            col = "name",
            data = df,
            col_wrap = col_wrap,
        )
        length_bias_plot.set_xlabels("Length")
        length_bias_plot.set_ylabels("Fluorescence [AU]")

        length_bias_plot.set(
            ylim = (0, df.readout_peak_median.max()),
            xlim = (df.length.min(), df.length.max()),
        )

        return length_bias_plot


    def plot_contamination(
            self,
            channel_x: str,
            channel_y: str,
            axes: plt.Axes,
            filtered: bool = False,
            hue: str = "start_time",
            normalize: bool = False,
        ) -> plt.Axes:
        """
        Plots contamination as a scatter plot
        :param channel_x: Name of the channel to be plotted on the x axis
            (e.g. readout_peak_median, bc_peak_median, control_peak_median)
        :param channel_y: Name of the channel to be plotted on the y axis
            (e.g. readout_peak_median, bc_peak_median, control_peak_median)
        :param filtered: True if sample_df should be used, False if raw
            plug_df should be used
        :param axes: plt.Axes object to draw on
        :param hue: Name of the column in plug_df that is used to color the dots
        :param normalize: True plug data should be scaled by its mean
        :return: plt.Axes object with the plot
        """

        if filtered:
            norm_df = self.sample_df
        else:
            norm_df = self.plug_df.loc[
                self.plug_df.cycle_nr.isin(self.sample_df.cycle_nr.unique())
            ]

        palette = list(self.palette[:2]) if hue == 'barcode' else None

        if normalize:
            norm_df = norm_df.assign(
                norm_x = norm_df[channel_x] / norm_df[channel_x].mean(),
                norm_y = norm_df[channel_y] / norm_df[channel_y].mean(),
            )
            contamination_plot = sns.scatterplot(
                x = "norm_x",
                y = "norm_y",
                hue = hue,
                data = norm_df,
                ax = axes,
                alpha = .5,
                palette = palette,
            )

            axes.set_xlabel(
                '%s median [%% of mean]' % (
                    channel_x.split('_')[0].capitalize()
                )
            )
            axes.set_ylabel(
                '%s median [%% of mean]' % (
                    channel_y.split('_')[0].capitalize()
                )
            )

        else:
            contamination_plot = sns.scatterplot(
                x = channel_x,
                y = channel_y,
                hue = hue,
                data = norm_df,
                ax = axes,
                alpha = .5,
                palette = palette,
            )
            axes.set_xlabel(
                '%s median [AU]' % channel_x.split('_')[0].capitalize()
            )
            axes.set_ylabel(
                '%s median [AU]' % channel_y.split('_')[0].capitalize()
            )

        return contamination_plot


    def plot_control_regression(self, axes: plt.Axes) -> plt.Axes:
        """
        Plots a scatter plot of control peak medians over experiment time and applies a linear regression to it
        :param axes: plt.Axes object to draw on
        :return: plt.Axes object with the plot
        """
        axes = sns.regplot(x = "start_time", y = "control_peak_median", data = self.sample_df, ax = axes)
        axes.set_title("Control Time Bias")
        axes.set_ylabel("Peak Median Fluorescence Intensity [AU]")
        axes.set_xlabel("Experiment Time [s]")
        return axes


    def plot_control_cycle_dist(self, axes: plt.Axes) -> plt.Axes:
        """
        Gathers control peak medians by cycle and plots a violin plot
        :param axes: plt.Axes object to draw on
        :return: plt.Axes object with the plot
        """
        axes = sns.violinplot(x = "cycle_nr", y = "control_peak_median", data = self.sample_df, ax = axes)
        axes.set_title("Control Intensity by Cycle")
        axes.set_ylabel("Peak Median Fluorescence Intensity [AU]")
        axes.set_xlabel("Cycle")
        return axes


    def plot_control_sample_dist(self, axes: plt.Axes) -> plt.Axes:
        """
        Gathers control peak medians by sample and plots a violin plot
        :param axes: plt.Axes object to draw on
        :return: plt.Axes object with the plot
        """
        axes = sns.violinplot(x = "name", y = "control_peak_median", data = self.sample_df, ax = axes)
        axes.set_title("Control Intensity by Sample")
        axes.set_ylabel("Peak Median Fluorescence Intensity [AU]")
        axes.set_xlabel("Sample")
        for tick in axes.get_xticklabels():
            tick.set_rotation(90)
        return axes


    def plot_control_readout_correlation(self, axes: plt.Axes) -> plt.Axes:
        """
        Correlates control and readout peak medians
        :param axes: plt.Axes object to draw on
        :return: plt.Axes object with the plot
        """
        # axes = sns.regplot(x = "control_peak_median", y = "readout_peak_median",  data = self.sample_df, ax = axes)
        axes = sns.scatterplot(x = "control_peak_median", y = "readout_peak_median", hue = "sample_nr", style = "cycle_nr", data = self.sample_df, ax = axes)
        axes.set_title("Readout - Control Correlation")
        axes.set_xlabel("Control Peak Median Fluorescence Intensity [AU]")
        axes.set_ylabel("Readout Peak Median Fluorescence Intensity [AU]")

        axes.set_ylim(self.sample_df.readout_peak_median.min() * 0.95, self.sample_df.readout_peak_median.max() * 1.05)
        axes.set_xlim(self.sample_df.control_peak_median.min() * 0.95, self.sample_df.control_peak_median.max() * 1.05)

        return axes


    def save(self, file_path: pl.Path):
        """
        Saves this PlugData object as pickle
        :param file_path: Path to the file to write to
        """
        with file_path.open("wb") as f:
            pickle.dump(self, f)


    def plot_compound_violins(self, column_to_plot: str= "readout_peak_z_score", by_cycle: bool = False) -> sns.FacetGrid:
        """
        Plots a violin plot per compound combination
        :param column_to_plot: Column to be plotted (e.g. readout_peak_z_score, readout_per_control_z_score).
        :param by_cycle: Produce separate plots for each cycle.
        :return: seaborn.FacetGrid object with the plot
        """

        self._check_sample_df_column(column_to_plot)

        height = 3 * (len(self.sample_df.cycle_nr.unique()) if by_cycle else 1) + 2
        aspect = round(len(self.sample_df.name.unique()) * 0.35) / height

        args = {'row': 'cycle_nr'} if by_cycle else {}

        grid = sns.catplot(x = 'name', y = column_to_plot, data = self.sample_df, kind = 'violin', height = height, aspect = aspect, **args)

        for ax in grid.axes.flat:

            ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)

        return grid


    def plot_compound_heatmap(
            self,
            column_to_plot: str,
            annotation_df: pd.DataFrame = None,
            annotation_column: str = "significant",
            by_cycle: bool = False,
            **kwargs
        ) -> sns.FacetGrid:
        """
        Plots a heatmap to visualize the different combinations
        :param column_to_plot: Name of the column to extract values from
        :param annotation_df: pd.DataFrame grouped by column, compound_a and
            compound_b for annotation
        :param annotation_column: Which column in annotation_df to use for
            the annotation
        :param by_cycle: Produce separate plots for each cycle.
        :return: seaborn.FacetGrid object with the plot
        """
        self._check_sample_df_column(column_to_plot)

        assert column_to_plot not in {"compound_a", "compound_b"},\
            f"You can not plot this coulumn on a heatmap: `{column_to_plot}`."

        if annotation_df is not None:

            annotation_df = annotation_df.reset_index()
            annotation_df = annotation_df.pivot(
                'compound_a',
                'compound_b',
                annotation_column,
            )
            annotation_df = (
                annotation_df.
                replace(True, '*').
                replace(False, '')
            )

        cycles = self.sample_df.cycle_nr.unique() if by_cycle else (None,)

        second_scale = (
            self.heatmap_second_scale in set(self.sample_df.compound_a) or
            self.heatmap_second_scale in set(self.sample_df.compound_b)
        )

        aspect_correction = 1.33 if second_scale else 1.0

        grid = sns.FacetGrid(
            data = self.sample_df,
            col = 'cycle_nr' if by_cycle else None,
            height = 5,
            aspect = (
                (.55 * len(cycles) if by_cycle else 1) *
                aspect_correction
            ),
        )

        for i, cycle in enumerate(cycles):

            data = self.sample_df
            data = data if cycle is None else data[data.cycle_nr == cycle]
            data = data[[column_to_plot, "compound_a", "compound_b"]]
            data = data.groupby(["compound_a", "compound_b"]).mean()
            data = data.reset_index()
            data = data.pivot("compound_a", "compound_b", column_to_plot)
            data = data.reindex(
                data.isna().
                sum(axis = 1).
                sort_values(ascending = False).
                index.
                to_list()
            )
            data = data[
                data.isna().
                sum(axis = 0).
                sort_values(ascending = True).
                index.
                to_list()
            ]

            if annotation_df is not None:

                annotation_df = (
                    annotation_df.reindex_like(data).
                    replace(np.nan, '')
                )

            if second_scale:

                second_scale_data = data.copy()
                data[self.heatmap_second_scale] = np.nan
                second_scale_data.loc[
                    :,
                    second_scale_data.columns != self.heatmap_second_scale
                ] = np.nan

                annot_second_scale = None

                if annotation_df is not None:

                    annot_second_scale = annotation_df.copy()
                    annot_second_scale.loc[
                        :,
                        annotation_df.columns != self.heatmap_second_scale
                    ] = ''

            ax = grid.axes.flat[i]

            vmin, vmax = self.heatmap_override_scale or (None, None)

            sns.heatmap(
                data,
                annot = annotation_df,
                fmt = "",
                ax = ax,
                vmin = vmin,
                vmax = vmax,
                **kwargs
            )

            if second_scale:

                vmin, vmax = (
                    self.heatmap_override_second_scale or
                    (None, None)
                )

                sns.heatmap(
                    second_scale_data,
                    annot = annot_second_scale,
                    cmap = 'viridis',
                    fmt = '',
                    ax = ax,
                    vmin = vmin,
                    vmax = vmax,
                    **kwargs
                )

            cycle_str = ('\nCycle #%u' % cycle) if by_cycle else ''
            unit_str = 'z-score' if 'z_score' in column_to_plot else 'AU'
            ax.set_title(
                f"{column_to_plot} by combination [{unit_str}]{cycle_str}"
            )
            ax.set_ylabel("")
            ax.set_xlabel("")

            if plt.matplotlib.__version__ == '3.1.1':

                ylim = ax.get_ylim()
                ax.set_ylim(ylim[0] + .5, ylim[1] - .5)

        return grid


    def _check_sample_df_column(self, column: str):
        """
        Checks if column is in sample_df
        :param column: Column to check
        """
        try:
            assert column in self.sample_df.columns.to_list(), f"Column {column} not in the column names of sample_df ({self.sample_df.columns.to_list()}), specify a column from the column names!"
        except AssertionError:
            module_logger.critical(f"Column {column} not in the column names of sample_df ({self.sample_df.columns.to_list()}), specify a column from the column names!")
            raise


    def _media_lin_reg_norm(self):
        """
        Normalizes sample_df using media control regression

        :return: updated sample_df
        """

        if self.normalize_using_media_control_lin_reg:

            sample_df = self.sample_df

            if self.normalize_using_control:
                readout_column = "readout_per_control"
            else:
                readout_column = "readout_peak_median"

            slope, intercept, _, _, _ = self.get_media_control_lin_reg(readout_column=readout_column)

            sample_df = sample_df.assign(readout_media_norm=sample_df[readout_column] / (sample_df["start_time"] * slope + intercept))
            sample_df = sample_df.assign(readout_media_norm_z_score=stats.zscore(sample_df.readout_media_norm))

            self.sample_df = sample_df


    def append(
            self,
            plug_df: pd.DataFrame,
            sample_df: pd.DataFrame = None,
            offset: float = 1.,
        ):

        # if PlugExperiment provided
        if hasattr(plug_df, 'plug_data'):

            plug_df = plug_df.plug_data

        # if PlugData provided
        if hasattr(plug_df, 'plug_df'):

            if sample_df is None and hasattr(plug_df, 'sample_df'):

                sample_df = plug_df.sample_df

            plug_df = plug_df.plug_df

        self.plug_df = self.concat_dfs(self.plug_df, plug_df, offset)

        if sample_df is not None and hasattr(self, 'sample_df'):

            self.sample_df = self.concat_dfs(
                self.sample_df,
                sample_df,
                offset,
            )

        if self.has_barcode:

            self._count_samples_by_cycle(update = False)
            self._sample_cycle_message()


    @staticmethod
    def concat_dfs(df1: pd.DataFrame, df2: pd.DataFrame, offset: float = 1.):

        tmax = df1.end_time.max()
        cmax = df1.cycle_nr.max()

        df2 = df2.copy()
        df2.start_time = df2.start_time + tmax + offset
        df2.end_time = df2.end_time + tmax + offset
        df2.cycle_nr = df2.cycle_nr - df2.cycle_nr.min() + cmax + 1

        return pd.concat([df1, df2]).reset_index()

