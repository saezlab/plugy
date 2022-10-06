#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# This file is part of the `plugy` python module
#
# Copyright
# 2018-2022
# EMBL & Heidelberg University
#
# Author(s): Dénes Türei (turei.denes@gmail.com)
#            Nicolas Peschke
#            Olga Ivanova
#
# Distributed under the GPLv3 License.
# See accompanying file LICENSE.txt or copy at
#     http://www.gnu.org/licenses/gpl-3.0.html
#
# Webpage: https://github.com/saezlab/plugy
#

from typing import Literal

import sys
import re
import logging
import pickle
import importlib as imp
import warnings
import collections
import itertools
import functools
import inspect
import typing
from dataclasses import dataclass, field
import pathlib as pl

import tqdm

import pandas as pd
import numpy as np
import scipy.stats as stats
import skimage.filters
import statsmodels.stats.multitest as multitest

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patch
import matplotlib.collections as mpl_coll
import seaborn as sns

from .. import misc
from ..data import pmt, bd
from ..data.config import PlugyConfig


module_logger = logging.getLogger(__name__)


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
    has_controls: bool = True

    heatmap_second_scale: str = 'pos_ctrl'
    heatmap_override_scale: tuple = None
    heatmap_override_second_scale: tuple = None

    palette: tuple = None
    font_scale: typing.Union[float, int] = 2
    scatter_dot_size: typing.Union[float, int] = 10
    short_labels: bool = True

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
        self.discard_samples()
        self._add_z_scores()
        self._baseline_adjust()
        self._check_sample_df_column(
            self.config.readout_analysis_column
        )
        self._check_sample_df_column(self.config.readout_column)
        self.calculate_z_factor()
        self.calculate_modified_z_factor()
        self.update_baseline()
        self.update_negative()
        self.update_fc()


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

            self.sample_df = self._z_score(
                df = self.sample_df,
                col = 'readout_peak_median',
                loc = 5,
            )

            if self.normalize_using_control:

                self.sample_df = self._z_score(
                    df = self.sample_df,
                    col = 'readout_per_control',
                    loc = 6,
                )

        else:

            module_logger.warning(
                f"Samples DataFrame contains {len(self.sample_df)} line(s), "
                f"omitting z-score calculation!"
            )


    def calculate_z_factor(self, modified = False):
        """
        Calculates the Z factor for each complete cycle according to
        https://en.wikipedia.org/wiki/Z-factor. Negative controls are
        the medium only samples and positive controls are the positive
        control samples with the mean of the corresponding single drug
        samples subtracted. Set the sample labels by the config options
        `medium_control_label` and `positive_control_label`. If the
        argument `modified` is True, it calculates a modified version
        of the Z factor, see `calculate_modified_z_factor`.
        """

        if not self.has_controls:

            return

        z_factors = []
        readout_col = self.readout_col
        label = 'modified ' if modified else ''
        mu = '\u03bc'
        sigma = '\u03c3'
        times = '\u00d7'
        division = '\u00f7'
        controls = ('pos', 'med') + (('neg',) if modified else ())

        for cycle in self.cycles:

            pos_control = self.positive_controls(
                subtract_single_drugs = True,
                cycle = cycle,
            )
            pos_control = pos_control[readout_col]

            medium_control = self.medium_only(cycle = cycle)
            medium_control = medium_control[readout_col]

            std_pos_c = np.std(pos_control)
            std_med_c = np.std(medium_control)
            mean_pos_c = np.mean(pos_control)
            mean_med_c = np.mean(medium_control)

            if modified:

                neg_control = self.negative_controls(cycle = cycle)
                neg_control = neg_control[readout_col]
                std_neg_c = np.std(neg_control)
                mean_neg_c = np.mean(neg_control)

                z_factor_numerator = 2 * (
                    std_pos_c +
                    std_med_c +
                    std_neg_c
                )
                z_factor_denominator = abs(
                    mean_pos_c -
                    mean_med_c -
                    mean_neg_c
                )

            else:

                z_factor_numerator = 3 * (
                    std_med_c +
                    std_pos_c
                )
                z_factor_denominator = abs(
                    mean_pos_c -
                    mean_med_c
                )

            the_z_factor = 1 - (z_factor_numerator / z_factor_denominator)
            z_factors.append(the_z_factor)

            # the three blocks below are only for logging
            _locals = locals()

            formula_parts = [
                [
                    '[%s%s=%.03f]' % (
                        sigma if metric == 'std' else mu,
                        ctrl,
                        _locals['%s_%s_c' % (metric, ctrl)],
                    )
                    for ctrl in controls
                ]
                for metric in ('std', 'mean')
            ]

            module_logger.info(
                (
                    '%sz-factor formula (cycle #%u):\n'
                    '    1 - ( %u %s ( %s ) %s %s %s %s ) = %.02f' % (
                        label,
                        cycle,
                        2 if modified else 3,
                        times,
                        ' + '.join(formula_parts[0]),
                        division,
                        '|',
                        ' - '.join(formula_parts[1]),
                        '|',
                        the_z_factor,
                    )
                ).capitalize()
            )

        attr = 'z_factors%s' % ('_%s' % label.strip() if label else '')

        setattr(self, attr, z_factors)

        z_factors = [round(e, 2) for e in z_factors]
        module_logger.info(
            f"Reporting {label}z-factor "
            f"by cycle: {z_factors}"
        )

        return z_factors


    def calculate_modified_z_factor(self):
        """
        Calculates a Z factor with the SD and mean of positive controls,
        medium controls and negative controls are all in the numerator and
        denominator, respectively. The positive + negative control samples
        are ignored instead of being part of the negative control group,
        while the positive control + medium samples are part of the positive
        control group. The means of the corresponding single drug samples
        are subtracted from the positive controls.
        """

        return self.calculate_z_factor(modified = True)


    @property
    def pos_ctrl_lab(self) -> str:
        """
        Returns the label of the positive control channel.
        """

        pos_ctrl_lab = misc.first(
            misc.to_set(self.config.positive_control_label) & self.compounds
        )

        module_logger.debug(f'Positive control label: `{pos_ctrl_lab}`')

        return pos_ctrl_lab


    @property
    def neg_ctrl_lab(self) -> str:
        """
        Returns the label of the negative control channel.
        """

        neg_ctrl_lab = misc.first(
            misc.to_set(self.config.negative_control_label) & self.compounds
        )

        module_logger.debug(f'Negative control label: `{neg_ctrl_lab}`')

        return neg_ctrl_lab


    @property
    def med_ctrl_lab(self) -> set[str]:
        """
        Returns the labels of the medium only channels in the current
        experiment. Or whatever is considered a baseline control.
        """

        med_ctrl_lab = {
            y.strip() for y in
            itertools.chain(*(
                x.split('+')
                for x in misc.to_tuple(self.config.medium_control_label)
            ))
        }

        module_logger.debug(f'Medium control label: `{med_ctrl_lab}`')

        return med_ctrl_lab


    @property
    def readout_col(self):
        """
        Returns the column name for the readout column according to the
        current settings. The name of this column can vary because normally
        at each normalization or correction step we create a new column.
        """

        return self.config.readout_analysis_column


    @property
    def compounds(self):
        """
        Returns a set of all compound labels in the current experiment.
        """

        return (
            set(self.sample_df.compound_a) |
            set(self.sample_df.compound_b)
        )


    @property
    def channels(self):
        """
        Returns the channel labels according to the config.
        """

        return self.config.channel_roles


    @property
    def cycles(self):
        """
        Return the cycle identifiers in the current experiment.
        """

        return self.sample_df.cycle_nr.unique()


    def positive_controls(self, subtract_single_drugs = True, cycle = None):
        """
        Returns a subset of the `sample_df` with only the positive control
        samples.

        :param bool subtract_single_drugs: Subtract the means of the
            corresponding single drug samples from the positive control
            values.
        :param int cycle: Restrict the data to these cycles. If None, all
            cycles will be used.
        """

        if cycle is None:

            return pd.concat([
                self.positive_controls(
                    subtract_single_drugs = subtract_single_drugs,
                    cycle = cycle,
                )
                for cycle in self.cycles
            ])

        pos_ctrl_lab = self.pos_ctrl_lab
        neg_ctrl_lab = self.neg_ctrl_lab
        med_ctrl_lab = self.med_ctrl_lab
        readout_col = self.readout_col

        pos_control = self.sample_df.loc[
            (
                (
                    (self.sample_df.compound_b == pos_ctrl_lab) &
                    (self.sample_df.compound_a != neg_ctrl_lab)
                ) |
                (
                    (self.sample_df.compound_a == pos_ctrl_lab) &
                    (self.sample_df.compound_b != neg_ctrl_lab)
                )
            ) &
            (self.sample_df.cycle_nr == cycle)
        ]

        if subtract_single_drugs:

            single_drugs = self.sample_df.loc[
                (
                    (
                         self.sample_df.compound_a.isin(med_ctrl_lab) &
                        (self.sample_df.compound_b != pos_ctrl_lab) &
                        (self.sample_df.compound_b != neg_ctrl_lab)
                    ) |
                    (
                         self.sample_df.compound_b.isin(med_ctrl_lab) &
                        (self.sample_df.compound_a != pos_ctrl_lab) &
                        (self.sample_df.compound_a != neg_ctrl_lab)
                    )
                ) &
                (self.sample_df.cycle_nr == cycle)
            ]

            # these warnings stuff just filter out the pandas setting
            # with copy warnings, which are an annoying example of the
            # bad design of pandas
            with warnings.catch_warnings():

                warnings.simplefilter('ignore')

                single_drugs['compound'] = np.where(
                    single_drugs.compound_a.isin(med_ctrl_lab),
                    single_drugs.compound_b,
                    single_drugs.compound_a
                )

            single_drugs = (
                single_drugs.groupby('compound')[readout_col].agg('mean')
            ).rename('single_drug_mean')

            with warnings.catch_warnings():

                warnings.simplefilter('ignore')

                pos_control['compound'] = np.where(
                    pos_control.compound_a == pos_ctrl_lab,
                    pos_control.compound_b,
                    pos_control.compound_a
                )

            pos_control = pos_control.join(
                single_drugs,
                on = 'compound'
            )
            pos_control[readout_col] = (
                pos_control[readout_col] -
                pos_control.single_drug_mean
            )

        return pos_control


    def negative_controls(self, cycle = None):
        """
        Returns a subset of the `sample_df` with only the negative control
        samples.

        :param int cycle: Restrict the data to these cycles. If None, all
            cycles will be used.
        """

        if cycle is None:

            return pd.concat([
                self.negative_controls(cycle = cycle)
                for cycle in self.cycles
            ])

        pos_ctrl_lab = self.pos_ctrl_lab
        neg_ctrl_lab = self.neg_ctrl_lab

        neg_control = self.sample_df.loc[
            (
                (self.sample_df.compound_a == neg_ctrl_lab) |
                (self.sample_df.compound_b == neg_ctrl_lab)
            ) &
            (
                (self.sample_df.compound_a != pos_ctrl_lab) &
                (self.sample_df.compound_b != pos_ctrl_lab)
            ) &
            (self.sample_df.cycle_nr == cycle)
        ]

        return neg_control


    def medium_only(self, cycle = None):
        """
        Returns a subset of the `sample_df` with the medium only samples.

        :param int cycle: Restrict the data to these cycles. If None, all
            cycles will be used.
        """

        if cycle is None:

            return pd.concat([
                self.medium_only(cycle = cycle)
                for cycle in self.cycles
            ])

        med_ctrl_lab = self.med_ctrl_lab

        medium_control = self.sample_df.loc[
             self.sample_df.compound_a.isin(med_ctrl_lab) &
             self.sample_df.compound_b.isin(med_ctrl_lab) &
            (self.sample_df.cycle_nr == cycle)
        ]

        return medium_control


    def baseline_lm(
            self,
            col: str | None = None,
            baseline: Literal['baseline', 'negative'] = 'baseline',
        ) -> misc.LinearRegression | None:
        """
        Calculates a linear regression on baseline control plugs over time.

        Args:
            col:
                A column in the samples data frame. If not provided, the
                ``readout_column`` will be used from the config.
            baseline:
                Use the medium only samples or the negative controls as
                baseline.

        Return:
            Tuple with slope, intercept, rvalue, pvalue and stderr of
            the regression. See https://docs.scipy.org/doc/scipy/reference/
            generated/scipy.stats.linregress.html for more information about
            the returned values.
        """

        get_baseline = dict(
            baseline = self.medium_only,
            negative = self.negative_controls,
        )

        baseline_df = get_baseline[baseline]()

        if not bool(len(baseline_df)):

            module_logger.warning(
                'Could not find baseline samples, unable to '
                'fit a linear regression on them. If the experiment '
                'contains such samples, check the config value of '
                f'`medium_control_label` (currently `{self.med_ctrl_lab}`).'
            )

            lm = None

        else:

            col = col or self.config.readout_column

            lm = misc.LinearRegression(
                x = baseline_df.start_time,
                y = baseline_df[col],
            )

        return lm


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
        the plot. This function has been replaced by
        ``plugy.data.pmt.PmtData.plot_pmt_data`` and is probably safe to
        remove.

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


    def highlight_plugs_below_peak(self, ax: mpl.axes.Axes) -> mpl.axes.Axes:

        plug_patches = {
            'readout': [],
            'barcode': [],
        }

        xmin, xmax = ax.get_xlim()

        for plug in self.plug_df.itertuples():

            if plug.end_time < xmin or plug.start_time > xmax:

                continue

            plug_type = 'barcode' if plug.barcode else 'readout'
            xleft = max(plug.start_time, xmin)
            xright = min(plug.end_time, xmax)
            patch = mpl_patch.Rectangle(
                xy = (xleft, 0),
                width = xright - xleft,
                height = getattr(plug, '%s_peak_median' % plug_type),
            )
            plug_patches[plug_type].append(patch)

        colors = {
            'readout': self.config.colors['green'],
            'barcode': self.config.colors['blue'],
        }

        for key, color in colors.items():

            ax.add_collection(
                mpl_coll.PatchCollection(
                    plug_patches[key],
                    facecolors = color,
                    alpha = .4,
                )
            )

        return ax


    def highlight_plugs_vspan(self, ax: mpl.axes.Axes) -> mpl.axes.Axes:

        xmin, xmax = ax.get_xlim()

        for plug in self.plug_df.itertuples():

            if plug.end_time < xmin or plug.start_time > xmax:

                continue

            color = self.config.colors['uv' if plug.barcode else 'green']

            module_logger.debug(
                'Highlighting plug between %.02f and %.02f (%s).' % (
                    max(plug.start_time, xmin),
                    min(plug.end_time, xmax),
                    color,
                )
            )

            ax.axvspan(
                xmin = max(plug.start_time, xmin),
                xmax = min(plug.end_time, xmax),
                facecolor = color,
                alpha = .4,
            )

        return ax


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
        xmin, xmax = axes.get_xlim()

        for sample in samples.itertuples():

            if sample.start_time > xmax or sample.start_time < xmin:

                continue

            axes.axvspan(
                xmin = max(sample.start_time, xmin),
                xmax = min(sample.end_time, xmax),
                facecolor = '#AAAAAA33',
                edgecolor = 'black',
                linewidth = 1,
            )
            axes.text(
                x = max(sample.start_time, xmin),
                y = ymax - .15,
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
                start_times = (
                    this_cycle.groupby('sample_nr').min('start_time')
                )
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
            lambda nr: self.get_compound_name(nr, 0)
        )
        df['compound_b'] = df.loc[~df.discard].sample_nr.apply(
            lambda nr: self.get_compound_name(nr, 1)
        )


    def discard_samples(self, samples: list[str] = None):
        """
        Discard certain samples or compound channels. ``samples`` is a list
        of sample labels or compound labels. In case of the latter, all
        samples with any of the given compounds will be discarded.
        """

        samples = self.config.exclude if samples is None else samples
        samples = misc.to_set(samples)

        self.sample_df = self.sample_df[
            ~(
                self.sample_df.name.isin(samples) |
                self.sample_df.compound_a.isin(samples) |
                self.sample_df.compound_b.isin(samples)
            )
        ]

    def _create_sample_df(self):

        sample_df = self.plug_df.loc[self.plug_df.discard == False]
        self.sample_df = sample_df.drop(columns = ['discard', 'barcode'])


    def get_compound_name(self, sample_nr, which):
        """
        :param sample_nr: Sample number in the sequence of samples.
        :param which: Either 0 or 1. Index within the compound combination.
        """

        seq = self.sample_sequence

        if self.channel_map is None:

            compound = 'Compound %u/%u' % (sample_nr, which)

        else:

            compound = self.channel_map.get_compounds(
                seq.sequence[sample_nr].open_valves
            )[which]

        return compound


    def get_sample_name(
            self,
            sample_nr: int,
            sample_sequence: bd.PlugSequence,
        ) -> str:
        """
        Returns a unified naming string for a sample.
        Concatenation of both compounds or single compound or cell control
        :param sample_nr: Sample number
        :param sample_sequence: bd.PlugSequence object to get open valves from
        """

        if self.channel_map is None:

            compounds = 'Sample %u' % sample_nr

        else:

            compounds = self.channel_map.get_compounds(
                sample_sequence.sequence[sample_nr].open_valves
            )
            compounds = [
                compound
                for compound in compounds
                if compound != 'FS'
            ]
            compounds = misc.sample_label(*compounds) or 'Cell Control'

        return compounds


    def plot_samples_cycles(
            self,
            # these type hints look pretty terrible
            ylim: tuple[(float | None,) * 2] = (None, None),
            samples: list[str] | None = None,
        ) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
        """
        Creates a plot with raw data for the individual samples and cycles.

        Args:
            samples:
                Restrict the figure to these samples only. A list of sample
                names, to see all sample names, refer to the ``name`` column
                of the ``sample_df``.

        Returns:
            A ``Figure`` and an ``Axes`` object with a matrix of samples
            vs. cycles, raw data of one sample, with its detected plugs,
            shown on each panel.
        """

        all_samples = self.sample_df.name.unique()
        names = samples or all_samples
        cycles = sorted(self.sample_df.cycle_nr.unique())

        missing = set(names) - set(all_samples)

        if missing:

            module_logger.warn(
                'Requested to plot raw data of sample(s) '
                f'that do not exist: {", ".join(missing)}.'
            )

        sample_cycle_fig, sample_cycle_ax = plt.subplots(
            nrows = len(names),
            ncols = len(cycles),
            figsize = (7 * len(cycles), 5 * len(names)),
            squeeze = False,
        )

        y_max = self.sample_df.readout_peak_median.max() * 1.1

        ylim = (
            ylim[0] or 0,
            ylim[1] or y_max
        )

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
                sample_cycle_ax[idx_y][idx_x].set_ylim(ylim)

        sample_cycle_fig.tight_layout()
        sample_cycle_fig.subplots_adjust(left = .15)

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

        peak_data = self.sample_df[
            (self.sample_df.cycle_nr == cycle_nr) &
            (self.sample_df.name == name)
        ]

        if len(peak_data) == 0:
            axes.text(0.5, 0.5, 'No Data')
        else:
            axes = self.plot_plug_pmt_data(
                axes = axes,
                cut = (
                    peak_data.start_time.min() - offset,
                    peak_data.end_time.max() + offset
                ),
            )

        axes.set_title(f'{name} • Cycle {cycle_nr + 1}', fontsize = 24)
        plt.setp(axes.get_xticklabels()[0], visible = False)

        return axes


    # QC Plots
    def plot_medium_control_trends(
            self,
            ax: mpl.axes.Axes,
            by_sample: bool = False,
        ) -> mpl.axes.Axes:
        """
        Plots a scatter plot with readout medians for the media control
        over the experiment time.

        Args:
            axes:
                Axes object to draw on.
            by_sample:
                True to plot swarmplot by sample number.

        Return:
            Axes object with the plot.
        """

        self.seaborn_setup()

        if self.normalize_using_control:

            readout_column = 'readout_per_control'
            ylab = 'Readout control ratio'

        else:

            readout_column = 'readout_peak_median'
            ylab = 'Readout median'

        plot_data = self.medium_only()

        with warnings.catch_warnings():

            warnings.simplefilter('ignore')
            # stupid pandas again
            plot_data['cycle_nr'] = plot_data['cycle_nr'] + 1

        if by_sample:

            with warnings.catch_warnings():

                warnings.simplefilter('ignore')

                ax = sns.swarmplot(
                    x = 'sample_nr',
                    y = readout_column,
                    data = plot_data,
                    ax = ax,
                    hue = 'cycle_nr',
                    dodge = True,
                    palette = list(self.palette),
                    size = self.scatter_dot_size,
                )

            ax.set_xlabel('Sample')
            ax.get_legend().set_title('Cycle')

        else:

            lm = self.baseline_lm(col = readout_column)

            with warnings.catch_warnings():

                warnings.simplefilter('ignore')

                ax = sns.scatterplot(
                    x = 'start_time',
                    y = readout_column,
                    data = plot_data,
                    ax = ax,
                    color = self.palette[0],
                    s = self.scatter_dot_size * 10,
                    alpha = .66,
                )

            misc.plot_line(lm.slope, lm.intercept, ax)

            label = ax.text(
                0.7,
                0.9,
                f'R²: {round(lm.rvalue, 2)}',
                transform = ax.transAxes,
            )

            label.set_bbox(dict(facecolor = 'white', alpha = 0.8))
            ax.set_xlabel('Time [s]')

        if self.config.figure_titles:

            ax.set_title('Values from medium control plugs')

        ax.set_ylabel(ylab)

        return ax


    @property
    def has_medium_control(self):

        return bool(len(self.medium_only()))


    def add_length_column(self, df = 'plug'):
        """
        Creates a new column ``length`` in ``sample_df`` with the difference
        of plug start and end times.

        Args:
            df (str): Which data frame add the column to. Either "plug" or
                "sample".
        """

        attr = self._check_df(df, 'add_length_column')

        if attr:

            setattr(
                self,
                attr,
                self.lengths(getattr(self, attr)),
            )


    def add_volume_column(self, flow_rate: float = None, df = 'plug'):
        """
        Creates a new column ``volume`` in ``sample_df`` with the volumes of
        the plugs in nanolitres.

        Args:
            flow_rate (float): The flow rate used at the data acquisition
                in microlitres per hour.
            df (str): Which data frame add the column to. Either "plug" or
                "sample".
        """

        attr = self._check_df(df, 'add_volume_column')

        if attr:

            setattr(
                self,
                attr,
                self.volumes(getattr(self, attr), flow_rate = flow_rate),
            )


    def _check_df(self, df, fun):
        """
        Args:
            df (str): Which data frame: either "plug" or "sample".
            fun (str): Name of the calling function (to be shown in messages).
        """

        if df not in ('plug', 'sample'):

            module_logger.error(
                '`%s`: df must be either "plug" or "sample".' % fun
            )
            return

        attr = '%s_df' % df

        if not hasattr(self, attr):

            module_logger.error(
                '`%s`: %s data frame does not exist yet.' % (fun, attr)
            )
            return

        return attr


    def plot_length_bias(self, col_wrap: int = 3) -> sns.FacetGrid:
        """
        Plots each plugs fluorescence over its length grouped by valve.
        Also fits a linear regression to show if there is a correlation
        between the readout and the plug length indicating non ideal mixing.

        :param col_wrap: After how many subplots the column should be wrapped.
        :return: sns.FacetGrid object with the subplots
        """

        self.add_length_column(df = 'sample')

        df = self.sample_df

        df = df.groupby('name').filter(lambda x: x.shape[0] > 1)

        length_bias_plot = sns.lmplot(
            x = "length",
            y = "readout_peak_median",
            col = "name",
            data = df,
            col_wrap = col_wrap,
            scatter_kws = {
                'color': self.palette[0],
            },
            line_kws = {
                'color': self.palette[1],
            },
            height = 3.8,
            aspect = 1.1,
        )
        length_bias_plot.set_xlabels("Plug length [s]")
        length_bias_plot.set_ylabels("Readout [AU]")
        length_bias_plot.set_titles('{col_name}')

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
                '%s [%% of mean]' % (
                    channel_x.split('_')[0].capitalize()
                )
            )
            axes.set_ylabel(
                '%s [%% of mean]' % (
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
                '%s [AU]' % channel_x.split('_')[0].capitalize()
            )
            axes.set_ylabel(
                '%s [AU]' % channel_y.split('_')[0].capitalize()
            )

        return contamination_plot


    def time_drift(
            self,
            ax: mpl.axes.Axes,
            var: str,
        ) -> mpl.axes.Axes:
        """
        Plots a scatter plot of one variable in the sample data frame against
        experiment time, with a linear regression line.

        Args:
            ax:
                Axes to draw on.
            var:
                Continuous variable in the samples data frame, to be mapped
                to the y axis.

        Return:
            Axes object with the plot.
        """

        ax = sns.regplot(
            x = 'start_time',
            y = var,
            data = self.sample_df,
            ax = ax,
            color = self.palette[0],
            scatter_kws = {
                'alpha': .33,
            },
            line_kws = {
                'color': self.palette[1],
            },
        )

        var_label = self._label(var)
        var_title = self._label(var, unit = False)

        if self.config.figure_titles:

            ax.set_title(f'{var_title}: time bias')

        ax.set_ylabel(var_label)
        ax.set_xlabel('Time [s]')

        return ax


    def violin_by_cycle(
            self,
            ax: mpl.axes.Axes,
            var: str,
        ) -> mpl.axes.Axes:
        """
        Violin plot of one variable in the samples data frame, grouped by
        experiment cycle.

        Args:
            ax:
                Axes object to draw on.
            var:
                Continuous variable in the samples data frame, to be
                mapped to the y axis.

        Return:
            Axes object with the plot.
        """

        ax = misc.seaborn_violin_fix(
            x = self.sample_df.cycle_nr + 1,
            y = self.sample_df[var],
            ax = ax,
            color = self.palette[0],
            box_color = 'white',
            midpoint_color = self.palette[0],
            violin_border_width = 0,
        )

        if self.config.figure_titles:

            var_ttl = self._label(var, unit = False)
            ax.set_title(f'{var_ttl}: cycle bias')

        ax.set_ylabel(self._label(var))
        ax.set_xlabel('Cycle')

        return ax


    @staticmethod
    def shorten_names(names):
        """
        Shortens the drug names to fit better on figures.
        """

        shorten = lambda name: ' + '.join(
            n[:5].strip()
            for n in name.split(' + ')
        )

        return [shorten(name) for name in names]


    def violin_by_sample(
            self,
            ax: mpl.axes.Axes,
            var: str,
        ) -> mpl.axes.Axes:
        """
        Violin plot of one variable in the sample data frame, grouped by
        samples.

        Args:
            ax:
                The axes to draw on.
            var:
                Continuous variable to be mapped to the y axis.

        Return:
            Axes with the plot.
        """

        names = self.shorten_names(self.sample_df.name)

        common_args = dict(
            x = names,
            y = var,
            data = self.sample_df,
            ax = ax,
        )

        ax = sns.violinplot(
            color = self.palette[0],
            linewidth = 0,
            width = .97,
            **common_args
        )

        ax = sns.boxplot(
            showbox = False,
            showfliers = False,
            showcaps = False,
            whiskerprops = {'linewidth': 0.0},
            medianprops = {'color': 'white'},
            **common_args
        )

        if self.config.figure_titles:

            var_ttl = self._label(var, unit = False)
            ax.set_title(f'{var_ttl} by sample')

        ax.set_ylabel(self._label(var))
        ax.set_xlabel('Sample')

        for tick in ax.get_xticklabels():

            tick.set_rotation(90)

        return ax


    def boxplot_by_sample(
            self,
            ax: mpl.axes.Axes,
            var: str,
        ) -> mpl.axes.Axes:
        """
        Boxplot of one variable in the sample data frame, grouped by
        samples.

        Args:
            ax:
                The axes to draw on.
            var:
                Continuous variable to be mapped to the y axis.

        Return:
            Axes with the plot.
        """

        # pandas is such a disaster...
        data = self.sample_df.copy()
        data['the_cycle'] = data.cycle_nr.map(lambda n: f'Cycle {n + 1}')

        if self.ncycles > 1:

            data = pd.concat((
                data,
                data.copy().assign(the_cycle = 'All'),
            ))
        data = self.by_cycle_and_all(
            df = self.sample_df,
            label_col = 'the_cycle',
        )

        # pandas is just ridiculous
        data['the_cycle'] = data['the_cycle'].astype('category')
        data['the_cycle'].cat.reorder_categories(
            sorted(data['the_cycle'].cat.categories)
        )

        data['short_name'] = self.shorten_names(data.name)

        ax = sns.boxplot(
            x = 'short_name',
            y = var,
            data = data,
            ax = ax,
            hue = 'the_cycle',
            palette = self.palette,
            linewidth = 1.,
            width = .97,
        )

        self._bpax = ax

        ax = misc.vstripes(ax)

        lh, ll = ax.get_legend_handles_labels()
        ax.legend_.remove()

        legend_args = misc.LEGEND_STYLE.copy()
        legend_args['handletextpad'] = 1.

        ax.legend(
            lh,
            ll,
            loc = 2,
            ncol = self.ncycles + 1 * (self.ncycles > 1),
            **legend_args
        )

        if self.config.figure_titles:

            var_ttl = self._label(var, unit = False)
            ax.set_title(f'{var_ttl} by sample')

        ax.set_ylabel(self._label(var))
        ax.set_xlabel('Sample')

        for tick in ax.get_xticklabels():

            tick.set_rotation(90)

        return ax


    @staticmethod
    def by_cycle_and_all(
            df: pd.DataFrame,
            label_col: str | None = None,
        ) -> pd.DataFrame:
        """
        Creates a data frame with records both by cycle and in a pool of
        all records. Records are duplicated in this data frame that is
        required for visualization purposes.
        """

        data = df.copy()

        if data.cycle_nr.nunique() > 1:

            # pandas is such a disaster...
            data_all = data.copy()
            data_all['cycle_nr'] = np.nan
            data = pd.concat((data, data_all))

        if label_col:

            data[label_col] = data.cycle_nr.map(
                lambda n: 'All' if np.isnan(n) else f'Cycle {int(n + 1)}'
            )

        return data


    def scatter(
            self,
            ax: mpl.axes.Axes,
            y: str,
            x: str,
        ) -> mpl.axes.Axes:
        """
        Scatter plot between two variables in the sample data frame. Sample
        sequence index and experiment cycle are mapped to hue and shape,
        respectively.

        Args:
            ax:
                Axes object to draw on.
            y:
                Name of a continuous variable in the samples data frame.
            x:
                Name of a continuous variable in the samples data frame.

        Return:
            Axes object with the plot.
        """

        labels = {
            'sample_nr': 'Sample',
            'cycle_nr': 'Cycle',
        }

        ax = sns.scatterplot(
            x = x,
            y = y,
            hue = 'sample_nr',
            style = self.sample_df.cycle_nr + 1,
            alpha = .7,
            data = self.sample_df,
            ax = ax,
        )

        if self.config.figure_titles:

            xttl = self._label(x, unit = False)
            yttl = self._label(y, unit = False)
            ax.set_title(f'{yttl}-{xttl} correlation')

        ax.set_xlabel(self._label(x))
        ax.set_ylabel(self._label(y))

        lh, ll = ax.get_legend_handles_labels()
        ax.legend_.remove()
        ll = [labels.get(l, l) for l in ll]

        ax.legend(
            lh,
            ll,
            loc = 4,
            **misc.LEGEND_STYLE
        )

        ax.set_ylim(
            self.sample_df[y].min() * 0.95,
            self.sample_df[y].max() * 1.05,
        )
        ax.set_xlim(
            self.sample_df[x].min() * 0.95,
            self.sample_df[x].max() * 1.5,
        )

        return ax


    def save(self, file_path: pl.Path):
        """
        Saves this PlugData object as pickle
        :param file_path: Path to the file to write to
        """
        with file_path.open("wb") as f:
            pickle.dump(self, f)


    def plot_compound_violins(
            self,
            ax: mpl.axes.Axes,
            column_to_plot: str = 'readout_peak_z_score',
            cycle: int = None,
        ) -> sns.FacetGrid:
        """
        Plots a violin plot per compound combination
        :param column_to_plot: Column to be plotted (e.g.
            readout_peak_z_score, readout_per_control_z_score).
        :param by_cycle: Produce separate plots for each cycle.
        :return: seaborn.FacetGrid object with the plot
        """

        self._check_sample_df_column(column_to_plot)

        data = self.sample_df

        if cycle is not None:

            data = data[data.cycle_nr == cycle]

        names = (
            self.shorten_names(data.name)
                if self.short_labels else
            data.name
        )

        misc.seaborn_violin_fix(
            x = names,
            y = column_to_plot,
            data = data,
            color = self.palette[0],
            width = 1.0,
            linewidth = 0,
            box_linewidth = 0.7,
            box_color = 'white',
            midpoint_color = self.palette[0],
            violin_border_width = 0,
            ax = ax,
        )

        ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
        ax.set_ylabel('Readout z-score')
        ax.set_xlim(-1, data.name.nunique())

        return ax


    def samples_heatmap(
            self,
            var: str,
            annotation_df: pd.DataFrame = None,
            annotation_column: str = 'stars',
            by_cycle: bool = False,
            center: float | str | tuple[float] | None = None,
            ax: mpl.axes.Axes | None = None,
            **kwargs
        ) -> sns.FacetGrid | mpl.axes.Axes:
        """
        Plots a heatmap to visualize the different combinations.

        Args:
            var:
                A variable in the samples data frame, to be mapped to hue.
            annotation_df:
                Data frame grouped by column `compound_a` and
                `compound_b` for annotation.
            annotation_column:
                Which column in annotation_df to use for the annotation.
            by_cycle:
                Produce separate plots for each cycle.
            center:
                Center the color scale at these values or the median
                of these samples. Can be a tuple if plotting by cycle.

        Return:
            An object with the figure, if axes has been provided, it is
            returned, otherwise a grid. Warning: if ``by_cycle`` is *True*,
            axes is provided, and the experiment has more than one cycles,
            cycles will be plotted over each other on the same axes, which
            is not a desired behaviour.
        """

        self._check_sample_df_column(var)

        assert var not in {'compound_a', 'compound_b'},\
            f'You can not plot this coulumn on a heatmap: `{var}`.'

        if annotation_df is not None:

            annotation_df = annotation_df.reset_index()
            annotation_df = annotation_df.pivot(
                'compound_a',
                'compound_b',
                annotation_column,
            )

        cycles = self.cycles if by_cycle else (None,)

        second_scale = (
            self.heatmap_second_scale in set(self.sample_df.compound_a) or
            self.heatmap_second_scale in set(self.sample_df.compound_b)
        )

        aspect_correction = 1.3 if second_scale else 1.0

        # setting up divergent palette center value
        if not center and self.config.heatmap_center_scale:

            center = tuple(
                self.medium_only(cycle = cycle)[self.readout_col].median()
                for cycle in cycles
            )

        center = misc.to_tuple(center) or (None,)

        if all(isinstance(k, str) for k in center):

            center = tuple(
                self.cycle(
                    cycle,
                    df = self.sample(*center)
                )[self.readout_col].median()
                for cycle in cycles
            )

        center = center * len(cycles) if len(center) == 1 else center

        if len(center) != len(cycles):

            msg = (
                'heatmap: `center` should be either one value, '
                'or should contain a value for each cycle.'
            )
            module_logger.error(msg)
            raise ValueError(msg)

        if ax is None:

            # creating the empty grid
            grid = sns.FacetGrid(
                data = self.sample_df,
                col = 'cycle_nr' if by_cycle else None,
                height = 7,
                aspect = (
                    (.5 * len(cycles) if by_cycle else 1) *
                    aspect_correction
                ),
            )

        for (i, cycle), _center in zip(enumerate(cycles), center):

            data = self._compound_heatmap_data(
                col = var,
                cycle = cycle,
                df = self.sample_df,
            )

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

            ax = grid.axes.flat[i] if ax is None else ax

            vmin, vmax = self.heatmap_override_scale or (None, None)

            sns.heatmap(
                data,
                annot = annotation_df,
                fmt = '',
                ax = ax,
                vmin = vmin,
                vmax = vmax,
                center = _center,
                cmap = self._heatmap_cmap(1, _center),
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
                    fmt = '',
                    ax = ax,
                    vmin = vmin,
                    vmax = vmax,
                    center = _center,
                    cmap = self._heatmap_cmap(2, _center),
                    **kwargs
                )

            cycle_str = (' • Cycle %u' % (cycle + 1)) if by_cycle else ''
            var_label = self._label(var)
            ax.set_title(f"{var_label} {cycle_str}")
            ax.set_ylabel("")
            ax.set_xlabel("")

            if plt.matplotlib.__version__ == '3.1.1':

                ylim = ax.get_ylim()
                ax.set_ylim(ylim[0] + .5, ylim[1] - .5)

        return grid if 'grid' in locals() else ax


    def _heatmap_cmap(self, idx: int, center: float | None):

        return (
            self.config[f'heatmap{"_second" if idx == 2 else ""}_cmap'] or
            (
                self.config[f'continuous_palette_{idx}']
                    if center is None else
                self.config.diverging_palette(idx)
            )
        )


    def _compound_heatmap_data(
            self,
            col: str, cycle: int | None = None,
            df: pd.DataFrame | None = None,
        ) -> pd.DataFrame:

        df = self.data if df is None else df
        df = df if cycle is None else df[df.cycle_nr == cycle]
        df = df[[col, 'compound_a', 'compound_b']]
        df = df.groupby(['compound_a', 'compound_b']).mean()
        df = df.reset_index()
        df = df.pivot('compound_a', 'compound_b', col)
        df = df.reindex(
            df.isna().
            sum(axis = 1).
            sort_values(ascending = False).
            index.
            to_list()
        )
        df = df[
            df.isna().
            sum(axis = 0).
            sort_values(ascending = True).
            index.
            to_list()
        ]

        return df


    def _check_sample_df_column(self, column: str):
        """
        Checks if column is in sample data frame.

        Args:
            column:
                The column to check.
        """

        try:

            msg = (
                f"Column {column} not in the column names of sample_df"
                f"({self.sample_df.columns.to_list()}), specify a column "
                "from the column names!"
            )

            assert column in self.sample_df.columns.to_list(), msg

        except AssertionError:

            module_logger.critical(msg)

            raise


    def _baseline_adjust(self) -> pd.DataFrame:
        """
        Regresses out the baseline from the readout signal in ``sample_df``.
        The baseline is defined by the level of the medium only samples.
        As these are repeated several times over the experiment, their drift
        can be captured by a linear regression, and in order to adjust the
        readout signal by regressing out this drift.
        """

        if self.normalize_using_media_control_lin_reg:

            sample_df = self.sample_df

            readout_column = (
                'readout_per_control'
                    if self.normalize_using_control else
                'readout_peak_median'
            )

            lm = self.baseline_lm(col = readout_column)

            if lm:

                sample_df = sample_df.assign(
                    readout_media_norm = (
                        sample_df[readout_column] /
                        lm[sample_df['start_time']]
                    )
                )

                sample_df = self._z_score(sample_df, 'readout_media_norm')

            else:

                module_logger.warning(
                    'Could not find medium only control samples, unable to '
                    'adjust readout values based on the drift of these '
                    'samples. If the experiment contains such samples, '
                    'check the config value of `medium_control_label` '
                    f'(currently `{self.med_ctrl_lab}`).'
                )

                new_readout_analysis_column = (
                    'readout_per_control_z_score'
                        if self.normalize_using_control else
                    'readout_z_score'
                )

                module_logger.critical(
                    'Using the column `%s` instead of `%s` in the analysis, '
                    'because the correction by medium only samples '
                    'failed.' % (
                        new_readout_analysis_column,
                        self.config.readout_analysis_column,
                    )
                )

                self.config.readout_analysis_column = (
                    new_readout_analysis_column
                )

            self.sample_df = sample_df


    def _z_score(
            self,
            df: pd.DataFrame,
            col: str,
            loc: int | None = None,
        ) -> pd.DataFrame:

        loc = df.shape[1] if loc is None else loc
        zcol = f"{re.sub('_median$', '', col)}_z_score"

        def _z_score(df, col):

            df.insert(
                column = zcol,
                value = stats.zscore(df[col]),
                loc = loc,
            )

            return df


        by_cycle = self.config.z_scores_by_cycle

        return (
            pd.concat(
                _z_score(grp.copy(), col)
                for _, grp in df.groupby('cycle_nr').groups
            )
                if by_cycle else
            _z_score(df, col)
        )


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


    def length_grid(self) -> sns.PairGrid:
        """
        Creates a `seaborn.PairGrid` figure of plug lengths and readout
        values (https://seaborn.pydata.org/tutorial/distributions.html).

        Returns:
            (seaborn.PairGrid): A `PairGrid` object which can be saved
                into a matplotlib figure.
        """

        data = self.sample_and_barcode_plugs
        data = self.lengths(data)

        labels = {
            'length': 'Length [s]',
            'start_time': 'Time [s]',
            'barcode_peak_median': 'Barcode [AU]',
            'control_peak_median': 'Control [AU]',
            'readout_peak_median': 'Readout [AU]',
            'readout_per_control': 'R:C ratio',
            '': '',
        }

        legend_labels = {
            'barcode': '',
            'False': 'Sample plugs',
            'True': 'Barcode plugs',
            'cycle_nr': '',
            '0': 'Cycle 1',
            '1': 'Cycle 2',
            '2': 'Cycle 3',
            '3': 'Cycle 4',
        }

        grid = sns.PairGrid(
            data = data,
            hue = 'barcode',
            vars = [
                'length',
                'start_time',
                'barcode_peak_median',
                'control_peak_median',
                'readout_peak_median',
                'readout_per_control',
            ],
            palette = self.config.palette,
            despine = False,
        )

        grid.map_upper(sns.scatterplot, style = data.cycle_nr)
        grid.map_lower(sns.kdeplot, fill = True)
        grid.map_diag(sns.histplot, kde = False)

        for ax in grid.axes.flatten():

            # seaborn is crazy
            ax.get_yaxis().set_label_coords(-0.4,0.5)
            ax.yaxis.set_label_text(labels[ax.yaxis.get_label_text()])
            ax.xaxis.set_label_text(labels[ax.xaxis.get_label_text()])

        # seaborn is crazy
        grid.add_legend(title = '', bbox_to_anchor=(1.2, .5))

        # shameful seaborn
        for l in grid.fig.legends:

            for t in l.get_texts():

                if t.get_text() in legend_labels:

                    t.set_text(legend_labels[t.get_text()])

        return grid


    def heatmap_matrix(self, **kwargs) -> mpl.figure.Figure:
        """
        Composite figure of compound-by-compound heatmaps: a grid of
        cycles vs. channels and derived variables. ``**kwargs`` is passed
        to ``seaborn.heatmap``.
        """

        df = self.data

        variables = (
            [f'{ch}_peak_median' for ch in self.channels] +
            [
                'readout_per_control',
                'readout_media_norm',
            ]
        )

        variables = [v for v in variables if v in df.columns]

        fig = plt.figure(
            figsize = (
                7 + self.ncycles * 5,
                len(variables) * 5
            ),
            constrained_layout = False,
        )

        gs = fig.add_gridspec(
            nrows = len(variables),
            ncols = self.ncycles + 1,
        )

        for i, var in enumerate(variables):

            var_str = var.replace('_', ' ').capitalize()

            vmin = df[var].min()
            vmax = df[var].max()

            center = (
                self.medium_only()[var].median()
                    if self.config.heatmap_center_scale else
                None
            )

            cmap = self._heatmap_cmap(1, center)

            for j, cycle in enumerate(itertools.chain(self.cycles, (None,))):

                ax = fig.add_subplot(gs[i, j])
                data = self._compound_heatmap_data(
                    col = var,
                    cycle = cycle,
                    df = df,
                )

                sns.heatmap(
                    data,
                    ax = ax,
                    fmt = '',
                    vmin = vmin,
                    vmax = vmax,
                    center = center,
                    cmap = cmap,
                    xticklabels = i == len(variables) - 1,
                    yticklabels = j == 0,
                    cbar = j == self.ncycles,
                    **kwargs
                )

                cycle_str = 'All' if cycle is None else f'Cycle #{cycle}'
                ax.set_ylabel(var_str if j == 0 else '')
                ax.set_xlabel(cycle_str if i == len(variables) - 1 else '')

        return fig


    @property
    def sample_and_barcode_plugs(self):
        """
        Returns the plug data frame with the plugs marked for discard removed,
        and the sample and barcode plugs kept.
        """

        data = self.plug_df.loc[
            np.logical_not(self.plug_df.discard) |
            self.plug_df.barcode
        ]

        return data


    @staticmethod
    def lengths(df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a new column with plug or sample length (depending on the
        input data frame). The length is the difference of the start and end
        times in seconds.

        Args:
            df (pandas.DataFrame): A data frame with `start_time` and
                `end_time` columns.

        Returns:
            (pandas.DataFrame): A copy of the data frame with a new column
                called `length`.
        """

        df = df.copy()

        # these warnings stuff just filter out the pandas setting
        # with copy warnings, which is an annoying example of the
        # bad design of pandas
        with warnings.catch_warnings():

            warnings.simplefilter('ignore')
            df['length'] = df.end_time - df.start_time

        return df


    @misc.class_or_instancemethod
    def volumes(
            cls_self,
            df: pd.DataFrame | None = None,
            flow_rate: float | None = None,
        ) -> pd.DataFrame:
        """
        Creates a new column with plug volume. The volumes are calculated by
        the length of the plugs, if the data frame has no `length` column,
        it will be added too.

        Args:
            df (pandas.DataFrame): A data frame with `start_time` and
                `end_time` columns.
            flow_rate (float): The flow rate used at the data acquisition
                in microlitres per hour. If not provided the value from
                the config will be used.

        Returns:
            (pandas.DataFrame): A copy of the data frame with a new column
                called `volume` containing the plug volumes in nanolitres.
                If the input data frame does not have a column `length`
                that will be added too.
        """

        if isinstance(cls_self, type):

            if df is None:

                msg = 'volumes: no data frame provided'
                module_logger.error(msg)

                raise ValueError(msg)

            if flow_rate is None:

                flow_rate = PlugyConfig.flow_rate
                msg = (
                    'volumes: no flow rate provided, '
                    f'using the default from config ({flow_rate})'
                )
                module_logger.warn(msg)
                warnings.warn(msg)

        else:

            df = cls_self.data if df is None else df
            flow_rate = flow_rate or cls_self.config.flow_rate

        df = df.copy()

        if 'length' not in df.columns:

            df = cls_self.lengths(df)

        # dividing by 3600 for hours -> seconds conversion
        # multiplying by 1000 for microlitres -> nanoliters conversion
        df['volume'] = df.length * flow_rate / 3.6

        return df


    def volume_stats(self, flow_rate: float | None = None) -> dict:
        """
        Prints and returns statistics about the volume of the sample plugs.

        Args:
            flow_rate:
                The flow rate used at the data acquisition
                in microlitres per hour.

        Return:
            A dictionary with cycle numbers and the string "all" as
            keys and dictionaries of statistics about the sample plug
            volumes as values.
        """

        data = self.volumes(self.sample_df, flow_rate = flow_rate)

        result = {
            'all': self._stats(data.volume)
        }

        for cycle in data.cycle_nr.unique():

            result[cycle] = self._stats(data[data.cycle_nr == cycle].volume)

        return result


    def size_density(
            self,
            volume: bool = False,
            flow_rate: float | None = None,
            boxplot: bool = False,
        ) -> sns.FacetGrid:
        """
        Density plot of plug sizes, either lengths in seconds or volumes
        in nanolitres.

        Args:
            volume:
                Use plug volumes in nanolitres instead of lengths in seconds.
            flow_rate:
                The flow rate used at the data acquisition in microlitres
                per hour.
            boxplot:
                Create boxplot instead of density plot.

        Return:
            A grid with 3 subplots, each a density plot of plug sizes:
            barcode plugs, sample plugs and both together.
        """

        data = self.sample_and_barcode_plugs
        data = self.volumes(df = data, flow_rate = flow_rate)

        # these warnings stuff just filter out the pandas setting
        # with copy warnings, which are an annoying example of the
        # bad design of pandas
        with warnings.catch_warnings():

            warnings.simplefilter('ignore')

            data['group'] = [
                'Barcode' if bc else 'Sample'
                for bc in data.barcode
            ]

        data_both = data.copy()
        data_both['group'] = 'Both'
        data = pd.concat([data, data_both])

        if boxplot:

            data = self.by_cycle_and_all(data, label_col = 'the_cycle')

        # stupid seaborn
        data['dummy_x'] = ''

        lab = 'Plug %s [%s]' % (
            'volume' if volume else 'length',
            'nl' if volume else 's',
        )

        grid = sns.FacetGrid(
            data,
            col = 'group',
            despine = False,
            height = 4,
        )

        if boxplot:

            grid.map(
                sns.boxplot,
                # fantastic api design from seaborn:
                # aesthetics can be passed only as positional args
                # not as kwargs
                # this way it's not possible to pass empty x,
                # neither some variables as vectors or scalars,
                # elsewhere seaborn made it mandatory to use keyword
                # arguments exactly bc positional args are more error
                # prone; furthermore, this behaviour is not documented,
                # and result only in obscure errors in downstream functions.
                # `map_dataframe` accepts variables as keyword arguments,
                # but they can be only column names, not vectors, neither
                # None. painful.
                # thanks seaborn for this great design!
                # next time I will think twice before relying on seaborn,
                # writing stuff from scratch would have taken fraction of
                # the time
                'dummy_x',
                'volume' if volume else 'length',
                'the_cycle',
                order = [''],
                hue_order = sorted(data.the_cycle.unique()),
                palette = self.palette,
                linewidth = 1.,
                width = .97,
            )

            # seaborn is so easy to use
            grid.add_legend(
                bbox_to_anchor=(1., .8),
                loc = 2,
                borderaxespad = 0.,
            )

            xlab = ''
            ylab = lab

        else:

            grid.map(
                sns.histplot,
                'volume' if volume else 'length',
                kde = True,
                color = self.palette[0],
                line_kws = {'color': self.palette[1]},
            )

            xlab = lab
            ylab = 'Density'

        # stupid seaborn
        data.drop('dummy_x', axis = 1)

        for ax in grid.axes.flatten():

            # I can not believe seaborn has no better solution
            # for all these trivial tasks
            ax.set_title(
                ax.get_title().split('=')[-1].strip()
            )

        grid.axes.flatten()[0].yaxis.set_label_text(ylab)
        grid.axes.flatten()[0].xaxis.set_label_text('')
        grid.axes.flatten()[1].xaxis.set_label_text(xlab)
        grid.axes.flatten()[2].xaxis.set_label_text('')

        return grid


    def univar_overview(
            self,
            var: str = None,
            var2: str = 'readout_peak_median',
            var3: str = 'control_peak_median',
            fig: mpl.figure.Figure | None = None,
        ) -> mpl.figure.Figure:
        """
        Creates a composite figure to investigate one variable and discover
        potential bias or drift across time, cycles, samples, or in relation
        to another variable. It is especially suitable to see if the control
        channel shows a bias with other variables or correlation with the
        readout channel.

        Args:
            var:
                A variable name in the plug or sample data frame.
            var2:
                Another variable: it will be used only for one panel,
                a scatter plot ``var`` vs. ``var2``.
        """

        if fig is None:

            fig = plt.figure(
                figsize = (25, 10),
                constrained_layout = False,
            )

        gs = fig.add_gridspec(
            nrows = 2,
            ncols = 5,
            width_ratios = [1.] * 4 + [1.5],
        )

        ax_violin = fig.add_subplot(gs[0, :])
        ax_time = fig.add_subplot(gs[1, 0])
        ax_cycle = fig.add_subplot(gs[1, 1])
        ax_scatter1 = fig.add_subplot(gs[1, 2])
        ax_scatter2 = fig.add_subplot(gs[1, 3])
        ax_heatmap = fig.add_subplot(gs[1, 4])

        ax_violin = self.boxplot_by_sample(ax = ax_violin, var = var)
        ax_time = self.time_drift(ax = ax_time, var = var)
        ax_cycle = self.violin_by_cycle(ax = ax_cycle, var = var)
        ax_scatter1 = self.scatter(ax = ax_scatter1, y = var2, x = var)
        ax_scatter2 = self.scatter(ax = ax_scatter2, y = var3, x = var)

        ax_heatmap = self.samples_heatmap(
            var = var,
            ax = ax_heatmap,
        )

        return fig


    def sample_sd_df(self) -> pd.DataFrame:
        """
        Creates a data frame of readout standard deviations and coefficients
        of variation (CV).
        """

        cv = lambda x: np.std(x) / np.mean(x) * 100.
        cv_stat = {'label': 'cv', 'fun': cv}

        return self.sample_stats(
            readout_peak_median = ('std', cv_stat),
            readout_per_control = ('std', cv_stat),
            readout_media_norm = ('std', cv_stat),
        )


    def sample_stats(self, long: bool = True, **kwargs) -> pd.DataFrame:
        """
        Calculates statistics for each sample in the samples data frame.

        Args:
            long:
                Create long or wide format data frame.
            kwargs:
                A mapping where keys are column names while values are
                aggregate function names or functions. For one column more
                than one statistics can be calculated by passing a tuple
                as value.

        Returns:
            (pandas.DataFrame): A data frame with each row representing
                one statistics for one sample, or each row representing
                one sample with all its statistics, if `long` is False.
        """

        agg_args = dict(
            (
                '%s___%s' % (
                    colname,
                    (
                        fun
                            if isinstance(fun, str) else
                        fun.__name__
                            if (
                                hasattr(fun, '__name__') and
                                fun.__name__ != '<lambda>'
                            ) else
                        fun['label']
                            if (
                                isinstance(fun, dict) and
                                'label' in fun
                            ) else
                        'agg%u' % i
                    )
                ),
                pd.NamedAgg(
                    colname,
                    fun['fun']
                        if isinstance(fun, dict) and 'fun' in fun else
                    fun,
                )
            )
            for colname, funs in kwargs.items()
            for i, fun in enumerate(
                funs
                    if isinstance(funs, tuple) else
                (funs,)
            )
            if colname in self.sample_df.columns
        )

        data = (
            self.sample_df.
            groupby(['cycle_nr', 'sample_nr']).
            agg(**agg_args)
        )

        if long:

            data = data.reset_index().melt(
                id_vars = ['cycle_nr', 'sample_nr'],
                var_name = 'var_stat',
                value_name = 'value',
            )

            data['stat'] = [v.split('___')[-1] for v in data['var_stat']]
            data['var'] = [v.split('___')[0] for v in data['var_stat']]

        return data


    def stats(
            self,
            cycle: int | None = None,
            extra_cols: list[str] = None,
        ) -> pd.DataFrame:
        """
        Calculates statistics for each sample: means, standard deviations,
        Wilcoxon tests, their adjusted p-values and significances.

        Args:
            cycle:
                Use only this experiment cycle. If None, all cycles will be
                used.
            extra_cols:
                Keep also these columns, calculate their means and standard
                deviations.

        Returns:
            A data frame with the sample level statistics, with compounds
            in row multi index.
        """

        module_logger.info('Calculating statistics')

        self.add_length_column(df = 'sample')

        col = self.readout_col

        medium_only = self.medium_only()
        samples = self.sample_df
        samples = samples[~samples.isin(medium_only)].dropna()

        if cycle is not None:

            samples = samples[samples.cycle_nr == cycle]

        by = ['compound_a', 'compound_b', 'name']
        default = ['length', 'baseline', 'negative', 'fc', col]
        extra = [
            c for c in misc.to_tuple(extra_cols)
            if c in samples and c not in by + default
        ]
        samples = samples[by + extra + default]

        sample_stats = samples.groupby(by = by).agg([np.mean, np.std])
        sample_stats.columns = [
            '_'.join(c)
            for c in sample_stats.columns.values
        ]

        p_values = []

        for combination, values in samples.groupby(by = by):

            p_values.append(
                stats.ranksums(
                    x = values[col],
                    y = medium_only[col],
                )[1]
            )

        sample_stats = sample_stats.assign(pval = p_values)

        significance, p_adjusted, _, alpha_corr_bonferroni = (
            multitest.multipletests(
                pvals = sample_stats.reset_index().pval,
                alpha = self.config.alpha,
                method = 'bonferroni',
            )
        )

        stars = [self.config.significance_stars(p) for p in p_adjusted]

        sample_stats = sample_stats.assign(
            p_adjusted = p_adjusted,
            significant = significance,
            stars = stars,
        )

        return sample_stats


    def stats_table(
            self,
            by_cycle: bool = True,
            exp_summary: bool = True,
            extra_cols: bool | list[str] = True,
            sort_by: list[str] | None = None,
            multi_index: bool = True,
        ) -> pd.DataFrame:
        """
        A large table of statistics.

        Args:
            by_cycle:
                Include statistics for each cycle.
            exp_summary:
                Include statistics for the entire experiment.
            extra_cols:
                Include extra columns, besides the defaults. Alternatively,
                a list of column names that overrides the extra columns.
            sort_by:
                Sort the conditions by these column names. For descending
                order prefix the column name with a hat (^). Cycle will
                always be the first column to sort by.
            multi_index:
                Keep conditions in multi index. If `False`, the multi index
                will be turned into regular columns.

        Return:
            A data frame of the statistics with conditions in row multi index.
        """

        extra_cols = (
            list(extra_cols)
                if isinstance(extra_cols, misc.LIST_LIKE) else
            misc.CHANNEL_VARS
                if extra_cols else
            ()
        )

        sort_by = ('cycle_nr',) + misc.to_tuple(sort_by)
        ascending = [not c[0] == '^' for c in sort_by]
        sort_by = [re.sub('^[\^]', '', c) for c in sort_by]

        tables = []

        for cycle in itertools.chain(
            self.cycles if by_cycle else (),
            (None,) if exp_summary else ()
        ):

            tbl = self.stats(cycle = cycle, extra_cols = extra_cols)
            tbl['cycle_nr'] = np.nan if cycle is None else cycle
            tables.append(tbl)

        tables = pd.concat(tables)
        tables.sort_values(by = sort_by, ascending = ascending, inplace = True)

        if not multi_index:

            tables.reset_index(inplace = True)

        return tables


    def _update_baseline(self, baseline: Literal['baseline', 'negative']):

        lm = self.baseline_lm(col = self.readout_col, baseline = baseline)

        self.sample_df[baseline] = (
            np.nan
                if lm is None else
            lm[self.sample_df.start_time]
        )


    def update_baseline(self):
        """
        Adds a column to the samples data frame with the baseline predicted
        by the linear model fit to the baseline (medium only) samples.
        """

        self._update_baseline(baseline = 'baseline')


    def update_negative(self):
        """
        Adds a column to the samples data frame with the zero line predicted
        by the linear model fit to the negative control samples. These
        represent the signal detected at very low levels of the measured
        activity.
        """

        self._update_baseline(baseline = 'negative')


    def update_fc(self):
        """
        Adds a column to the samples data frame with the fold changes. Fold
        changes are calculated by subtracting the zero (negative) level from
        all values and dividing each value by the corresponding baseline.
        """

        samples = self.sample_df

        bline = samples.baseline - samples.negative
        bline_min = bline.min()
        values = samples[self.readout_col] - samples.negative

        if bline_min <= 0:

            bline = bline - bline_min + .001
            values = values - bline_min + .001

        samples['fc'] = values / bline


    def mean_cv(self):
        """
        Calculates the coefficient of variation for all cycles and the
        whole screen, for all available readout metrics.
        """

        cv_data = self.sample_sd_df()
        cv_data = cv_data[cv_data['stat'] == 'cv']

        readout_vars = cv_data['var'].unique()
        cycles = tuple(cv_data['cycle_nr'].unique()) + ('all',)

        result = collections.defaultdict(dict)

        for cycle, var in itertools.product(cycles, readout_vars):

            cv = cv_data[cv_data['var'] == var]

            if cycle != 'all':

                cv = cv[cv['cycle_nr'] == cycle]

            cv = cv.value

            result[cycle][var] = {
                'mean': cv.mean(),
                'ci_low': cv.mean() - cv.std(),
                'ci_high': cv.mean() + cv.std(),
            }

        return dict(result)


    def sample_sd_violin(self) -> sns.FacetGrid:

        def seaborn_violin_fix_fix(*args, **kwargs):
            """
            A tribute to seaborn's genious design idea of passing
            map arguments only as positional, not as keyword args.
            Congratulations!
            """

            x, y = args

            misc.seaborn_violin_fix(
                x = x,
                y = y,
                **kwargs
            )


        labels = {
            'readout_peak_median': 'Readout\n',
            'readout_per_control': 'Readout:control\nratio',
            'readout_media_norm': 'R:C corrected by\nmedium control',
        }

        data = self.sample_sd_df()
        data = data[data.stat == 'std']

        data.cycle_nr = data.cycle_nr + 1

        grid = sns.FacetGrid(
            data,
            col = 'var',
            despine = False,
            height = 5,
        )

        bg_color = self.config.palette[0]
        fg_color = 'white'

        grid.map(
            seaborn_violin_fix_fix,
            # seaborn, is this serious??
            'cycle_nr',
            'value',
            color = bg_color,
            midpoint_color = bg_color,
            box_color = fg_color,
            violin_border_width = 0,
        )

        for ax in grid.axes.flatten():

            # I can not believe seaborn has no better solution
            # for all these trivial tasks
            ax.set_title(
                labels[ax.get_title().split('=')[-1].strip()]
            )

        grid.axes.flatten()[0].yaxis.set_label_text('Standard deviation')
        grid.axes.flatten()[0].xaxis.set_label_text('')
        grid.axes.flatten()[1].xaxis.set_label_text('Cycle')

        if len(grid.axes.flatten()) > 2:

            grid.axes.flatten()[2].xaxis.set_label_text('')

        return grid


    @staticmethod
    def _stats(data: pd.Series) -> dict:
        """
        For a numeric `Series` creates a dictionary of basic descriptive
        statistics, such as mean, median, quantiles, standard deviation.

        Args:
            (pandas.Series): A pandas.Series or numpy.ndarray object
                of numeric data type.

        Returns:
            (dict): A dictionary of descriptive statistics, with strings
                as keys and numbers as values.
        """

        return {
            'mean': data.mean(),
            'median': data.median(),
            'q1': data.quantile(.25),
            'q2': data.quantile(.75),
            'min': data.min(),
            'max': data.max(),
            'ci90_low': data.mean() - 1.645 * data.std(),
            'ci90_high': data.mean() + 1.645 * data.std(),
            'sd': data.std(),
        }


    def sample_sd(self, variable = 'readout_peak_median', silent = False):
        """
        Prints and returns the standard deviation of a variable within
        samples, with confidence intervals.

        Args:
            variable (str): The name of the variable to calculate the
                standard deviations from.

        Returns:
            (dict): A dictionary with cycle numbers and the string "all" as
                keys and dictionaries of statistics about the standard
                deviations of samples as values.
        """

        def sd_stats(data):

            return self._stats(data.value)


        args = {variable: 'std'}
        data = self.sample_stats(**args)
        result = {}

        for cycle in data.cycle_nr.unique():

            result[cycle] = sd_stats(data[data.cycle_nr == cycle])

        result['all'] = sd_stats(data)

        if not silent:

            sys.stdout.write(
                '\nStandard deviation of `%s` within samples:\n\n' % variable
            )
            sys.stdout.write(
                pd.DataFrame(result).__repr__()
            )
            sys.stdout.write('\n\n')
            sys.stdout.flush()

        return result


    def __len__(self) -> int:

        return self.nplugs


    @property
    def nplugs(self) -> int:
        """
        Number of plugs.
        """

        return len(self.plug_df) if hasattr(self, 'plug_df') else 0


    @property
    def nsamples(self) -> int:
        """
        Number of samples.
        """

        return self._numof('sample_nr')

    @property
    def ncycles(self) -> int:
        """
        Nuber of cycles.
        """

        return self._numof('cycle_nr')


    def _numof(self, col: str) -> int:

        return self.plug_df[col].nunique() if hasattr(self, 'plug_df') else 0


    def __repr__(self) -> str:

        return (
            f'<PlugData: {len(self)} plugs, '
            f'{self.nsamples} samples, '
            f'{self.ncycles} cycles>'
        )


    def __getitem__(self, keys) -> pd.DataFrame:

        df = self.data

        if isinstance(keys, misc.SIMPLE_TYPES):

            if isinstance(keys, int):

                return df.iloc[keys]

            elif isinstance(keys, str):

                return self.sample(keys, df = df)

        elif hasattr(keys, '__len__'):

            if all(isinstance(k, int) for k in keys):

                return df.iloc[keys[0], keys[1]]

            elif all(isinstance(k, str) for k in keys):

                return self.sample(*keys, df = df)

        raise ValueError(f'Don\'t know how to process key: {keys}')


    def __contains__(self, other):

        return (
            (
                other in self.compounds or
                other in self.channels or
                other in self.sample_df.name.values
            )
                if isinstance(other, str) and hasattr(self, 'sample_df') else
            self.start < other < self.end
                if isinstance(other, (int, float)) else
            False
        )


    def sample(self, *args, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Selects a sample based on its name or a pair of conditions (compounds).
        """

        df = df if isinstance(df, pd.DataFrame) else self.data

        if len(args) == 1:

            return df[df.name == args[0]]

        elif len(args) >= 2:

            args = misc.to_set(args)

            return df[
                df.compound_a.isin(args) |
                df.compound_b.isin(args)
            ]

        raise ValueError(f'Don\'t know how to process sample name: {args}')


    def cycle(self, *cycle, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Selects cycle(s) based on their number(s).
        """

        df = df if isinstance(df, pd.DataFrame) else self.data

        if cycle and cycle != (None,):

            cycle = misc.to_set(cycle)
            df = df[df.cycle_nr.isin(cycle)]

        return df


    @property
    def data(self):
        """
        The sample data frame, if available, otherwise the plug data frame.
        """

        return self.sample_df if hasattr(self, 'sample_df') else self.plug_df


    @property
    def start(self):
        """
        Beginning of the first plug (in seconds).
        """

        return self.plug_df.start_time.min()


    @property
    def end(self):
        """
        End of the last plug (in seconds).
        """

        return self.plug_df.end_time.max()


    @staticmethod
    def _label(var: str, unit: bool = True) -> str:
        """
        Axis label from variable names of the samples data frame.
        """

        label = (
            misc.label(var).
            replace(' peak median', '').
            replace('z score', '[z-score]').
            replace('media norm', '(baseline corr.)')
        )

        if not label.endswith(']'):

            label = f'{label} [AU]'

        if not unit:

            label = re.sub(' \[.*\]', '', label)

        return label
