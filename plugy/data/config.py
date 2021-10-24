#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# This file is part of the `plugy` python module
#
# Copyright
# 2018-2021
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


import os
import sys
import re
import time
import pathlib as pl
import typing
import logging
import hashlib

from dataclasses import dataclass, field


module_logger = logging.getLogger(__name__)


@dataclass
class PlugyConfig(object):

    # File Paths
    input_dir: str = '.'
    pmt_file: typing.Union[pl.Path, str, re.Pattern] = \
        re.compile(
            r'(?:fluor|exp)(?:[-\w ]*)?'
            r'(?:\.txt)(?:\.gz)?',
            re.IGNORECASE
        )
    seq_file: typing.Union[pl.Path, str, re.Pattern] = \
        re.compile(r'seq(?:[-\w ]*)?(?:\.[tc]sv)?', re.IGNORECASE)
    channel_file: typing.Union[pl.Path, str, re.Pattern] = \
        re.compile(r'channel(?:[-\w ]*)?(?:\.[tc]sv)?', re.IGNORECASE)
    result_base_dir: typing.Union[pl.Path, str] = None
    result_subdirs: bool = False
    timestamp_result_subdirs: bool = False

    ## General config
    name: str = None
    # file type for saving the figures
    figure_export_file_type: str = 'png'
    # colors for representing channels on the figures
    colors: dict = field(
        default_factory = lambda: {
            'green': '#5D9731',
            'uv': '#3A73BA',
            'orange': '#F68026',
        }
    )
    # run the entire workflow
    run: bool = False
    # run the initial steps: set up the objects and load the data
    init: bool = None
    # run the plug detection
    plugs: bool = None
    # run the sample detection
    samples: bool = None
    # run the quality control
    qc: bool = None
    # run the analysis
    analysis: bool = None
    # label of the control channels in the channel layout
    control_label: typing.Union[set, str] = 'FS'
    medium_control_label: typing.Union[set, str] = 'FS'
    negative_control_label: typing.Union[set, str] = field(
        default_factory = lambda: {'neg-ctrl', 'neg_ctrl'}
    )
    positive_control_label: typing.Union[set, str] = field(
        default_factory = lambda: {'pos-ctrl', 'pos_ctrl'}
    )
    # do not raise an error if the quality control fails
    # but proceed with quantification and visualization
    # of the results
    ignore_qc_result: bool = True
    # whether to write log messages also to the STDOUT
    # or only to the log file
    log_to_stdout: bool = True

    # sequence settings
    allow_lt4_valves: bool = False

    ## PMT configuration
    # function, name and column index of the channels
    channels: dict = field(
        default_factory = lambda: {
            'barcode': ('uv', 3),
            'control': ('orange', 2),
            'readout': ('green', 1),
        }
    )
    # PMT acquisition rate in Hz
    acquisition_rate: int = 300
    # crop the data by removing the segments before the first
    # and after the second element of this tuple (in seconds)
    cut: tuple = (None, None)

    # set the acquisition times by a linear interpolation
    # based on the `acquisition_rate`
    correct_acquisition_time: bool = True
    # set the value of the channels listed here to zero
    ignore_channels: set = field(default_factory = set)
    # scale certain channels by a constant factor
    fake_gains: dict = field(default_factory = dict)
    # a default scaling factor for the remaining channels
    fake_gain_default: float = 1.0
    # I don't know what this intended to be,
    # raises NotImplementedError
    fake_gain_adaptive: bool = False
    # above this value set the value of the barcode channel to 1.0
    barcode_raw_threshold: float = None

    # parameters for the peak detection algorithm;
    # we use a generic signal processing algorithm
    # from scipy.signal.find_peaks
    auto_detect_cycles: bool = True
    peak_min_threshold: float = 0.05
    peak_max_threshold: float = 2.0
    peak_min_distance: float = 0.03
    peak_min_prominence: float = 0
    peak_max_prominence: float = 10
    peak_min_width: float = 0.5
    peak_max_width: float = 2.5
    width_rel_height: float = 0.5
    # max distance between centers of the peaks to merge
    # if we merge the peaks by their centers
    merge_peaks_distance: float = 0.2
    # alternative is to merge them by 'center', but unless for
    # some weird reason supposedly distinct peaks overlap
    # merging by overlap is a superior method
    merge_peaks_by: str = 'overlap'
    n_bc_adjacent_discards: int = 1
    # lowest number of barcode plugs separating two cycles
    # barcode segments with fewer plugs separate samples within cycles
    min_end_cycle_barcodes: int = 12
    # lowest number of barcode plugs separating 2 samples
    # if fewer barcode plugs between sample plugs, the sample
    # counter won't be increased
    min_between_samples_barcodes: int = 2
    # lowest number of plugs in a sample
    # if barcode comes without reaching yet this threshold,
    # the sample counter won't be increased
    min_plugs_in_sample: int = 3

    # if False, the workflow stops after plug identification and
    # quantification, sample and barcode plugs won't be distinguished
    has_barcode: bool = True
    # if False, plug detection stops after identification of barcode plugs
    # and no sample and cycle numbers will be assigned to the plugs
    has_samples_cycles: bool = True
    # the number of samples within one cycle; no need to provide if sequence
    # file is available; otherwise it helps in evaluating barcode detection
    # methods even if the sequence is unknoen
    samples_per_cycle: int = None
    # The experiment has positive (and optionally negative) control samples.
    # If False, the tasks which require those samples will be skipped.
    has_controls: bool = True
    # the name of a method for identifying barcode plugs:
    # * `simple`: the simplest method, works with not too noisy data:
    #    plugs are barcode if the channel of the blue value is the highest
    # * `adaptive`: change the value of the `blue_highest_times`
    #    parameter to find the one giving the best result
    barcoding_method: str = 'simple'
    # override parameters for the selected barcode method
    barcoding_param: dict = field(default_factory = dict)
    # default parameters for the barcode methods
    barcoding_param_defaults: dict = field(
        default_factory = lambda: {
            'simple': {
                # a scaling factor for the `blue_highest` method: the blue
                # channel must be at least this times higher than the control
                # channel for barcode plugs
                'times': 1.0,
            },
            'adaptive': {
                'adaptive_method': 'simple',
                'higher_threshold_factor': 1.,
                'thresholding_method': 'local',
                'block_size': 7,
            }
        }
    )

    append: list = field(default_factory = list)

    # Analysis
    normalize_using_control: bool = True
    normalize_using_media_control_lin_reg: bool = True
    readout_column: str = "readout_peak_median"
    readout_analysis_column: str = "readout_media_norm_z_score"

    # Plotting config
    seaborn_context: str = "notebook"
    seaborn_context_dict: dict = field(
        default_factory = lambda: {
            'grid.linewidth': .9,
        }
    )
    seaborn_style: str = "whitegrid"
    seaborn_style_dict: dict = field(
        default_factory = lambda: {
            'axes.edgecolor': '#000000',
            'grid.color': '#000000',
        }
    )
    font_scale: typing.Union[float, int] = 2
    scatter_dot_size: typing.Union[float, int] = 10
    plot_git_caption: bool = True

    heatmap_second_scale: str = 'pos_ctrl'
    heatmap_override_scale: tuple = None
    heatmap_override_second_scale: tuple = None

    # shorten drug names on certain plots
    short_labels: bool = True

    # enable/disable figure titles
    figure_titles: bool = True

    # QC
    contamination_threshold: float = 0.03

    # Statistics
    alpha: float = 0.05

    # Palette for plotting
    palette: tuple = (
        '#7264B9',
        '#5B205F',
        '#9E1639',
        '#ED0772',
        '#D22027',
    )

    def __post_init__(self):
        # Creating result dir for each individual run


        self._set_result_dir()
        self.start_logging()
        self._set_paths()
        self._set_name()


    def _set_result_dir(self):

        self.result_base_dir = (
            pl.Path(self.result_base_dir)
                if isinstance(self.result_base_dir, str) else
            self.result_base_dir
        )

        self.result_dir = (
            self.result_base_dir or
            pl.Path.cwd().joinpath('results')
        )

        if self.result_subdirs:

            current_time = time.strftime("%Y%m%d_%H%M")

            subdir = (
                f"{os.path.splitext(self.pmt_file.name)[0]}_{current_time}"
                    if self.timestamp_result_subdirs else
                self.name
            )

            self.result_dir = self.result_dir.joinpath(
                self.result_dir,
                subdir,
            )

        os.makedirs(str(self.result_dir), exist_ok = True)


    def _set_paths(self):

        for attr in (
            'pmt_file',
            'channel_file',
            'seq_file',
        ):

            path = getattr(self, attr)

            if not isinstance(path, pl.Path):

                path = self._find_file(path = path, in_dir = self.input_dir)

                if not os.path.exists(path):

                    msg = 'File not found: %s; `%s`' % (attr, path)
                    module_logger.critical(msg)
                    module_logger.critical(
                        'Files in directory `%s`: %s' % (
                            self.input_dir,
                            ', '.join(os.listdir(self.input_dir))
                        )
                    )
                    path = None

                    if attr == 'pmt_file':

                        raise FileNotFoundError(msg)

                    else:

                        setattr(self, attr, None)

                else:

                    msg = 'Input file `%s`: %s' % (attr, path)
                    module_logger.info(msg)
                    path = pl.Path(path)

                setattr(self, attr, path)


    @staticmethod
    def _find_file(path, in_dir):

        if hasattr(path, 'pattern'):

            subdir, fname = (
                ('', path.pattern)
                    if sys.platform.lower().startswith('win') else
                os.path.split(path.pattern)
            )

            in_dir = in_dir + subdir

            for f in os.listdir(in_dir):

                this_path = os.path.join(subdir, f)

                if path.match(this_path):

                    path = os.path.join(in_dir, this_path)
                    break

        return path.pattern if hasattr(path, 'pattern') else path


    def _set_name(self):

        self.identity = hashlib.md5(
            ';'.join(
                (
                    str(self.pmt_file),
                    str(self.seq_file),
                    str(self.channel_file),
                )
            ).encode('utf-8')
        ).hexdigest()[:7]

        self.name = (
            self.name or
            '%s-%s' % (
                os.path.splitext(self.pmt_file.name)[0],
                self.identity,
            )
        )


    def start_logging(self):
        """
        Starts logging to STDOUT and to a log file in the result directory
        :param config: PlugyConfig object to retrieve the result_dir from
        :return: None
        """

        logging.shutdown()

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt = "%d.%m.%y %H:%M:%S",
        )

        logger = logging.getLogger()
        logger.handlers = []
        logger.setLevel(logging.DEBUG)

        if self.log_to_stdout:

            stream_handler = logging.StreamHandler(stream=sys.stdout)
            stream_handler.setLevel(logging.INFO)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(
            self.result_dir.joinpath('plugy.log'),
            mode = 'a'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)


    @property
    def channel_names(self):

        return dict(
            (key, value[0])
            for key, value in self.channels.items()
        )


    def channel_color(self, channel):

        return (
            self.colors[self.channels[channel][0]]
                if channel in self.channels else
            '#000000'
        )
