#!/usr/bin/env python

import os
import logging
import pathlib as pl

import lib.plugy.plugy.exp as exp
import lib.plugy.plugy.data.config as config

logging.basicConfig(
    level = logging.DEBUG,
    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt = '%d.%m.%y %H:%M:%S',
    filename = 'log',
)

datadir = os.path.join('experiments', '121219', 'data')
pmt_file = '20191213BxPC3drugs_1.txt'
seq_file = 'sequence_121219.csv'
channel_file = 'channel_map_121219.csv'

plugy_config = config.PlugyConfig(
    pmt_file = pl.Path(os.path.join(datadir, pmt_file)),
    seq_file = pl.Path(os.path.join(datadir, seq_file)),
    channel_file = pl.Path(os.path.join(datadir, channel_file)),
    auto_detect_cycles = True,
    peak_max_width = 2.5,
    figure_export_file_type = 'png',
)

plug_exp = exp.PlugExperiment(plugy_config)

plug_exp.sample_statistics
