#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Author      Nicolas Peschke
# Date        04.04.2019
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
import sys
import time

import subprocess as sp
import matplotlib.pyplot as plt
import numpy as np

from .data.config import PlugyConfig


def add_git_hash_caption(fig: plt.Figure, offset: float = 0.8):
    """
    Adds a caption with the current git hash in a human readable form based on the most recent tag.
    If the repository contains changes that are not committed yet, the keyword "-dirty" is appended to the caption.
    :param fig: matplotlib.pyplot.Figure to add the caption to.
    :param offset: Determines how many millimeter the caption is offset from the bottom left corner of the figure.
    :return: None
    """
    sha = sp.run(["git", "describe", "--tags", "--long", "--dirty"], capture_output = True, text = True).stdout.strip()
    rel_position = offset / (fig.get_size_inches() * 25.4)
    fig.text(rel_position[0], rel_position[1], f"Created on {time.ctime()} with braille-kidney {sha}", fontsize = "x-small", fontweight = "light")


def plot_line(slope: float, intercept: float, axes: plt.Axes):
    """
    Plots a line using slope and intercept

    :param slope: Slope of the line
    :param intercept: Y-Axis intercept of the line
    :param axes: plt.Axes object to plot on
    :return: None
    """
    x_values = np.array(axes.get_xlim())
    y_values = intercept + slope * x_values
    axes.plot(x_values, y_values, '--')


def start_logging(config: PlugyConfig):
    """
    Starts logging to STDOUT and to a log file in the result directory
    :param config: PlugyConfig object to retrieve the result_dir from
    :return: None
    """
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%d.%m.%y %H:%M:%S")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(config.result_dir.joinpath("plugy_run.log"), mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

