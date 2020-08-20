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

    logging.shutdown()

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%d.%m.%y %H:%M:%S")

    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(config.result_dir.joinpath("plugy_run.log"), mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)


def to_int(value):

    return int(float(value))


def to_set(value):
    """
    Makes sure the object `value` is a set, if it is a list converts
    it to set, otherwise it creates a single element set out of it.
    If `value` is None returns empty set.
    """

    if isinstance(value, set):

        return value

    elif value is None:

        return set()

    elif not isinstance(value, basestring) and hasattr(value, '__iter__'):

        return set(value)

    else:

        return {value}


def matplotlib_331_fix():

    def get_extents(self, transform = None, **kwargs):

        Bbox = plt.matplotlib.transforms.Bbox
        Path = plt.matplotlib.path.Path

        if transform is not None:
            self = transform.transform_path(self)
        if self.codes is None:
            xys = self.vertices
        elif len(np.intersect1d(self.codes, [Path.CURVE3, Path.CURVE4])) == 0:
            xys = self.vertices[self.codes != Path.CLOSEPOLY]
        else:
            xys  =  []
            for curve, code in self.iter_bezier(**kwargs):
                # places where the derivative is zero can be extrema
                _, dzeros  =  curve.axis_aligned_extrema()
                # as can the ends of the curve
                xys.append(curve([0, *dzeros, 1]))
            xys  =  np.concatenate(xys)
        if len(xys):
            return Bbox([xys.min(axis = 0), xys.max(axis = 0)])
        else:
            return Bbox.null()


    if plt.matplotlib.__version__ == '3.3.1':

        plt.matplotlib.path.Path.get_extents = get_extents


matplotlib_331_fix()