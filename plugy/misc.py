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
import time
import inspect
import subprocess as sp

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def add_git_hash_caption(fig: plt.Figure, offset: float = 0.8):
    """
    Adds a caption with the current git hash in a human readable form based
    on the most recent tag.
    If the repository contains changes that are not committed yet,
    the keyword "-dirty" is appended to the caption.

    :param fig: matplotlib.pyplot.Figure to add the caption to.
    :param offset: Determines how many millimeter the caption is offset
        from the bottom left corner of the figure.

    :return: None
    """

    sha = sp.run(
        ['git', 'describe', '--tags', '--long', '--dirty'],
        capture_output = True,
        text = True,
        cwd = os.path.dirname(os.path.realpath(__file__)),
    ).stdout.strip()
    rel_position = offset / (fig.get_size_inches() * 25.4)
    fig.text(
        rel_position[0],
        rel_position[1],
        f'Created on {time.ctime()} with plugy {sha}',
        fontsize = 'xx-small',
        fontweight = 'light',
    )


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
    axes.plot(x_values, y_values, '--', c = '#D22027')


def to_int(value):

    return int(float(value))


def to_tuple(value):

    return (
        value
            if isinstance(value, tuple) else
        ()
            if value is None else
        (value,)
    )


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

    elif not isinstance(value, str) and hasattr(value, '__iter__'):

        return set(value)

    else:

        return {value}


def first(value):
    """
    Returns first element of an iterator or the value unchanged if it's not
    an iterator.
    """

    if hasattr(value, '__iter__'):

        try:

            value = next(value.__iter__())

        except StopIteration:

            value = None

    return value


def prettyfloat(n):

    return '%.02g' % n if isinstance(n, float) else str(n)


def dict_str(dct):

    if not isinstance(dct, dict):

        return str(dct)

    return ', '.join(
        '%s=%s' % (str(key), prettyfloat(val))
        for key, val in dct.items()
    )


def ntuple_str(nt, pretty_floats = True):

    return (
        ', '.join(
            '%s=%s' % (field, prettyfloat(value))
            for field, value in zip(nt._fields, nt)
        )
            if pretty_floats else
        nt.__repr__().split('(')[1][:-1]
    )


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


def seaborn_violin_fix(
        box_color = None,
        midpoint_color = None,
        violin_border_width = None,
        box_linewidth = 1.5,
        **kwargs,
    ):
    """
    Draws miniature boxplots in the middle of the violins on a violinplot.
    Seaborn has hardcoded values for the colors of these boxplots, hence
    we need to provide a modified method to have any control over the
    aesthetics.
    """

    def add_boxes(ax, data, support, density, center, plotter, parent_env):

        def swap(*args):

            return reversed(args) if plotter.orient == 'v' else args


        q25, q50, q75 = np.percentile(data, [25, 50, 75])
        whisker_lim = 1.5 * (q75 - q25)
        h1 = np.min(data[data >= (q25 - whisker_lim)])
        h2 = np.max(data[data <= (q75 + whisker_lim)])

        box_color = parent_env['box_color'] or plotter.gray
        midpoint_color = parent_env['midpoint_color'] or 'white'

        # Draw a boxplot using lines and a point
        ax.plot(
            *swap([h1, h2], [center, center]),
            linewidth = parent_env['box_linewidth'],
            color = box_color,
        )
        ax.plot(
            *swap([q25, q75], [center, center]),
            linewidth = parent_env['box_linewidth'] * 3,
            color = box_color,
        )
        ax.scatter(
            *swap(q50, center),
            zorder = 3,
            color = midpoint_color,
            edgecolor = box_color,
            s = np.square(parent_env['box_linewidth'] * 2),
        )


    valid_args = {
        'x', 'y', 'hue', 'data', 'order', 'hue_order',
        'bw', 'cut', 'scale', 'scale_hue', 'gridsize',
        'width', 'inner', 'split', 'dodge', 'orient', 'linewidth',
        'color', 'palette', 'saturation',
    }

    ax = kwargs.pop('ax', None)
    violin_args = {
        k: v.default
        for k, v in inspect.signature(sns.violinplot).parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    violin_args.update(kwargs)
    violin_args = dict(
        it for it in violin_args.items()
        if it[0] in valid_args
    )

    plotter = sns.categorical._ViolinPlotter(**violin_args)

    if ax is None:

        ax = plt.gca()

    plotter.plot(ax)

    for i, group_data in enumerate(plotter.plot_data):

        if plotter.plot_hues is None:

            support, density = plotter.support[i], plotter.density[i]
            violin_data = sns.utils.remove_na(group_data)
            add_boxes(ax, violin_data, support, density, i, plotter, locals())

        else:

            for j, hue_level in enumerate(plotter.hue_names):

                support, density = (
                    plotter.support[i][j],
                    plotter.density[i][j],
                )
                if support.size <= 1:

                    continue

                hue_mask = plotter.plot_hues[i] == hue_level
                violin_data = sns.utils.remove_na(group_data[hue_mask])

                add_boxes(
                    ax, violin_data, support, density,
                    i + (offsets[j] if plotter.split else 0), # noqa: F821
                    locals(),
                )

    # this is an example how to remove the violin borders
    # unfortunately seaborn doesn't provide more convenient API for that
    if violin_border_width is not None:

        _ = [
            ax.collections[i].set_linewidth(violin_border_width)
            for i in range(len(ax.collections))
        ]

    return ax


matplotlib_331_fix()