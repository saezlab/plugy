"""
Author      Nicolas Peschke
Date        04.04.2019
"""

import time

import subprocess as sp
import matplotlib.pyplot as plt


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
