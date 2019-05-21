"""
Author      Nicolas Peschke
Date        20.05.2019
"""

import pathlib as pl
import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from ..main import Plugy


@dataclass
class EvalPlugy(Plugy):
    infile: pl.Path
    results_dir: pl.Path = pl.Path.cwd().joinpath("results")
    cut: tuple = (None, None)
    drugs: list = field(default_factory=list)
    signal_threshold: float = .02
    adaptive_signal_threshold: bool = True
    peak_minwidth: float = 5
    channels: dict = field(default_factory=lambda: {"green": "#5D9731", "blue":   "#3A73BA", "orange": "#F68026"})
    colors: dict = field(default_factory=lambda: {"green": "#5D9731", "blue":   "#3A73BA", "orange": "#F68026"})
    discard: tuple = (2, 1)
    x_ticks_density: float = 5
    gaussian_smoothing_sigma: float = 33
    adaptive_threshold_blocksize: float = 111
    adaptive_threshold_method: str = "gaussian"
    adaptive_threshold_sigma: float = 190
    adaptive_threshold_offset: float = 0.01
    merge_close_peaks: float = 50
    drug_sep: str = "&"
    direct_drug_combinations: bool = False
    barcode_intensity_correction: float = 1.0
    ignore_orange_channel: bool = False
    acquisition_rate: int = 300
    figure_file_type: str = ".svg"

    def __post_init__(self):
        """
        Upon initialization of the EvalPlugy object, the infile is read and preprocessed
        """
        self.data = None
        self.read()

        self.correct_acquisition_time()

        if self.ignore_orange_channel:
            self.set_channel_values("orange", 0)

    def correct_acquisition_time(self):
        """
        Corrects reported acquisition times by multiplying the index of the measurement by 1/acquisition rate
        :return: None
        """
        time_between_samplings = 1/self.acquisition_rate
        # idx = 0
        # for line in np.nditer(self.data, op_flags=["readwrite"]):
        #     print(line)
        #     line[0] = idx * time_between_samplings
        #     idx += 1
        with np.nditer(self.data, op_flags=["readwrite"]) as data:
            for idx, value in enumerate(data):
                if idx % 4 is 0:
                    value[...] = (idx / 4) * time_between_samplings

        print(self.data)

    def set_channel_values(self, channel: str, value: float = 0):
        """
        Sets each measurement for the specified channel to the specified value
        :param channel: The channel to modify: "green", "blue" or "orange"
        :param value: All measurements of this channel are set to this value
        :return: None
        """
        self.data[channel] = value
