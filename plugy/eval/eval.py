"""
Author      Nicolas Peschke
Date        20.05.2019
"""

import pathlib as pl
import numpy as np

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
    channels: dict = field(default_factory=lambda: {"green": "#5D9731", "blue": "#3A73BA", "orange": "#F68026"})
    colors: dict = field(default_factory=lambda: {"green": "#5D9731", "blue": "#3A73BA", "orange": "#F68026"})
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

        if self.ignore_orange_channel:
            self.set_channel_values(ignore_orange=True)

        else:
            self.set_channel_values()

    def set_channel_values(self, correct_time: bool = True, ignore_green: bool = False, ignore_orange: bool = False, ignore_uv: bool = False):
        """
        Sets & corrects values in the multichannel acquisition data.
        :param correct_time: If the time should be corrected from having 100 measurements at a single
                             timepoint to evenly spaced measurements using the acquisition rate.
        :param ignore_green: If all values of the green channel should be set to 0
        :param ignore_orange: If all values of the orange channel should be set to 0
        :param ignore_uv: If all values of the uv channel should be set to 0
        """
        time_between_samplings = 1 / self.acquisition_rate

        # Iterate through self.data and overwrite if iteration successful
        with np.nditer(self.data, op_flags=["readwrite"]) as data:
            for idx, value in enumerate(data):
                # Get the column index
                col = idx % 4

                # Change column values depending on parameters
                if correct_time and (col == 0):
                    value[...] = (idx / 4) * time_between_samplings

                if ignore_green and (col == 1):
                    value[...] = 0

                if ignore_orange and (col == 2):
                    value[...] = 0

                if ignore_uv and (col == 3):
                    value[...] = 0
