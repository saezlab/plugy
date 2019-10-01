"""
Author      Nicolas Peschke
Date        19.09.2019

This file is part of the `plugy` python module

Copyright
2018-2019
EMBL, Heidelberg University
File author(s): Dénes Türei (turei.denes@gmail.com)
                Nicolas Peschke
Distributed under the GPLv3 License.
See accompanying file LICENSE.txt or copy at
    http://www.gnu.org/licenses/gpl-3.0.html

"""
import itertools
import logging
import csv

import pathlib as pl
import collections as coll

from dataclasses import dataclass

import warnings

module_logger = logging.getLogger("plugy.data.bd")


@dataclass
class ChannelMap(object):
    input_file: pl.Path

    def __post_init__(self):
        module_logger.info(f"Creating ChannelMap object from file {self.input_file.absolute()}")
        module_logger.debug(f"Configuration:")
        for k, v in self.__dict__.items():
            module_logger.debug(f"{k}: {v}")

        self.cells = list()
        self.substrate = list()
        self.media = list()
        self.bc = list()
        self.drugs = list()

        self.map = self.read_input_file()

    # noinspection PyAttributeOutsideInit
    def read_input_file(self):
        module_logger.info(f"Reading file")
        mapping = dict()
        with self.input_file.open("r") as f:
            for line in f:
                k, v = line.strip().split(":")

                try:
                    k = int(k)
                except ValueError:
                    raise ValueError(f"Channel has to be an int you specified {k}")

                if k not in range(9, 25):
                    raise ValueError(f"Channel out of BD range (9-24) you specified channel {k}")

                if str(v).startswith("CELLS"):
                    self.cells.append(k)
                elif str(v).startswith("SUBSTRATE"):
                    self.substrate.append(k)
                elif str(v).startswith("BC"):
                    self.bc.append(k)
                elif str(v).startswith("FS"):
                    self.media.append(k)
                else:
                    self.drugs.append(k)

                mapping[k] = v

        return mapping


class PlugSequence(object):
    Sample = coll.namedtuple("Sample", ["open_duration", "n_replicates", "name", "open_valves"])

    @classmethod
    def from_csv_file(cls, input_file: pl.Path):
        sequence = list()
        with input_file.open("r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 0:
                    continue
                else:
                    # noinspection PyCallByClass
                    sequence.append(cls.Sample(open_duration=int(row[0]), n_replicates=int(row[1]), name=row[2], open_valves=[int(i) for i in row[3:]]))

        return cls(sequence=tuple(sequence))

    # noinspection PyCallByClass
    @classmethod
    def from_channel_map(cls, channel_map: ChannelMap, n_replicates: int = 12, n_control: int = 12, n_barcode: int = 5, n_cycle_bc: int = 15, open_duration: int = 1, generate_barcodes: bool = True):
        """
        Generates a PlugSequence object from a ChannelMap
        :param channel_map: ChannelMap to generate the PlugSequence from
        :param n_replicates: Number of replicates for drug combinations
        :param n_control: Number of replicates for the cell only controls
        :param n_barcode: Number of barcodes between samples
        :param n_cycle_bc: Number of barcodes at the beginning and end of a cylcle
        :param open_duration: How long the braille valves should be opened per plug in seconds
        :param generate_barcodes: If barcodes should be encoded in the Sequence. Set to False if barcodes should be generated using "Samples on Demand v5"
        :return: PlugSequence object
        """
        samples = list()

        # Generate templates
        control = cls.Sample(open_duration=open_duration, n_replicates=n_control, name="Cell Control", open_valves=channel_map.cells + channel_map.substrate + channel_map.media)
        barcode = cls.Sample(open_duration=open_duration, n_replicates=n_barcode, name="Barcode", open_valves=channel_map.media + channel_map.bc)
        individual_drugs = list()
        for drug in channel_map.drugs:
            individual_drugs.append(cls.Sample(open_duration=open_duration, n_replicates=n_replicates, name=channel_map.map[drug], open_valves=channel_map.cells + channel_map.substrate + [channel_map.media[0]] + [drug]))
            if generate_barcodes:
                individual_drugs.append(barcode)

        # Generate sample list
        samples.append(cls.Sample(open_duration=open_duration, n_replicates=n_cycle_bc, name="Start Cycle Barcode", open_valves=channel_map.media + channel_map.bc))
        samples.append(control)
        if generate_barcodes:
            samples.append(barcode)
        samples = samples + individual_drugs

        for idx, combination in enumerate(itertools.combinations(channel_map.drugs, 2)):
            if idx % 10 == 0:
                samples.append(control)
                samples.append(barcode)

            samples.append(
                cls.Sample(open_duration=open_duration, n_replicates=n_replicates, name=f"{channel_map.map[combination[0]]} + {channel_map.map[combination[1]]}", open_valves=channel_map.cells + channel_map.substrate + list(combination)))
            if generate_barcodes:
                samples.append(barcode)

        samples.append(control)
        if generate_barcodes:
            samples.append(barcode)

        samples.append(cls.Sample(open_duration=open_duration, n_replicates=n_cycle_bc, name="End Cycle Barcode", open_valves=channel_map.media + channel_map.bc))
        return cls(sequence=tuple(samples))

    def __init__(self, sequence: tuple):
        """
        Handles the plug sequence of the braille display
        """
        self.sequence = sequence
        self.check_sequence()

    def check_sequence(self):
        """
        Tests sanity of sequence
        """
        module_logger.debug("Checking plug sequence")

        for idx, sample in enumerate(self.sequence):
            if not isinstance(sample, PlugSequence.Sample):
                raise TypeError(f"Samples in the plug sequence have to be of class PlugSequence.Sample, you specified {type(sample)} in sample {idx}")

            if len(sample.open_valves) < 4:
                warnings.warn(f"Less than 4 valves open ({len(sample.open_valves)}) in sample {idx}")

            elif len(sample.open_valves) > 4:
                raise ValueError(f"Sample {idx} found with more than 4 valves open ({len(sample.open_valves)}), THIS WILL DESTROY THE CHIP!")

            for valve in sample.open_valves:
                if valve not in range(9, 25):
                    raise ValueError(f"Sample {idx} contains valves that are not used on the chip ({sample.open_valves})")

