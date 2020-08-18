#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Author      Nicolas Peschke
# Date        19.09.2019
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

import collections as coll
import csv
import itertools
import logging
import pathlib as pl
import warnings
from dataclasses import dataclass

module_logger = logging.getLogger("plugy.data.bd")

Sample = coll.namedtuple("Sample", ["open_duration", "n_replicates", "name", "open_valves"])


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

    def get_compounds(self, open_valves):
        compounds = list()

        for open_valve in open_valves:
            if (open_valve in self.drugs) or (open_valve in self.media):
                compounds.append(self.map[open_valve])

            # if open_valves in self.cells:
            #     cells = True
            # if open_valves in self.substrate:
            #     substrate = True
            # if open_valves in self.media:
            #     media = True
            # if open_valves in self.bc:
            #     bc = True
            #
            #     drugs = True

        return compounds

    def get_compound_list(self):
        """
        Retrieves a list of the compounds used in the experiment including the media
        :return: list containing the names of the compounds
        """
        compounds = list()

        compounds.extend([self.map[valve] for valve in self.media])
        compounds.extend([self.map[valve] for valve in self.drugs])

        return compounds


class PlugSequence(object):
    @classmethod
    def from_csv_file(cls, input_file: pl.Path):
        """
        Reads a "Samples on Demand v5" compatible csv file and creates a PlugSequence object from it.
        :param input_file: File path to read from
        :return: PlugSequence object
        """
        module_logger.info(f"Reading PlugSequence from {input_file.absolute()}")
        sequence = list()
        with input_file.open("r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 0:
                    continue
                else:
                    # noinspection PyCallByClass
                    sequence.append(Sample(open_duration=float(row[0]), n_replicates=int(row[1]), name=row[2], open_valves=[int(i) for i in row[3:]]))

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
        module_logger.info(f"Creating PlugSequence from ChannelMap")
        module_logger.debug("Configuration:")
        for k, v in locals().items():
            module_logger.debug(f"{k}: {v}")
        samples = list()

        # Generate templates
        control = Sample(open_duration=open_duration, n_replicates=n_control,
                         name="Cell Control", open_valves=channel_map.cells + channel_map.substrate + channel_map.media)
        barcode = Sample(open_duration=open_duration, n_replicates=n_barcode,
                         name="Barcode", open_valves=channel_map.media + channel_map.bc)

        cycle_bc = Sample(open_duration=open_duration, n_replicates=n_cycle_bc,
                          name="Cycle Barcode", open_valves=channel_map.media + channel_map.bc)

        if len(channel_map.bc) == 1:
            barcode_substitute = channel_map.drugs[-1]
            module_logger.warning(f"Single barcode channel detected, substituting last drug valve ({barcode_substitute}) for missing barcode")
            barcode.open_valves.append(barcode_substitute)
            cycle_bc.open_valves.append(barcode_substitute)

        individual_drugs = list()
        for drug in channel_map.drugs:
            individual_drugs.append(Sample(open_duration=open_duration, n_replicates=n_replicates,
                                           name=channel_map.map[drug],
                                           open_valves=channel_map.cells + channel_map.substrate + [channel_map.media[0]] + [drug]))
            if generate_barcodes:
                individual_drugs.append(barcode)

        # Generate sample list
        samples.append(cycle_bc)
        samples.append(control)
        if generate_barcodes:
            samples.append(barcode)
        samples = samples + individual_drugs

        for idx, combination in enumerate(itertools.combinations(channel_map.drugs, 2)):
            if idx % 10 == 0:
                samples.append(control)
                if generate_barcodes:
                    samples.append(barcode)

            samples.append(Sample(open_duration=open_duration, n_replicates=n_replicates,
                                  name=f"{channel_map.map[combination[0]]} + {channel_map.map[combination[1]]}",
                                  open_valves=channel_map.cells + channel_map.substrate + list(combination)))
            if generate_barcodes:
                samples.append(barcode)

        samples.append(control)
        if generate_barcodes:
            samples.append(barcode)

        samples.append(cycle_bc)
        return cls(sequence=tuple(samples), channel_map=channel_map)

    def __init__(self, sequence: tuple, **kwargs):
        """
        Handles the plug sequence of the braille display
        """
        self.sequence = sequence
        self.check_sequence()

        try:
            if isinstance(kwargs["channel_map"], ChannelMap):
                self.channel_map = kwargs["channel_map"]
        except KeyError:
            pass

    def check_sequence(self):
        """
        Tests sanity of sequence
        """
        module_logger.debug("Checking plug sequence")

        for idx, sample in enumerate(self.sequence):
            if not isinstance(sample, Sample):
                raise TypeError(f"Samples in the plug sequence have to be of class Sample, you specified {type(sample)} in sample {idx}")

            if len(sample.open_valves) < 4:
                warnings.warn(f"Less than 4 valves open ({len(sample.open_valves)}) in sample {idx}")

            elif len(sample.open_valves) > 4:
                raise ValueError(f"Sample {idx} found with more than 4 valves open ({len(sample.open_valves)}), THIS WILL DESTROY THE CHIP!")

            for valve in sample.open_valves:
                if valve not in range(9, 25):
                    raise ValueError(f"Sample {idx} contains valves that are not used on the chip ({sample.open_valves})")

    def save_csv(self, path: pl.Path):
        """
        Saves the PlugSequence as a csv file that is compatible to "Samples on Demand v5"
        :param path: Path to write to
        """
        module_logger.info(f"Writing sequence to file {path.absolute()}")
        with path.open("w", newline="\r\n") as f:
            f.write("\n")
            for sample in self.sequence:
                f.write(f"{str(sample.open_duration)},{str(sample.n_replicates)},{str(sample.name)},{','.join([str(i) for i in sample.open_valves])}")
                f.write("\n")

    def get_samples(self, **kwargs):
        """
        Filters barcodes out of the plug sequence and returns a PlugSequence object without barcodes
        :param kwargs: channel_map: optionally overrides the ChannelMap object that might be already present in the PlugSequence object
        :return: tuple containing Samples filtered
        """

        if "channel_map" in kwargs.keys():
            if isinstance(kwargs["channel_map"], ChannelMap):
                c_map = kwargs["channel_map"]
            else:
                raise TypeError(f"channel_map of type {type(kwargs['channel_map'])} but has to be type ChannelMap")
        else:
            try:
                c_map = self.channel_map
            except AttributeError:
                raise AttributeError("No ChannelMap object specified or found to get sample information from! You can specify one with the channel_map keyword argument.")

        filtered_samples = list()

        for sample in self.sequence:
            bc = False
            # if c_map.cells in sample.open_valves:
            #     cells.append(True)
            # else:
            #     cells.append(False)
            #
            # if c_map.substrate in sample.open_valves:
            #     substrate.append(True)
            # else:
            #     substrate.append(False)

            # for valve in sample.open_valves:
            #     if valve in c_map.drugs:
            #         filtered_samples.append(sample)
            #         continue

            for bc_valve in c_map.bc:
                if bc_valve in sample.open_valves:
                    bc = True
                    break

            if not bc:
                filtered_samples.append(sample)

        return PlugSequence(tuple(filtered_samples))
