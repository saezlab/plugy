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

import logging
import tempfile
import unittest

import pathlib as pl

from ..data.bd import ChannelMap, PlugSequence, Sample

logging.basicConfig(level = logging.DEBUG,
                    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt = '%d.%m.%y %H:%M:%S')


class TestChannelMapping(unittest.TestCase):
    def setUp(self) -> None:
        """
        Prepare dictionary with channel mapping and produce string of the dict to write to file
        """
        self.test_mapping = {9: "CELLS", 10: "SUBSTRATE", 11: "FS", 12: "FS"}
        for idx, i in enumerate(range(13, 23)):
            self.test_mapping[i] = f"Drug {idx + 1}"
        self.test_mapping[23] = "BCL"
        self.test_mapping[24] = "BCH"

        self.test_file_content = "\n".join([f"{k}:{v}" for k, v in self.test_mapping.items()])

    def test_read_input_file_mapping(self):
        """
        Test if a file with the correct contents is read properly
        """
        with tempfile.NamedTemporaryFile(mode = "w+t", suffix = ".txt") as self.channel_file:
            self.channel_file.write(self.test_file_content)
            self.channel_file.seek(0)
            self.channel_file_path = pl.Path(self.channel_file.name)

            mapping = ChannelMap(input_file = self.channel_file_path)

        self.assertEqual(self.test_mapping, mapping.map)

    def test_read_input_file_bc(self):
        """
        Test if a file with the correct contents is read properly
        """
        with tempfile.NamedTemporaryFile(mode = "w+t", suffix = ".txt") as self.channel_file:
            self.channel_file.write(self.test_file_content)
            self.channel_file.seek(0)
            self.channel_file_path = pl.Path(self.channel_file.name)

            mapping = ChannelMap(input_file = self.channel_file_path)

        self.assertEqual([23, 24], mapping.bc)

    def test_read_input_file_drugs(self):
        """
        Test if a file with the correct contents is read properly
        """
        with tempfile.NamedTemporaryFile(mode = "w+t", suffix = ".txt") as self.channel_file:
            self.channel_file.write(self.test_file_content)
            self.channel_file.seek(0)
            self.channel_file_path = pl.Path(self.channel_file.name)

            mapping = ChannelMap(input_file = self.channel_file_path)

        self.assertEqual([i for i in range(13, 23)], mapping.drugs)

    def test_read_input_file_cells(self):
        """
        Test if a file with the correct contents is read properly
        """
        with tempfile.NamedTemporaryFile(mode = "w+t", suffix = ".txt") as self.channel_file:
            self.channel_file.write(self.test_file_content)
            self.channel_file.seek(0)
            self.channel_file_path = pl.Path(self.channel_file.name)

            mapping = ChannelMap(input_file = self.channel_file_path)

        self.assertEqual([9], mapping.cells)

    def test_read_input_file_substrate(self):
        """
        Test if a file with the correct contents is read properly
        """
        with tempfile.NamedTemporaryFile(mode = "w+t", suffix = ".txt") as self.channel_file:
            self.channel_file.write(self.test_file_content)
            self.channel_file.seek(0)
            self.channel_file_path = pl.Path(self.channel_file.name)

            mapping = ChannelMap(input_file = self.channel_file_path)

        self.assertEqual([10], mapping.substrate)

    def test_read_input_file_media(self):
        """
        Test if a file with the correct contents is read properly
        """
        with tempfile.NamedTemporaryFile(mode = "w+t", suffix = ".txt") as self.channel_file:
            self.channel_file.write(self.test_file_content)
            self.channel_file.seek(0)
            self.channel_file_path = pl.Path(self.channel_file.name)

            mapping = ChannelMap(input_file = self.channel_file_path)

        self.assertEqual([11, 12], mapping.media)

    def test_read_broken_input_file_channel(self):
        """
        Tests if a channel not on the braille chip is raising a ValueError
        """
        with tempfile.NamedTemporaryFile(mode = "w+t", suffix = ".txt") as self.channel_file:
            self.channel_file.write("1:Test")
            self.channel_file.seek(0)
            self.channel_file_path = pl.Path(self.channel_file.name)

            with self.assertRaises(ValueError) as cm:
                mapping = ChannelMap(self.channel_file_path)

        self.assertEqual(cm.exception.args[0], f"Channel out of BD range (9-24) you specified channel 1")

    def test_read_broken_input_file_channel_nan(self):
        """
        Tests if there is an ValueError raised when channel is something else than an int
        """
        with tempfile.NamedTemporaryFile(mode = "w+t", suffix = ".txt") as self.channel_nan_file:
            self.channel_nan_file.write("ABC:Test")
            self.channel_nan_file.seek(0)
            self.channel_nan_file_path = pl.Path(self.channel_nan_file.name)

            with self.assertRaises(ValueError) as cm:
                mapping = ChannelMap(self.channel_nan_file_path)

        self.assertEqual(cm.exception.args[0], "Channel has to be an int you specified ABC")


class TestPlugSequenceReadWrite(unittest.TestCase):
    def setUp(self) -> None:
        """
        Creates the test_file_content and a corresponding test_sequence that contains the true data
        """
        self.test_file_content = "\n1,12,Valve9,12,14,16,9\n1,12,Valve10,12,14,16,10\n1,12,Valve11,12,14,16,11\n"

        self.test_sequence = (Sample(1, 12, "Valve9", [12, 14, 16, 9]),
                              Sample(1, 12, "Valve10", [12, 14, 16, 10]),
                              Sample(1, 12, "Valve11", [12, 14, 16, 11]))

    def test_from_csv_file(self):
        """
        Test if a csv file with the correct contents is read properly
        """
        with tempfile.NamedTemporaryFile(mode = "w+t", suffix = ".csv") as self.sequence_file:
            self.sequence_file.write(self.test_file_content)
            self.sequence_file.seek(0)
            self.sequence_file_path = pl.Path(self.sequence_file.name)

            plug_sequence = PlugSequence.from_csv_file(input_file = self.sequence_file_path)

        self.assertEqual(self.test_sequence, plug_sequence.sequence)

    def test_save_csv_file(self):
        """
        Tests writing sequence to a csv file
        """
        plug_sequence = PlugSequence(self.test_sequence)
        with tempfile.NamedTemporaryFile(mode = "w+t", suffix = ".csv") as self.save_file:
            plug_sequence.save_csv(pl.Path(self.save_file.name))
            self.save_file.seek(0)
            self.assertEqual(self.test_file_content, self.save_file.read())


class TestPlugSequenceGenerate(unittest.TestCase):
    def setUp(self) -> None:
        self.test_gen_map_content = "9:CELLS\n10:SUBSTRATE\n11:FS\n12:FS\n13:Drug 1\n14:Drug 2\n15:Drug 3\n23:BCL\n24:BCH"
        self.test_sequence = (Sample(1, 15, "Cycle Barcode", [11, 12, 23, 24]),
                              Sample(1, 12, "Cell Control", [9, 10, 11, 12]),
                              Sample(1, 10, "Barcode", [11, 12, 23, 24]),
                              Sample(1, 12, "Drug 1", [9, 10, 11, 13]),
                              Sample(1, 10, "Barcode", [11, 12, 23, 24]),
                              Sample(1, 12, "Drug 2", [9, 10, 11, 14]),
                              Sample(1, 10, "Barcode", [11, 12, 23, 24]),
                              Sample(1, 12, "Drug 3", [9, 10, 11, 15]),
                              Sample(1, 10, "Barcode", [11, 12, 23, 24]),
                              Sample(1, 12, "Cell Control", [9, 10, 11, 12]),
                              Sample(1, 10, "Barcode", [11, 12, 23, 24]),
                              Sample(1, 12, "Drug 1 + Drug 2", [9, 10, 13, 14]),
                              Sample(1, 10, "Barcode", [11, 12, 23, 24]),
                              Sample(1, 12, "Drug 1 + Drug 3", [9, 10, 13, 15]),
                              Sample(1, 10, "Barcode", [11, 12, 23, 24]),
                              Sample(1, 12, "Drug 2 + Drug 3", [9, 10, 14, 15]),
                              Sample(1, 10, "Barcode", [11, 12, 23, 24]),
                              Sample(1, 12, "Cell Control", [9, 10, 11, 12]),
                              Sample(1, 10, "Barcode", [11, 12, 23, 24]),
                              Sample(1, 15, "Cycle Barcode", [11, 12, 23, 24]))

    def test_from_channel_map(self):
        with tempfile.NamedTemporaryFile(mode = "w+t", suffix = ".txt") as self.channel_file:
            self.channel_file.write(self.test_gen_map_content)
            self.channel_file.seek(0)
            self.channel_map = ChannelMap(pl.Path(self.channel_file.name))

        plug_sequence = PlugSequence.from_channel_map(self.channel_map, n_replicates = 12, n_control = 12, n_barcode = 10, n_cycle_bc = 15, open_duration = 1, generate_barcodes = True)
        self.assertEqual(self.test_sequence, plug_sequence.sequence)


class TestPlugSequenceCheck(unittest.TestCase):
    """
    Tests sanity checking of the PlugSequence input
    """
    def test_sequence_type(self):
        with self.assertRaises(TypeError) as cm:
            PlugSequence((Sample(1, 12, "Test", [11, 12, 13, 14]), "test"))
        self.assertEqual(cm.exception.args[0], "Samples in the plug sequence have to be of class Sample, you specified <class 'str'> in sample 1")

    def test_sequence_many_valves(self):
        with self.assertRaises(ValueError) as cm:
            PlugSequence((Sample(1, 12, "Test", [11, 12, 13, 14]), Sample(1, 12, "Test", [11, 12, 13, 14, 15])))
        self.assertEqual(
            cm.exception.args[0],
            "Sample 1 found with more than 4 valves open (5), "
            "THIS MIGHT DAMAGE THE CHIP!"
        )

    def test_sequence_few_valves(self):
        with self.assertWarns(UserWarning) as cm:
            PlugSequence((Sample(1, 12, "Test", [11, 12, 13, 14]), Sample(1, 12, "Test", [11, 12, 13])))
        self.assertEqual(cm.warning.args[0], "Less than 4 valves open (3) in sample 1")

    def test_sequence_other_valves(self):
        with self.assertRaises(ValueError) as cm:
            PlugSequence((Sample(1, 12, "Test", [11, 12, 13, 14]), Sample(1, 12, "Test", [1, -12, 13, 14])))
        self.assertEqual(cm.exception.args[0], "Sample 1 contains valves that are not used on the chip ([1, -12, 13, 14])")


if __name__ == '__main__':
    unittest.main()
