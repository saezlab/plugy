"""
Author      Nicolas Peschke
Date        26.09.2019

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
import tempfile
import unittest

import pathlib as pl

from ..data.bd import ChannelMap, PlugSequence


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

    def test_read_input_file(self):
        """
        Test if a file with the correct contents is read properly
        """
        with tempfile.NamedTemporaryFile(mode="w+t", suffix=".txt") as self.channel_file:
            self.channel_file.write(self.test_file_content)
            self.channel_file.seek(0)
            self.channel_file_path = pl.Path(self.channel_file.name)

            mapping = ChannelMap(input_file=self.channel_file_path)

        self.assertEqual(self.test_mapping, mapping.map)

    def test_read_broken_input_file_channel(self):
        """
        Tests if a channel not on the braille chip is raising a ValueError
        """
        with tempfile.NamedTemporaryFile(mode="w+t", suffix=".txt") as self.channel_file:
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
        with tempfile.NamedTemporaryFile(mode="w+t", suffix=".txt") as self.channel_nan_file:
            self.channel_nan_file.write("ABC:Test")
            self.channel_nan_file.seek(0)
            self.channel_nan_file_path = pl.Path(self.channel_nan_file.name)

            with self.assertRaises(ValueError) as cm:
                mapping = ChannelMap(self.channel_nan_file_path)

        self.assertEqual(cm.exception.args[0], "Channel has to be an int you specified ABC")


class TestPlugSequenceRead(unittest.TestCase):
    def setUp(self) -> None:
        """
        Creates the test_file_content and a corresponding test_sequence that contains the true data
        """
        self.test_file_content = "\n1,12,Valve9,12,14,16,9\n1,12,Valve10,12,14,16,10\n1,12,Valve11,12,14,16,11"

        self.test_sequence = (PlugSequence.Sample(1, 12, "Valve9", [12, 14, 16, 9]),
                              PlugSequence.Sample(1, 12, "Valve10", [12, 14, 16, 10]),
                              PlugSequence.Sample(1, 12, "Valve11", [12, 14, 16, 11]))

    def test_from_csv_file(self):
        """
        Test if a csv file with the correct contents is read properly
        """
        with tempfile.NamedTemporaryFile(mode="w+t", suffix=".csv") as self.sequence_file:
            self.sequence_file.write(self.test_file_content)
            self.sequence_file.seek(0)
            self.sequence_file_path = pl.Path(self.sequence_file.name)

            plug_sequence = PlugSequence.from_csv_file(input_file=self.sequence_file_path)

        self.assertEqual(plug_sequence.sequence, self.test_sequence)


if __name__ == '__main__':
    unittest.main()
