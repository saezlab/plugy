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
import collections

import pathlib as pl

from ..data.bd import ChannelMap


class TestChannelMapping(unittest.TestCase):
    def setUp(self) -> None:
        # header = "9:CELLS\n10:SUBSTRATE\n11:FS\n12:FS"
        # drugs = [f"{i}:Drug{i}" for i in range(13, 23)]
        # barcodes = "23:BC Low\n24:BC High"
        # self.test_file_content = "\n".join([header, "\n".join(drugs), barcodes])

        self.test_mapping = {9: "CELLS", 10: "SUBSTRATE", 11: "FS", 12: "FS"}
        for idx, i in enumerate(range(13, 23)):
            self.test_mapping[i] = f"Drug {idx + 1}"
        self.test_mapping[23] = "BCL"
        self.test_mapping[24] = "BCH"

        self.test_file_content = "\n".join([f"{k}:{v}" for k, v in self.test_mapping.items()])

    def test_read_input_file(self):
        with tempfile.NamedTemporaryFile(mode="w+t", suffix=".txt") as self.channel_file:
            self.channel_file.write(self.test_file_content)
            self.channel_file.seek(0)
            self.channel_file_path = pl.Path(self.channel_file.name)

            mapping = ChannelMap(input_file=self.channel_file_path)

        self.assertEqual(self.test_mapping, mapping.map)

    def test_read_broken_input_file_channel(self):
        with tempfile.NamedTemporaryFile(mode="w+t", suffix=".txt") as self.channel_file:
            self.channel_file.write("1:Test")
            self.channel_file.seek(0)
            self.channel_file_path = pl.Path(self.channel_file.name)

            with self.assertRaises(ValueError) as cm:
                mapping = ChannelMap(self.channel_file_path)

        self.assertEqual(cm.exception.args[0], f"Channel out of BD range (9-24) you specified channel 1")

    def test_read_broken_input_file_channel_nan(self):
        with tempfile.NamedTemporaryFile(mode="w+t", suffix=".txt") as self.channel_nan_file:
            self.channel_nan_file.write("ABC:Test")
            self.channel_nan_file.seek(0)
            self.channel_nan_file_path = pl.Path(self.channel_nan_file.name)

            with self.assertRaises(ValueError) as cm:
                mapping = ChannelMap(self.channel_nan_file_path)

        self.assertEqual(cm.exception.args[0], "Channel has to be an int you specified ABC")




class TestPlugSequence(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
