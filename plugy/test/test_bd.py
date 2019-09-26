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
import unittest


class TestChannelMapping(unittest.TestCase):
    def setUp(self) -> None:
        header = "9:CELLS\n10:SUBSTRATE\n11:FS\n12:FS\n13:Drug1"
        drugs = [f"{i}:Drug{i}" for i in range(13,)]
        # self.test_file_content =

    def test_read_input_file(self):
        self.assertEqual(True, False)


class TestPlugSequence(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
