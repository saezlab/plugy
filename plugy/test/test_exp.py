"""
Author      Nicolas Peschke
Date        30.10.2019

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
import unittest.mock as mock
import tempfile
import logging
import itertools

import pathlib as pl

from ..exp import PlugExperiment
from ..data.config import PlugyConfig

logging.basicConfig(level = logging.DEBUG,
                    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt = '%d.%m.%y %H:%M:%S')


class TestPlugExperiment(unittest.TestCase):
    def test_check_config_file_names(self):
        """
        Tests if filename checking is properly implemented
        """
        with mock.patch.object(target = PlugExperiment, attribute = "__post_init__", new = PlugExperiment.check_config):
            with tempfile.NamedTemporaryFile(mode = "w+t", suffix = ".csv") as self.test_file:
                test_file_path = pl.Path(self.test_file.name)
                test_file_names = set(itertools.permutations([None, None, None, test_file_path, test_file_path, test_file_path], 3))

                for file_names in test_file_names:
                    with self.subTest():
                        if None in file_names:
                            with self.assertRaises(AssertionError) as cm:
                                PlugExperiment(config = PlugyConfig(pmt_file = file_names[0], seq_file = file_names[1], channel_file = file_names[2]))
                            self.assertEqual(cm.exception.args[0], "One or more file paths are not properly specified, see the log for more information!")

                        else:
                            try:
                                PlugExperiment(config = PlugyConfig(pmt_file = file_names[0], seq_file = file_names[1], channel_file = file_names[2]))
                            except AssertionError:
                                self.fail("PlugExperiment raised an AssertionError even though it was supplied with proper file names")


if __name__ == '__main__':
    unittest.main()
