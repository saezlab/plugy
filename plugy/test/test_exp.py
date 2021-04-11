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


import os
import unittest
import tempfile
import logging
import shutil

import pathlib as pl

from ..exp import PlugExperiment
from ..data.config import PlugyConfig

logging.basicConfig(
    level = logging.DEBUG,
    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt = '%d.%m.%y %H:%M:%S',
)


class TestPlugExperiment(unittest.TestCase):


    def setUp(self):

        self.tmpdir = tempfile.mkdtemp()


    def tearDown(self):

        shutil.rmtree(self.tmpdir)


    def test_check_input_file_names(self):
        """
        Tests if input file search is properly implemented.
        """

        fnames = (
            ('exp.txt', 'channels.tsv', 'sequence_879.csv'),
            ('Experiment_42.txt.gz', 'CHANNELS-xy.CSV', None),
            ('exp.xlsx', 'channel_layout.csv', 'Seq029.tsv'),
            ('fluorescence.txt', 'channel_layout.csv', 'Seq029.tsv'),
            ('fluor.TXT.gz', 'CHANNELS-xy.CSV', 'seq.tsv'),
        )

        attrs = ('pmt', 'channel', 'seq')

        for test_fnames in fnames:

            for fn in test_fnames:

                if fn is None:

                    continue

                path = pl.Path(self.tmpdir, fn)
                path.touch()

            if test_fnames[0].endswith('xlsx'):

                self.assertRaises(
                    FileNotFoundError,
                    lambda: PlugyConfig(input_dir = self.tmpdir)
                )

            else:

                cfg = PlugyConfig(input_dir = self.tmpdir)

                for fn, attr in zip(test_fnames, attrs):

                    attr = '%s_file' % attr
                    path = (
                        None
                            if fn is None else
                        pl.Path(self.tmpdir, fn)
                    )

                    self.assertEqual(getattr(cfg, attr), path)

            for f in os.listdir(self.tmpdir):

                os.remove(os.path.join(self.tmpdir, f))


if __name__ == '__main__':

    unittest.main()
