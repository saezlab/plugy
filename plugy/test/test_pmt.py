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
import unittest
import unittest.mock as mock
import tempfile

import logging
import gzip
import pathlib as pl

import numpy as np

import pandas as pd
import pandas.util.testing as pd_test

from ..data import pmt

FILE_CONTENT = """LabVIEW Measurement
Writer_Version\t2
Reader_Version\t2
Separator\tTab
Decimal_Separator\t,
Multi_Headings\tNo
X_Columns\tNo
Time_Pref\tRelative
Operator\tEveline
Date\t2019/07/22
Time\t17:18:18,938945770263671875
***End_of_Header***

Channels\t4
Samples\t100\t100\t100\t100
Date\t2019/07/22\t2019/07/22\t2019/07/22\t2019/07/22
Time\t17:18:18,938945770263671875\t17:18:18,938945770263671875\t17:18:18,938945770263671875\t17:18:18,938945770263671875
X_Dimension\tTime\tTime\tTime\tTime
X0\t0,0000000000000000E+0\t0,0000000000000000E+0\t0,0000000000000000E+0\t0,0000000000000000E+0
Delta_X\t1,000000\t1,000000\t1,000000\t1,000000
***End_of_Header***
X_Value\tUntitled\tUntitled 1\tUntitled 2\tUntitled 3\tComment
\t0,000000\t0,055544\t0,032960\t0,071718
\t0,000000\t0,054323\t0,032044\t0,049745
\t0,000000\t0,055239\t0,032655\t0,050050
\t0,000000\t0,053713\t0,031739\t0,049135
\t0,000000\t0,055544\t0,032655\t0,048830
\t0,000000\t0,055849\t0,033265\t0,050356
"""

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%d.%m.%y %H:%M:%S')


class TestPmtData(unittest.TestCase):
    def setUp(self):
        """
        Creates DataFrame for each unittest to compare results
        """
        self.test_df = pd.DataFrame({"time": [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
                                     "green": [0.055544, 0.054323, 0.055239, 0.053713, 0.055544, 0.055849],
                                     "orange": [0.032960, 0.032044, 0.032655, 0.031739, 0.032655, 0.033265],
                                     "uv": [0.071718, 0.049745, 0.050050, 0.049135, 0.048830, 0.050356]})

    def test_gz_file_open(self):
        """
        Checks if reading gz file returns the expected DataFrame
        """
        with tempfile.NamedTemporaryFile(mode="w+b", suffix=".txt.gz", delete=True) as self.gz_file:
            with gzip.GzipFile(mode="wb", fileobj=self.gz_file) as gz:
                gz.write(FILE_CONTENT.encode())
            self.gz_file.seek(0)
            self.gz_file_path = pl.Path(self.gz_file.name)

            self.data = pmt.PmtData(self.gz_file_path).read_txt()

        pd_test.assert_frame_equal(self.test_df, self.data)

    def test_txt_file_open(self):
        """
        Checks if reading normal txt file returns the expected DataFrame
        """
        with tempfile.NamedTemporaryFile(mode="w+t", suffix=".txt", delete=True) as self.txt_file:
            self.txt_file.write(FILE_CONTENT)
            self.txt_file.seek(0)
            self.txt_file_path = pl.Path(self.txt_file.name)

            self.data = pmt.PmtData(self.txt_file_path).read_txt()

        pd_test.assert_frame_equal(self.test_df, self.data)

    def test_other_file_open(self):
        """
        Checks error handling in read_txt
        """
        self.suffix = ".any"
        with tempfile.NamedTemporaryFile(suffix=self.suffix) as self.any_file:
            self.any_file_path = pl.Path(self.any_file.name)
            with self.assertRaises(NotImplementedError) as cm:
                data = pmt.PmtData(self.any_file_path).read_txt()

            self.assertEqual(cm.exception.args[0], f"Input file has to be either .txt or .txt.gz, {self.any_file_path.suffix} files are not implemented!")

    def test_find_data(self):
        """
        Checks error handling in find_data
        """
        # Checking exception raise with empty file
        with tempfile.TemporaryFile() as self.empty_file:
            with self.assertRaises(AssertionError) as cm:
                pmt.PmtData.find_data(self.empty_file)

            self.assertEqual(cm.exception.args[0], "No lines detected in input_file! Check the contents of the file!")

        # Checking error raised with too long header
        with tempfile.TemporaryFile(mode="w+t") as self.wrong_header_file:
            self.wrong_header_file.writelines(["test\n" for _ in range(100)])
            self.wrong_header_file.seek(0)

            with self.assertRaises(AssertionError) as cm:
                pmt.PmtData.find_data(self.wrong_header_file)

        # Checking with real header
        with tempfile.TemporaryFile(mode="w+t") as self.right_file:
            self.right_file.write(FILE_CONTENT)
            self.right_file.seek(0)

            self.assertEqual(pmt.PmtData.find_data(self.right_file), 22)

    # noinspection PyArgumentList
    def test_set_channel_value_ignore(self):
        """
        Tests ignoring individual channels
        """
        test_df_zero_green = self.test_df.assign(green=0.0)
        test_df_zero_uv = self.test_df.assign(uv=0.0)
        test_df_zero_orange = self.test_df.assign(orange=0.0)

        with unittest.mock.patch.object(target=pmt.PmtData, attribute="read_txt", new=lambda _: self.test_df):
            with self.subTest():
                data = pmt.PmtData(pl.Path(), correct_acquisition_time=False, ignore_green_channel=True).data
                pd_test.assert_frame_equal(data, test_df_zero_green)

                data = pmt.PmtData(pl.Path(), correct_acquisition_time=False, ignore_orange_channel=True).data
                pd_test.assert_frame_equal(data, test_df_zero_orange)

                data = pmt.PmtData(pl.Path(), correct_acquisition_time=False, ignore_uv_channel=True).data
                pd_test.assert_frame_equal(data, test_df_zero_uv)

    # noinspection PyArgumentList
    def test_set_channel_value_time(self):
        """
        Tests correcting the time values
        """
        with unittest.mock.patch.object(target=pmt.PmtData, attribute="read_txt", new=lambda _: self.test_df):
            for acq in range(100, 500, 100):
                with self.subTest(acq=acq):
                    test_df_time = self.test_df.assign(time=np.linspace(0, (1 / acq) * (len(self.test_df) - 1), len(self.test_df)))
                    data = pmt.PmtData(pl.Path(), correct_acquisition_time=True, acquisition_rate=acq).data
                    pd_test.assert_frame_equal(data, test_df_time)

            test_df_time = self.test_df.assign(time=np.linspace(0, len(self.test_df) - 1, len(self.test_df)))
            data = pmt.PmtData(pl.Path(), correct_acquisition_time=True, acquisition_rate=1).data
            pd_test.assert_frame_equal(data, test_df_time)

            data = pmt.PmtData(pl.Path(), correct_acquisition_time=False).data
            pd_test.assert_frame_equal(data, self.test_df)

    def test_cut_data(self):
        """
        Tests if input is cut properly
        """
        with unittest.mock.patch.object(target=pmt.PmtData, attribute="read_txt", new=lambda _: self.test_df):
            cut = (1, 4)
            data = pmt.PmtData(input_file=pl.Path(), acquisition_rate=1, correct_acquisition_time=True, cut=cut).data
            self.assertTrue(len(data.time) == 4)
            self.assertTrue(min(data.time) >= cut[0])
            self.assertTrue(max(data.time) <= cut[1])

            cut = (None, None)
            data = pmt.PmtData(input_file=pl.Path(), acquisition_rate=1, correct_acquisition_time=True, cut=cut).data
            self.assertTrue(len(data.time) == len(self.test_df))

            cut = (3, None)
            data = pmt.PmtData(input_file=pl.Path(), acquisition_rate=1, correct_acquisition_time=True, cut=cut).data
            self.assertTrue(len(data.time) == 3)
            self.assertTrue(min(data.time) >= cut[0])

            cut = (None, 3)
            data = pmt.PmtData(input_file=pl.Path(), acquisition_rate=1, correct_acquisition_time=True, cut=cut).data
            self.assertTrue(len(data.time) == 4)
            self.assertTrue(max(data.time) <= cut[1])

            cut = (4, 1)
            with self.assertRaises(AttributeError) as cm:
                data = pmt.PmtData(input_file=pl.Path(), acquisition_rate=1, correct_acquisition_time=True, cut=cut).data
            self.assertEqual(cm.exception.args[0], f"Cut has to be specified like cut=(min, max) you specified {cut}")

    # noinspection DuplicatedCode
    def test_cut_additional_data(self):
        """
        Tests if cutting of supplied df works opposed to using the df in the PmtData object
        """
        with unittest.mock.patch.object(target=pmt.PmtData, attribute="read_txt", new=lambda _: self.test_df):
            raw_data = pmt.PmtData(input_file=pl.Path(), acquisition_rate=1, correct_acquisition_time=True, cut=(None, None))

        cut = (1, 4)
        data = raw_data.cut_data(cut=cut)
        self.assertTrue(len(data.time) == 4)
        self.assertTrue(min(data.time) >= cut[0])
        self.assertTrue(max(data.time) <= cut[1])

        cut = (None, None)
        data = raw_data.cut_data(cut=cut)
        self.assertTrue(len(data.time) == len(self.test_df))

        cut = (3, None)
        data = raw_data.cut_data(cut=cut)
        self.assertTrue(len(data.time) == 3)
        self.assertTrue(min(data.time) >= cut[0])

        cut = (None, 3)
        data = raw_data.cut_data(cut=cut)
        self.assertTrue(len(data.time) == 4)
        self.assertTrue(max(data.time) <= cut[1])

        cut = (4, 1)
        with self.assertRaises(AttributeError) as cm:
            data = raw_data.cut_data(cut=cut)
        self.assertEqual(cm.exception.args[0], f"Cut has to be specified like cut=(min, max) you specified {cut}")

    def test_digital_gain(self):
        """
        Tests digital gain method
        """
        with unittest.mock.patch.object(target=pmt.PmtData, attribute="read_txt", new=lambda _: self.test_df):
            data = pmt.PmtData(input_file=pl.Path(), acquisition_rate=1, correct_acquisition_time=False, digital_gain_uv=2).data
            pd_test.assert_frame_equal(data, self.test_df.assign(uv=[0.143436, 0.09949, 0.1001, 0.09827, 0.09766, 0.100712]))

            data = pmt.PmtData(input_file=pl.Path(), acquisition_rate=1, correct_acquisition_time=False, digital_gain_green=2).data
            pd_test.assert_frame_equal(data, self.test_df.assign(green=[0.111088, 0.108646, 0.110478, 0.107426, 0.111088, 0.111698]))

            data = pmt.PmtData(input_file=pl.Path(), acquisition_rate=1, correct_acquisition_time=False, digital_gain_orange=2).data
            pd_test.assert_frame_equal(data, self.test_df.assign(orange=[0.06592, 0.064088, 0.06531, 0.063478, 0.06531, 0.06653]))


if __name__ == '__main__':
    unittest.main()
