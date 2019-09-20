import unittest
import tempfile
import gzip
import pathlib as pl
import pandas as pd
import pandas.util.testing as pd_test
import lib.plugy.plugy.data.pmt as pmt

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

            data = pmt.PmtData(self.gz_file_path).read_txt()

        pd_test.assert_frame_equal(self.test_df, data)

    def test_txt_file_open(self):
        """
        Checks if reading normal txt file returns the expected DataFrame
        """
        with tempfile.NamedTemporaryFile(mode="w+t", suffix=".txt", delete=True) as self.txt_file:
            self.txt_file.write(FILE_CONTENT)
            self.txt_file.seek(0)
            self.txt_file_path = pl.Path(self.txt_file.name)

            data = pmt.PmtData(self.txt_file_path).read_txt()

        pd_test.assert_frame_equal(self.test_df, data)


if __name__ == '__main__':
    unittest.main()
