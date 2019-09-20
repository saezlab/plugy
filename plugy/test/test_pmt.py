import unittest
import tempfile
import gzip
import pathlib as pl
import pandas as pd
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

# FILE_DF = pd.DataFrame({"time": [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
#                         "green": [0.055544, 0.054323, 0.055239, 0.053713, 0.055544, 0.055849],
#                         "orange": [0.032960, 0.032044, 0.032655, 0.031739, 0.032655, 0.033265],
#                         "uv": [0.071718, 0.049745, 0.050050, 0.049135, 0.048830, 0.050356]})


# FILE_DF = pd.DataFrame(
#     [0.000000, 0.055544, 0.032960, 0.071718]
#     [0.000000, 0.054323, 0.032044, 0.049745]
#     [0.000000, 0.055239, 0.032655, 0.050050]
#     [0.000000, 0.053713, 0.031739, 0.049135]
#     [0.000000, 0.055544, 0.032655, 0.048830]
#     [0.000000, 0.055849, 0.033265, 0.050356])


class TestPmtData(unittest.TestCase):
    def setUp(self):
        self.txt_file = tempfile.NamedTemporaryFile(mode="wt", suffix=".txt", delete=True)
        self.txt_file.write(FILE_CONTENT)
        # print(self.txt_file.name)
        self.txt_file_path = pl.Path(self.txt_file.name)

        # self.gz_file = tempfile.NamedTemporaryFile(mode="w+b", suffix=".txt.gz", delete=True)
        # # self.gz_file.write(FILE_CONTENT.encode())
        # gz = gzip.GzipFile(mode="wb", fileobj=self.gz_file)
        # # gz.write(FILE_CONTENT)
        # gz.write(FILE_CONTENT.encode())
        # gz.close()
        # self.gz_file.seek(0)
        # self.gz_file_path = pl.Path(self.gz_file.name)

        self.test_df = pd.DataFrame({"time": [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
                                     "green": [0.055544, 0.054323, 0.055239, 0.053713, 0.055544, 0.055849],
                                     "orange": [0.032960, 0.032044, 0.032655, 0.031739, 0.032655, 0.033265],
                                     "uv": [0.071718, 0.049745, 0.050050, 0.049135, 0.048830, 0.050356]})
        # print(self.gz_file_path)

    def tearDown(self) -> None:
        self.txt_file.close()
        self.gz_file.close()

    def test_temp_gz_file(self):
        self.assertTrue(self.gz_file_path.exists())

    def test_temp_txt_file(self):
        self.assertTrue(pl.Path(self.txt_file.name).exists())

    # def test_true(self):
    #     self.assertEqual(True, True)

    def test_gz_file_open(self):
        with tempfile.NamedTemporaryFile(mode="w+b", suffix=".txt.gz", delete=True) as self.gz_file:
            # self.gz_file.write(FILE_CONTENT.encode())
            with gzip.GzipFile(mode="wb", fileobj=self.gz_file) as gz:
                # gz.write(FILE_CONTENT)
                gz.write(FILE_CONTENT.encode())
                # gz.close()
            self.gz_file.seek(0)
            self.gz_file_path = pl.Path(self.gz_file.name)

            data = pmt.PmtData(self.gz_file_path).read_txt()

        self.assertTrue(data.equals(self.test_df))
        # self.assertEqual(data.all(), FILE_DF.all())
        # self.assertTrue(data.equals(self.test_df))
    #     # self.assertEqual(True, True)
    #     assert 1


class MyTestCase(unittest.TestCase):
    def test_something(self, tmpdir):
        print(tmpdir)
        # self.assertEqual(True, True)
        assert 1


def test_something_else(tmp_path):
    print(tmp_path)
    assert 1


if __name__ == '__main__':
    unittest.main()
