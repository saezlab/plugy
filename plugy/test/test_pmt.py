import unittest
import tempfile
import gzip
import pathlib as pl
import lib.plugy.plugy.data.pmt as pmt

CONTENT = """LabVIEW Measurement
Writer_Version	2
Reader_Version	2
Separator	Tab
Decimal_Separator	,
Multi_Headings	No
X_Columns	No
Time_Pref	Relative
Operator	Eveline
Date	2019/07/22
Time	17:18:18,938945770263671875
***End_of_Header***

Channels	4
Samples	100	100	100	100
Date	2019/07/22	2019/07/22	2019/07/22	2019/07/22
Time	17:18:18,938945770263671875	17:18:18,938945770263671875	17:18:18,938945770263671875	17:18:18,938945770263671875
X_Dimension	Time	Time	Time	Time
X0	0,0000000000000000E+0	0,0000000000000000E+0	0,0000000000000000E+0	0,0000000000000000E+0
Delta_X	1,000000	1,000000	1,000000	1,000000
***End_of_Header***
X_Value	Untitled	Untitled 1	Untitled 2	Untitled 3	Comment
	0,000000	0,055544	0,032960	0,071718
	0,000000	0,054323	0,032044	0,049745
	0,000000	0,055239	0,032655	0,050050
	0,000000	0,053713	0,031739	0,049135
	0,000000	0,055544	0,032655	0,048830
	0,000000	0,055849	0,033265	0,050356"""


class TestPmtData(unittest.TestCase):
    def setUp(self):
        with tempfile.NamedTemporaryFile(mode="wt", suffix=".txt") as txt_file:
            txt_file.write(CONTENT)
            self.txt_file = txt_file
            print(self.txt_file.name)
            self.txt_file_path = pl.Path(self.txt_file.name)

            # with tempfile.NamedTemporaryFile(suffix=".txt.gz") as gz_file:
            #     self.gz_file_path = pl.Path(gz_file.name)
            #     self.gz_file = gzip.GzipFile(mode="wb", fileobj=txt_file)

    # def tearDown(self) -> None:
    #

    # def test_temp_gz_file(self):
    #     self.assertTrue(self.gz_file_path.exists())

    def test_temp_txt_file(self):
        # self.assertTrue(self.txt_file_path.exists())
        self.assertTrue(pl.Path(self.txt_file.name).exists())

    # def test_true(self):
    #     self.assertEqual(True, True)

    # def test_gz_file_open(self):
    #     pmt.PmtData(self.gz_file_path)
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
