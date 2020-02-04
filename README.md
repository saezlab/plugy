# Plugy Tutorial
[![pipeline status](https://git.embl.de/grp-merten/plugy/badges/master/pipeline.svg)](https://git.embl.de/grp-merten/plugy/commits/master)

This notebook will show you how to run a plugy based analysis of a Braille experiment.

#### Issue reporting
Please feel free to open issues for the tutorial at https://git.embl.de/grp-merten/plugy-tutorial/issues
in case there are some things unclear or you spotted some errors in the tutorial.

In case you have some issue or feature request with plugy directly, please open an issue
directly on the site of plugy's repository https://git.embl.de/grp-merten/plugy/issues 

#### Running plugy

Imports & Setup

You can now install plugy as a package using `pip` in your `conda` environment.
To install `pip` in your `conda` environment run the following lines on your `bash` or `conda` prompt.
```
# Activate your conda environment replacing 'YOUR_ENVIRONMENT' with the name of your environment
conda activate YOUR_ENVIRONMENT

# Install pip git support, such that plugy can be directly installed from gitlab
conda install pip git

# Install plugy into your environment
pip install git+git+https://git.embl.de/grp-merten/plugy@master
```

Logging is to have log output to be printed to screen and to file,
pathlib is to handle file paths in an OS independent manner. 
```python
import logging
import pathlib as pl
```
Importing plugy modules
```python
import plugy.exp as exp
import plugy.data.config as config
```
Setting up logging 
```python
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%d.%m.%y %H:%M:%S')
```
Running Plugy

Creating configuration
```python
plugy_config = config.PlugyConfig(pmt_file=pl.Path("data/pmt_data.txt.gz"),
                                  seq_file=pl.Path("data/sequence.csv"),
                                  channel_file=pl.Path("data/channel_map.csv"),
                                  auto_detect_cycles=True,
                                  peak_max_width=2.5,
                                  figure_export_file_type="png")
```
Running the analysis
this will automatically create result directories for each individual run.
```python
plug_exp = exp.PlugExperiment(plugy_config)
```
If you want to interact with the data use the contents of the plug_exp object. 
It contains all the plug, pmt, channel and sequence data that was used in the analysis.  
This would give you a DataFrame containing the statistics to each sample

```python
plug_exp.sample_statistics()
```
