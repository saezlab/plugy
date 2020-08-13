# Plugy Tutorial
[![pipeline status](https://git.embl.de/grp-merten/plugy/badges/master/pipeline.svg)](https://git.embl.de/grp-merten/plugy/commits/master)

This notebook will show you how to run a plugy based analysis of a Braille experiment.

#### Issue reporting
In case you have some issue or feature request with `plugy`, please open an issue
with a detailed description of the problem and the steps taken to resolve
it here: https://git.embl.de/grp-merten/plugy/issues 

#### Running plugy

Imports & Setup

You can now install `plugy` as a package using `pip` in your `conda` environment.
To install `pip` in your `conda` environment run the following lines on your `bash` or `conda` prompt.
```
# Activate your conda environment replacing 'YOUR_ENVIRONMENT' with the name of your environment
conda activate YOUR_ENVIRONMENT

# Install pip git support, such that plugy can be directly installed from gitlab
conda install pip git

# Install plugy into your environment
pip install git+https://git.embl.de/grp-merten/plugy@master

# If you want to use the latest development version use this instead
pip install --force-reinstall git+https://git.embl.de/grp-merten/plugy@dev
```

Importing modules
```python
# Handling file paths
import pathlib as pl

# Importing plugy modules
import plugy.exp as exp
import plugy.data.config as config
import plugy.misc as misc
```

Running Plugy

Creating configuration & setting up logging
This will automatically create result directories for each individual run.
The `misc.start_logging` function will cause `plugy` to display INFO and higher level 
messages on screen and log every message to a log file in the result directory.
```python
plugy_config = config.PlugyConfig(pmt_file=pl.Path("data/pmt_data.txt.gz"),
                                  seq_file=pl.Path("data/sequence.csv"),
                                  channel_file=pl.Path("data/channel_map.csv"),
                                  auto_detect_cycles=True,
                                  peak_max_width=2.5,
                                  figure_export_file_type="png")

misc.start_logging(plugy_config)
```
Running the analysis

```python
plug_exp = exp.PlugExperiment(plugy_config)
```
If you want to interact with the data use the contents of the plug_exp object. 
It contains all the plug, pmt, channel and sequence data that was used in the analysis. 
 
For example, this would give you a DataFrame containing the statistics to each sample:
```python
plug_exp.sample_statistics
```
