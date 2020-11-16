# Plugy: Python module for plug microfluidics data analysis
[![pipeline status](https://git.embl.de/grp-merten/plugy/badges/master/pipeline.svg)](https://git.embl.de/grp-merten/plugy/commits/master)

## Issues

Please feel free to open issues for the tutorial at https://git.embl.de/grp-merten/plugy-tutorial/issues
in case there are some things unclear or you spotted some errors in the tutorial.

In case you have some issue or feature request with `plugy` directly, please open an issue
directly on the site of the `plugy` repository https://git.embl.de/grp-merten/plugy/issues

## Installation

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
```

## Quick start

Importing the modules

```python
# Importing plugy modules
import plugy.exp as exp
import plugy.data.config as config
```

Setting up Plugy

```python
plugy_config = config.PlugyConfig(
    pmt_file = 'data/pmt_data.txt.gz',
    seq_file = 'data/sequence.csv',
    channel_file = 'data/channel_map.csv',
    auto_detect_cycles = True,
    peak_max_width = 2.5,
    figure_export_file_type = 'png',
)

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

## Tutorial

You can find more examples in the plugy guide:
https://git.embl.de/grp-merten/plugy/-/blob/dev/notebooks/plugy_guide.ipynb

## Development history

https://git.embl.de/grp-merten/plugy/-/blob/dev/NEWS.md