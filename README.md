# Plugy: Python module for plug microfluidics data analysis
[![pipeline status](https://github.com/saezlab/plugy/actions/workflows/python-package-conda.yml/badge.svg)](https://git.embl.de/grp-merten/plugy/commits/master)

## Issues

Feedback, questions, bug reports are welcome:
https://github.com/saezlab/plugy/issues

## Installation

Imports & Setup

You can now install `plugy` as a package using `pip` in your `conda`
environment. To install `pip` in your `conda` environment run the following
lines on your `bash` or `conda` prompt.

```bash
# Activate your conda environment replacing 'YOUR_ENVIRONMENT' with the name of your environment
conda activate YOUR_ENVIRONMENT

# Install pip git support, such that plugy can be directly installed from gitlab
conda install pip git

# Install plugy into your environment
pip install git+https://github.com/saezlab/plugy@master

# If you want to use the latest development version use this instead
pip install --force-reinstall git+https://github.com/saezlab/plugy@dev
```

## Quick start

This notebook will show you how to run a plugy based analysis of a drug
combination Braille display microfluidics experiment.

First, make sure your Python shell is running in the working directory where
(or in its subdirectories) you have the data and where you want to save the
results.

The simplest workflow, which is sufficient most of the times, looks like this:

```python
import plugy
exp = plugy.PlugExperiment()
exp.main()
```

Further settings, parameters can be passed to `PlugExperiment`:

```python
import plugy
exp = plugy.PlugExperiment(
    peak_min_threshold = 0.02,
    barcoding_param = {
        'times': (.2, 4.0),
    },
    heatmap_second_scale = 'pos-ctrl',
)
exp.main()
```

If you want to interact with the data use the contents of the `exp` object.
It contains all the plug, pmt, channel and sequence data that was used in the
analysis. For example, a `pandas.DataFrame` containing the statistics for each
sample:

```python
exp.sample_statistics
```

## Tutorial

You can find more examples in the plugy guide:
https://github.com/saezlab/plugy/blob/master/notebooks/plugy_guide.ipynb

## Development history

https://github.com/saezlab/plugy/blob/master/NEWS.md
