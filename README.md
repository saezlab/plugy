# Plugy Tutorial
This notebook will show you how to run a plugy based analysis of a Braille experiment.Imports & Setup
General imports, logging is to have log output to be printed to screen and to file,
pathlib is to handle file paths in an OS independent manner. 
```
import logging
import pathlib as pl
```
Importing plugy modules
```
import lib.plugy.plugy.exp as exp
import lib.plugy.plugy.data.config as config
```
Setting up logging 
```
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%d.%m.%y %H:%M:%S')
```
Running Plugy
```
Creating configuration plugy_config = config.PlugyConfig(pmt_file=pl.Path("data/pmt_data.txt.gz"),
                                  seq_file=pl.Path("data/sequence.csv"),
                                  channel_file=pl.Path("data/channel_map.csv"),
                                  auto_detect_cycles=True,
                                  peak_max_width=2.5,
                                  figure_export_file_type="png")
```
Running the analysis
this will automatically create result directories for each individual run.
```
plug_exp = exp.PlugExperiment(plugy_config)
```
If you want to interact with the data use the contents of the plug_exp object. 
It contains all the plug, pmt, channel and sequence data that was used in the analysis.  
This would give you a DataFrame containing the statistics to each sample

```
plug_exp.sample_statistics
```