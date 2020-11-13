## 12 Nov 2020

* Better separation of the phases of the workflow
  - Raw data can be processed independently of further steps
  - Even without channel map or sample sequence
  - Or without any barcoding
  - We can detect the plugs without barcode
  - We can detect the barcode without identifying samples
* Control the workflow with new `config` options
  - `run`: attempt to run the whole typical workflow
  - `init`: only load the data
  - `plugs`: run only the plug detection
  - `has_barcode`: attempt to identify barcode plugs
  - `has_samples_cycles`: attempt to identify samples and cycles
  - `samples_per_cycle`: if run without a sequence file, you can
    provide here the number of samples in a cycle
* New parameters for the sample and cycle detection
  - `min_end_cycle_barcodes`: after how many consecutive barcode plugs
    we shall we start a new cycle
  - `min_between_samples_barcodes`: lowest number of barcode plugs
    separating two samples
  - `min_plugs_in_sample`: lowest number of plugs in a sample
* A new experimental method for barcode detection which aims to deal
  with the irregular, noisy data. This method doesn't work perfectly,
  requires fine-tuning of the parameters and normally we shouldn't
  use it as a decent quality experiment can be processed with the
  conventional methods. New parameters in `config`:
  - `barcoding_method`: either `simple` or `adaptive`
  - `barcoding_param`: dictionary with parameters for the barcoding
    method
  The `simple` method is the old behaviour and has only one parameter:
  - `times`: the blue channel should be at least this times higher
    than the control channel for barcode plugs
  With the `adaptive` method you can choose an alternative method and
  also scan a parameter space for this method.
* From now we save the results (plots) simply in a designated
  directory, by default `results`, without creating subdirectories for
  each run adding timestamp to the directory names. If you need
  subdirectories and timestamps use the following options:
  - `result_subdirs`
  - `timestamp_result_subdirs`
* A new option to make `plugy` silent, i.e. not to flood the console
  with log messages. We always have the log messages in the log file.
  - `log_to_stdout`: we keep the old behaviour, if you want to make it
    silent set this option to `False`
* No longer need to manually start the logging, it happens automatically
* No need to define paths with `pathlib`, you can simply provide them as
  strings
