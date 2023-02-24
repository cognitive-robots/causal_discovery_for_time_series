# Causal Discovery for Time Series
Framework for evaluating temporal observation-based causal discovery techniques on real and synthetic data.

## Methods

* GrangerPW: https://www.jair.org/index.php/jair/article/view/13428
* GrangerMV: https://pubmed.ncbi.nlm.nih.gov/20481753
* TCDF: https://www.mdpi.com/2504-4990/1/1/19
* NAVARMLP: https://link.springer.com/chapter/10.1007/978-3-030-88942-5_35
* PCMCIParCorr: http://proceedings.mlr.press/v124/runge20a.html
* tsFCI: https://www.academia.edu/download/30837559/10.1.1.173.254.pdf#page=131
* CDNOD: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5617646
* VarLiNGAM: https://www.jmlr.org/papers/v11/hyvarinen10a.html
* TiMINo: https://proceedings.neurips.cc/paper/2013/hash/47d1e990583c9c67424d369f3414728e-Abstract.html
* DYNOTEARS: http://proceedings.mlr.press/v108/pamfil20a.html

## Scripts

### Test AD
Evaluates a given method against a batch of autonomous driving CSV data files. Assumes the CSV data files describe a two-agent convoy scenario.

    usage: test_ad.py [-h] [--processor-count PROCESSOR_COUNT] [--verbose] [--max-time-lag MAX_TIME_LAG] [--sig-level SIG_LEVEL] method dataset variable

Parameters:
* method: Method to evaluate.
* dataset: Dataset folder under the "data" directory to acquire CSV data files from.
* variable: Sub-dataset to use under the dataset directory. Determines the variable of interest for each agent captured in the CSV data files.
* -h: Displays the help message for the script.
* --processor-count: Specifies the maximum number of processors to use. Only usable with methods which do not rely upon using an intermediary output file due to race condition risks.
* --verbose: Dictates that output should be verbose.
* --max-time-lag: Specifies the maximum time lag parameter for the selected method.
* --sig-level: Specifies the significance alpha parameter for the selected method.

### Run All AD Tests
Evaluates all methods, across all datasets and sub-datasets for a given set of parameters. Primarily useful for setting off batch jobs.

    usage: [SKIP_TO_METHOD=?] [OVERRIDE_METHODS=?] [OVERRIDE_DATASETS=?] [OVERRIDE_VARIABLES=?] ./run_all_ad_tests.sh

Parameters:
* SKIP_TO_METHOD: Any methods listed before the specified method in the script will be skipped.
* OVERRIDE_METHODS: Completely overrides the methods to evaluate listed in the script.
* OVERRIDE_DATASETS: Completely overrides the datasets to evaluate upon listed in the script.
* OVERRIDE_VARIABLES: Completely overrides the variables/sub-datasets to evaluate upon listed in the script.
