# :crystal_ball: PLASMA

**PLASMA (PopuLation Allele-Specific MApping) is a statistical fine-mapping method for functional data using QTL and allelic-imbalance signal.**

[Preprint for the PLASMA method](https://www.biorxiv.org/content/10.1101/650242v1)

Developed at the [Gusev Lab](http://gusevlab.org/) at the Dana Farber Cancer Institute / Harvard Medical School.

## Installation and Dependencies

PLASMA utilizes Python 3.6+ and requires the following Python packages for core functionality:
* [Numpy 1.15+](https://scipy.org/install.html)
* [Scipy 1.1+](https://scipy.org/install.html)

The following packages are optional, but are used for pre/post-processing:
* [Pandas 0.23+](https://pandas.pydata.org/pandas-docs/stable/install.html)
* [Matplotlib 2.2+](https://matplotlib.org/users/installing.html)
* [Seaborn 0.9+](https://seaborn.pydata.org/installing.html)
* [Pysam 0.15+](https://pysam.readthedocs.io/en/latest/installation.html)
* [PyVCF 0.6+](https://pyvcf.readthedocs.io/en/latest/INTRO.html)

All packages can be installed using Python's pip package manager.

To download PLASMA, click "Clone or Download" or enter in command line:
```
git clone https://github.com/austintwang/plasma
```

## `run_plasma.py` : Quick-start fine-mapping script

The `run_plasma.py` script conducts fine-mapping of a single locus with default PLASMA parameters and outputs.

### Input Files and Parameters
The script requires the following files:
* Two text files (one for each haplotype), specifying the haplotype-specific genotypes, across samples and marker. Each row should represent an individual, and each column should represent a marker. The ordering of samples and markers should be the same for both files.
* Two text files (one for each haplotype), specifying the haplotype-specific phenotypes (e.g. read counts) across samples. The order of samples should be the same as that of the genotype files.
* A text file, specifying the total phenotype across samples. If none is provided, then the total phenotype is assumed to be the sum of the haplotype-specific phenotypes.

Other parameters include:
* Individual-level or global beta-binomial overdispersions.
* AS-Only and QTL-only modes, where the total phenotype and allele-specific phenotypes are ignored, respectively.
* Search parameters, including the maximum number of causal variants and the search mode (exhaustive or stochastic shotgun search)
* The confidence level when creating the credible set

### Output Files
The script outputs two files in the specified output directory:
* `cset.txt`: The minimal set of markers that contains the set of true causal markers, at the specified confidence level. `1` and `0` denote that a marker is included in and excluded from the credible set, respectively. The order of the markers is the same as that in the genotype files.
* `ppas.txt`: The marginal posterior probabilities of each marker being causal.

### Usage
Usage of the script is as follows:
```
usage: run_plasma.py [-h] [--total_exp_path TOTAL_EXP_PATH]
                     [--overdispersion_path OVERDISPERSION_PATH]
                     [--overdispersion_global OVERDISPERSION_GLOBAL]
                     [--as_only] [--qtl_only] [--search_mode SEARCH_MODE]
                     [--max_causal MAX_CAUSAL] [--confidence CONFIDENCE]
                     hap_A_path counts_A_path hap_B_path counts_B_path out_dir

positional arguments:
  hap_A_path            Path to haplotype A genotypes file
  counts_A_path         Path to haplotype A mapped counts file
  hap_B_path            Path to haplotype B genotypes file
  counts_B_path         Path to haplotype B mapped counts file
  out_dir               Path to output directory

optional arguments:
  -h, --help            show this help message and exit
  --total_exp_path TOTAL_EXP_PATH, -t TOTAL_EXP_PATH
                        Path to total QTL phenotype file (Default: Sum of
                        counts files)
  --overdispersion_path OVERDISPERSION_PATH, -o OVERDISPERSION_PATH
                        Path to individual-level AS overdispersion file
                        (Default: Global overdispersion)
  --overdispersion_global OVERDISPERSION_GLOBAL, -g OVERDISPERSION_GLOBAL
                        Global AS overdispersion (Default: 0)
  --as_only, -a         AS-Only Mode
  --qtl_only, -q        QTL-Only Mode
  --search_mode SEARCH_MODE, -s SEARCH_MODE
                        Causal configuration search mode (Default:
                        "exhaustive")
  --max_causal MAX_CAUSAL, -m MAX_CAUSAL
                        Maximum number of causal configurations searched
                        (Default: 1)
  --confidence CONFIDENCE, -c CONFIDENCE
                        Credible set confidence level (Default: 0.95)

```

## PLASMA API

PLASMA additionally has a Python API, which exposes the full feature set of PLASMA. Documentation for the PLASMA Python API is currently in progress. 

Features of the API include:
* Alternative data input formats, including direct use of association statistics
* User specification of hyperparameters, including heritabilities and correlations between the AS and QTL phenotypes
* Additional fine-mapping outputs
* Colocalization analysis across multiple quantitative allele-specific phenotypes
* An allele-specific simulation framework for quantitative phenotypes
* Ability to extend PLASMA via subtyping

To see the latest code, check the `dev` branch.