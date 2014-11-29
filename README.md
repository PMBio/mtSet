mtSet
======

Set tests are a powerful approach for genome-wide association testing between groups of genetic variants and individual quantitative traits.
mtSet is an implementation of efficient set test algorithms for joint analysis across multiple traits. mtSet can account for confounding factors such as relatedness and can be used for analysis of single traits.

The implementation is in Python and builds on the Gaussian process toolbox in pygp (https://github.com/PMBio/pygp) and the linear mixed model library LIMIX (https://pypi.python.org/pypi/limix).

By Francesco Paolo Casale (casale@ebi.ac.uk), Barbara Rakitsch (rakitsch@ebi.ac.uk) and Oliver Stegle (stegle@ebi.ac.uk)

## Requirements

#### Limix
mtSet requires [LIMIX](https://pypi.python.org/pypi/limix). The easiest way to install LIMIX is

    pip install limix
  
For a more detailed installation guide, we refer to the [LIMIX](https://pypi.python.org/pypi/limix) homepage.

#### Plink
mtSet also requires [Plink 1.9](https://www.cog-genomics.org/plink2), which is used to calculate the genetic relatedness matrix from bed genotype files.

## Installation

The repository can be cloned using git

    git clone https://github.com/PMBio/mtSet.git
    
mtSet can be run from the command line using the scripts in _mtSet_/_bin_ in a portable manner as shown in the [tutorial](https://github.com/PMBio/mtSet/wiki/Tutorial).

mtSet can be also installed by using the _setup.py_ file.

## How to use mtSet

Please find instructions and examples in the [tutorial](https://github.com/PMBio/mtSet/wiki/Tutorial).

## License
See [LICENSE] https://github.com/PMBio/mtSet/blob/master/LICENSE
