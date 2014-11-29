mtSet
======

Set tests are a powerful approach for genome-wide association testing between groups of genetic variants and individual quantitative traits.
mtSet is an implementation of efficient set test algorithms for joint analysis across multiple traits. mtSet can account for confounding factors such as relatedness and can be used for analysis of single traits.

The implementation is in Python and builds on the Gaussian process toolbox in pygp (https://github.com/PMBio/pygp) and the linear mixed model library LIMIX (https://pypi.python.org/pypi/limix).

By Francesco Paolo Casale (casale@ebi.ac.uk), Barbara Rakitsch (rakitsch@ebi.ac.uk) and Oliver Stegle (stegle@ebi.ac.uk)

## Requirements

mtSet requires LIMIX (https://pypi.python.org/pypi/limix)

## How to use mtSet?

After cloning the repository, mtSet can be run from the command line using the scripts in bin as shown in the tutorial. A demo on how to use mtSet in python can be found in demos.

Extensive documentation will be available soon.

## License
See [LICENSE] https://github.com/PMBio/mtSet/blob/master/LICENSE
