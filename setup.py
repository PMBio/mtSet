try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import distutils.cmd
from setuptools import find_packages
import sys,os,re

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def strip_rc(version):
    return re.sub(r"rc\d+$", "", version)

def check_versions(min_versions):
    """
    Check versions of dependency packages
    """
    from distutils.version import StrictVersion

    try:
        import scipy
        spversion = scipy.__version__
    except ImportError:
        raise ImportError("mtSet requires scipy")

    try:
        import numpy
        npversion = numpy.__version__
    except ImportError:
        raise ImportError("mtSet requires numpy")
 
    try:
        import pandas
        pandasversion = pandas.__version__
    except ImportError:
        raise ImportError("mtSet requires pandas")

    #match version numbers
    try:
        assert StrictVersion(strip_rc(npversion)) >= min_versions['numpy']
    except AssertionError:
        raise ImportError("Numpy version is %s. Requires >= %s" %
                (npversion, min_versions['numpy']))
    try:
        assert StrictVersion(strip_rc(spversion)) >= min_versions['scipy']
    except AssertionError:
        raise ImportError("Scipy version is %s. Requires >= %s" %
                (spversion, min_versions['scipy']))
    try:
        assert StrictVersion(strip_rc(pandasversion)) >= min_versions['pandas']
    except AssertionError:
        raise ImportError("pandas version is %s. Requires >= %s" %
                (pandasversion, min_versions['pandas']))

if __name__ == '__main__':
    min_versions = {
        'numpy' : '1.6.0',
        'scipy' : '0.9.0',
        'pandas' : '0.12.0',
        'scons' : '2.1.0',
                   }
    check_versions(min_versions)

    setup(
        name = 'mtSet',
        version = '0.1',
        author = 'Paolo Casale, Barbara Rakitsch, Oliver Stegle',
        author_email = "casale@ebi.ac.u, rakitsch@ebi.ac.uk, stegle@ebi.ac.uk",
        description = ('Efficient linear mixd model set tests'),
        url = "https://github.com/PMBio/mtSet",
        long_description = read('README.md'),
        license = 'BSD',
        keywords = 'linear mixed models, GWAS, QTL',
        scripts = ['mtSet/bin/mtSet_analyze'],
        #packages = ['mtSet'],
        packages = find_packages(),
        install_requires=['scipy>=0.13', 'numpy>=1.6', 'matplotlib>=1.2', 'nose', 'pandas'],
        )
