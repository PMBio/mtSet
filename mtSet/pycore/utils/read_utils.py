import scipy as SP
import os

def readBimFile(basefilename):
    """
    Helper fuinction that reads bim files
    """
    # read bim file
    bim_fn = basefilename+'.bim'
    rv = SP.loadtxt(bim_fn,delimiter='\t',usecols = (0,3),dtype=int)
    return rv

def readCovarianceMatrixFile(cfile,readCov=True,readEig=True):
    """"
    reading in similarity matrix

    cfile   File containing the covariance matrix. The corresponding ID file must be specified in cfile.id)
    """
    covFile = cfile+'.cov'
    evalFile = cfile+'.cov.eval'
    evecFile = cfile+'.cov.evec'

    RV = {}
    if readCov:
        assert os.path.exists(covFile), '%s is missing.'%covFile
        RV['K'] = SP.loadtxt(covFile)
    if readEig:
        assert os.path.exists(evalFile), '%s is missing.'%evalFile
        assert os.path.exists(evecFile), '%s is missing.'%evecFile
        RV['eval'] = SP.loadtxt(evalFile)
        RV['evec'] = SP.loadtxt(evecFile)

    return RV

def readPhenoFile(pfile):
    """"
    reading in phenotype file

    pfile   root of the file containing the phenotypes as NxP matrix
            (N=number of samples, P=number of traits)
    """
    phenoFile = pfile+'.phe'
    assert os.path.exists(phenoFile), '%s is missing.'%phenoFile
    Y = SP.loadtxt(phenoFile)
    return Y

def readNullModelFile(nfile):
    """"
    reading file with null model info

    nfile   File containing null model info
    """
    nullmod_file = nfile+'.p0'
    assert os.path.exists(nullmod_file), '%s is missing.'%nullmod_file
    params = SP.loadtxt(nullmod_file)
    rv = {'params0_g':params[0,:],'params0_n':params[1,:]}
    return rv

def readWindowsFile(wfile):
    """"
    reading file with windows

    wfile   File containing window info
    """
    window_file = wfile+'.wnd'
    assert os.path.exists(window_file), '%s is missing.'%window_file
    rv = SP.loadtxt(window_file)
    return rv

