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

def readCovarianceMatrixFile(cfile):
    """"
    reading in similarity matrix

    cfile   File containing the covariance matrix. The corresponding ID file must be specified in cfile.id)
    """
    covFile = cfile+'.cov'
    idFile  = cfile+'.cov.id'

    assert os.path.exists(covFile), '%s is missing.'%covFile
    assert os.path.exists(idFile), '%s is missing.'%idFile

    K   = SP.loadtxt(covFile)
    ids = SP.loadtxt(idFile,dtype=str)
    assert K.shape[0]==K.shape[1], 'dimension mismatch'
    assert ids.shape[0]==K.shape[0], 'dimension mismatch'
    assert SP.all(ids[:,0]==ids[:,1]), 'ids are not symmetric in %s.id'%cfile

    return K,ids


def readCovariatesFile(fFile):
    """"
    reading in covariate file

    cfile   file containing the fixed effects as NxP matrix
            (N=number of samples, P=number of covariates)
    """
    assert os.path.exists(fFile), '%s is missing.'%fFile
    F = SP.loadtxt(fFile)
    return F


def readPhenoFile(pfile,idx=None):
    """"
    reading in phenotype file

    pfile   root of the file containing the phenotypes as NxP matrix
            (N=number of samples, P=number of traits)
    """

    usecols = None
    if idx!=None:
        """ different traits are comma-seperated """
        usecols = [int(x) for x in idx.split(',')]
        
    phenoFile = pfile+'.phe'
    assert os.path.exists(phenoFile), '%s is missing.'%phenoFile

    Y = SP.loadtxt(phenoFile,usecols=usecols)
    
    if (usecols is not None) and (len(usecols)==1): Y = Y[:,SP.newaxis]
    return Y

def readNullModelFile(nfile):
    """"
    reading file with null model info

    nfile   File containing null model info
    """
    nullmod_file = nfile+'.p0'
    assert os.path.exists(nullmod_file), '%s is missing.'%nullmod_file
    params = SP.loadtxt(nullmod_file)

    rv = {'params0_g':params[0],'params0_n':params[1]}
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

