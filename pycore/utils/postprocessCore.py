import sys
sys.path.append('./../..')
import limix
import pycore.modules.chi2mixture as C2M
import os
import subprocess
import pdb
import sys
import csv
import glob
import numpy as NP
from optparse import OptionParser
import time

def postprocess(options):
    """ perform parametric fit of the test statistics and provide permutation and test pvalues """

    resdir = options.resdir
    out_file = options.outfile
    tol = options.tol

    print '.. load permutation results'
    file_name = os.path.join(resdir,'perm*','*.res')
    files = glob.glob(file_name)
    LLR0 = []
    for _file in files:
        print _file
        LLR0.append(NP.loadtxt(_file,usecols=[6]))
    LLR0 = NP.concatenate(LLR0)

    print '.. fit test statistics'
    t0 = time.time()
    c2m = C2M.Chi2mixture(tol=4e-3)
    c2m.estimate_chi2mixture(LLR0)
    pv0 = c2m.sf(LLR0)
    t1 = time.time()
    print 'finished in %s seconds'%(t1-t0)

    print '.. export permutation results'
    perm_file = out_file+'.perm'
    RV = NP.array([LLR0,pv0]).T
    NP.savetxt(perm_file,RV)

    print '.. load test results'
    file_name = os.path.join(resdir,'test','*.res')
    files = glob.glob(file_name)
    RV_test = []
    for _file in files:
        print _file
        RV_test.append(NP.loadtxt(_file))
    RV_test = NP.concatenate(RV_test)

    print '.. calc pvalues'
    pv = c2m.sf(RV_test[:,-1])[:,NP.newaxis]

    print '.. export test results'
    perm_file = out_file+'.test'
    RV_test = NP.hstack([RV_test,pv])
    NP.savetxt(perm_file,RV_test)
