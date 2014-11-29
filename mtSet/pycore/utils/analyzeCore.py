import sys
sys.path.append('./../../..')
import limix
import os
import subprocess
import pdb
import sys
import csv
import numpy as NP
from optparse import OptionParser
import time
import mtSet.pycore.modules.multiTraitSetTest as MTST
from mtSet.pycore.utils.read_utils import readNullModelFile
from mtSet.pycore.utils.read_utils import readWindowsFile
from mtSet.pycore.utils.read_utils import readCovarianceMatrixFile
from mtSet.pycore.utils.read_utils import readPhenoFile

# fastLMM
sys.path.append('./../plink')
import mtSet.plink.plink_reader
 
def scan(bfile,Y,K,params0,wnds,minSnps,i0,i1,perm_i,outfile):

    if perm_i is not None:
        print 'Generating permutation (permutation %d)'%perm_i
        NP.random.seed(perm_i)
        perm = NP.random.permutation(Y.shape[0])

    mtSet = MTST.MultiTraitSetTest(Y,K)
    #mtSet.setNull(null)
    #bed = Bed(bfile,standardizeSNPs=False)

    bim = plink_reader.readBIM(bfile)
    fam = plink_reader.readFAM(bfile)
   
    wnd_file = csv.writer(open(outfile,'wb'),delimiter='\t')
    for wnd_i in range(i0,i1):
        print '.. window %d - (%d, %d-%d) - %d snps'%(wnd_i,int(wnds[wnd_i,1]),int(wnds[wnd_i,2]),int(wnds[wnd_i,3]),int(wnds[wnd_i,-1]))
        if int(wnds[wnd_i,-1])<minSnps:
            print 'SKIPPED: number of snps lower than minSnps'
            continue
        #RV = bed.read(PositionRange(int(wnds[wnd_i,-2]),int(wnds[wnd_i,-1])))
        RV = plink_reader.readBED(bfile, blocksize = 1, start = int(wnds[wnd_i,4]), nSNPs = int(wnds[wnd_i,5]), order  = 'F',standardizeSNPs=False,ipos = 2,bim=bim,fam=fam)
        
        Xr = abs(RV['snps']-2)
        if perm_i is not None:
            Xr = Xr[perm,:]
        rv = mtSet.optimize(Xr)
        line = NP.concatenate([wnds[wnd_i,:],rv['LLR']])
        wnd_file.writerow(line)
    pass

def analyze(options):

    # load data
    print 'import data'
    K,ids = readCovarianceMatrixFile(options.cfile)
    Y = readPhenoFile(options.pfile)
    null = readNullModelFile(options.nfile)
    wnds = readWindowsFile(options.wfile)

    if options.i0 is None: options.i0 = 1
    if options.i1 is None: options.i1 = wnds.shape[0]

    # name of output file
    if options.perm_i is not None:
        out_dir = os.path.join(options.outdir,'perm%d'%options.perm_i)
    else:
        out_dir = os.path.join(options.outdir,'test')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    n_digits = len(str(wnds.shape[0]))
    fname = str(options.i0).zfill(n_digits)
    fname+= '_'+str(options.i1).zfill(n_digits)+'.res'
    outfile = os.path.join(out_dir,fname)

    # analysis
    print 'fitting model'
    t0 = time.time()
    scan(options.bfile,Y,K,null,wnds,options.minSnps,options.i0,options.i1,options.perm_i,outfile)
    t1 = time.time()
    print '... finished in %s seconds'%(t1-t0)

