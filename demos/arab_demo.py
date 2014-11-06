# This demo illustrates how mtSet can be used for multi and single-trait set tests
# We consider a dataset of 192 samples and 3 flowering time phenotypes in A.thaliana from Atwell et al 2010 (Nature)
# Phenotypes were quantile-normalized to a gaussian distribution beforehand

# Here we consider 3 different models
# - mtSet: multi-trait analysis where relatedness is accounted for as a random effect
# - stSet: single-trait analysis where relatedness is accounted for as a random effect
# - mtSet1VC: multi-trait analysis where relatedness is account for with the first 5 principal components of the kinship as fixed effects

import ipdb
import sys
import limix
sys.path.append('./..')

import pycore.modules.splitter as SPLIT
import pycore.modules.multiTraitSetTest as MTST
import pycore.modules.chi2mixture as C2M
from pycore.utils.utils import smartAppend
from pycore.utils.utils import smartDumpDictHdf5

import scipy as SP
import h5py
import pylab as PL
import copy
import os
import cPickle
import time as TIME

# data and cache files
files = {}
files['data'] = 'data/arab107_preprocessed.hdf5'
files['out_file'] = 'data/results.hdf5'
files['split_cache'] = 'windows_split.hdf5'
files['mtSet_null_cache'] = 'mtSet_null_cache.hdf5'
files['stSet_null_cache'] = 'stSet_null_cache.hdf5'
files['mtSet1VC_null_cache'] = 'mtSet1VC_null_cache.hdf5'

# settings for splitting the genome in different regions and permutations
settings = {}
settings['window_size'] = 1e4
settings['minNumberSnps'] = 4 # considers only windows with at least 4 SNPs

settings['n_windows'] = 10
settings['n_permutations'] = 10

if __name__ == "__main__":

    # N = number of samples
    # P = number of phenotypes
    # V = number of variants
    # K = number of covariates

    # import data
    f  = h5py.File(files['data'],'r')
    phenotype   = f['phenotype'][:]   # phenotype matrix (NxP)
    phenotypeID = f['phenotypeID'][:] # phenotype ids (P-vector)
    genotype    = f['genotype']       # genotype matrix (NxV)
    relatedness = f['relatedness'][:] # relatedness matrix (NxN)
    geno_pos    = f['geno_pos'][:]    # genotype positions (V-vector)
    geno_chrom  = f['geno_chrom'][:]  # genotype choromosomes (V-vector)
    covariates  = f['covariates'][:]  # covariate matrix (NxK)

    # here we consider no covariates for mtSet and stSet
    # while we consider 6 covariates for mtSet1VC
    # (intercept term and first 5 pcs of the relatedness matrix)

    # multi trait set test class
    mtSet    = MTST.MultiTraitSetTest(phenotype,relatedness)
    mtSet1VC = MTST.MultiTraitSetTest(phenotype,F=covariates)

    print '.. fit null models'
    mtSet_null_info = mtSet.fitNull(cache=True,fname=files['mtSet_null_cache'],rewrite=True)
    stSet_null_info = mtSet.fitNullTraitByTrait(cache=True,fname=files['stSet_null_cache'],rewrite=True)
    mtSet_null_info = mtSet1VC.fitNull(cache=True,fname=files['mtSet1VC_null_cache'],rewrite=True)

    print '.. precompute genotype windows'
    split = SPLIT.Splitter(pos=geno_pos,chrom=geno_chrom)
    split.splitGeno(size=settings['window_size'],minSnps=settings['minNumberSnps'],cache=True,fname=files['split_cache'])
    nWindows = split.get_nWindows()

    RV = {}
    print '.. set test scan'
    for window_idx in range(settings['n_windows']):

        print '\t.. window %d'%window_idx

        # consider genetic region
        Iregion, rv_windows = split.getWindow(window_idx)
        region = genotype[:,Iregion]

        # fit models
        rv_mtSet = mtSet.optimize(region)
        rv_stSet = mtSet.optimizeTraitByTrait(region)
        rv_mtSet1VC = mtSet1VC.optimize(region)

        # store LLR (log likelihood ratios) and window positions
        smartAppend(RV,'window_chromosome',rv_windows['chrom'][0])
        smartAppend(RV,'window_start',rv_windows['start'][0])
        smartAppend(RV,'window_end',rv_windows['end'][0])
        smartAppend(RV,'llr_mtSet',rv_mtSet1VC['LLR'][0])
        smartAppend(RV,'llr_stSet',SP.concatenate([rv_stSet[key]['LLR'] for key in rv_stSet.keys()]))
        smartAppend(RV,'llr_mtSet1VC',rv_mtSet1VC['LLR'][0])


    # consider permutations

    for permutation_i in range(settings['n_permutations']):
        print '.. permutation %d' % permutation_i

        # set seed and generate sample permutation
        SP.random.seed(permutation_i)
        permutation = SP.random.permutation(phenotype.shape[0])

        for window_idx in range(settings['n_windows']):

            print '\t.. window %d'%window_idx

            # consider genetic region and permute
            Iregion, rv_windows = split.getWindow(window_idx)
            region = genotype[:,Iregion]
            permuted_region = region[permutation,:]

            # fit models
            rv_mtSet = mtSet.optimize(permuted_region)
            rv_stSet = mtSet.optimizeTraitByTrait(permuted_region)
            rv_mtSet1VC = mtSet1VC.optimize(permuted_region)

            # store permutation LLRs
            smartAppend(RV,'permutation_llr_mtSet',rv_mtSet1VC['LLR'][0])
            smartAppend(RV,'permutation_llr_stSet',SP.concatenate([rv_stSet[key]['LLR'] for key in rv_stSet.keys()]))
            smartAppend(RV,'permutation_llr_mtSet1VC',rv_mtSet1VC['LLR'][0])

    # vectorize outputs
    for key in RV.keys():   RV[key] = SP.array(RV[key])
       
    print '.. calculate p-values'
    print '(for accurate estimate of pvalues either the number of windows or the number of permutations should be increased)'

    c2m = C2M.Chi2mixture(tol=4e-3)

    # obtain p-values for mtSet
    c2m.estimate_chi2mixture(RV['permutation_llr_mtSet'])
    RV['pv_mtSet'] = c2m.sf(RV['llr_mtSet'])
    RV['permutation_pv_mtSet'] = c2m.sf(RV['permutation_llr_mtSet'])

    # obtain p-values for stSet
    RV['pv_stSet'] = SP.zeros_like(RV['llr_stSet'])
    RV['permutation_pv_stSet'] = SP.zeros_like(RV['permutation_llr_stSet'])
    for p in range(phenotype.shape[1]):
        c2m.estimate_chi2mixture(RV['permutation_llr_stSet'][:,p])
        RV['pv_stSet'][:,p] = c2m.sf(RV['llr_stSet'][:,p])
        RV['permutation_pv_stSet'][:,p] = c2m.sf(RV['permutation_llr_stSet'][:,p])

    # obtain p-values for mtSet1VC
    c2m.estimate_chi2mixture(RV['permutation_llr_mtSet1VC'])
    RV['pv_mtSet1VC'] = c2m.sf(RV['llr_mtSet1VC'])
    RV['permutation_pv_mtSet1VC'] = c2m.sf(RV['permutation_llr_mtSet1VC'])

    print '.. export results in %s'%files['out_file']
    fout = h5py.File(files['out_file'],'w')
    smartDumpDictHdf5(RV,fout)
    fout.close()

