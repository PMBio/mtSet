#!/bin/bash 
BFILE=./../data/1000G_chr22/chrom22 #specify here bed basename
CFILE=./out_stSet/chrom22
PFILE=./out_stSet/pheno
WFILE=./out_stSet/windows
NFILE=./out_stSet/null
WSIZE=10000
RESDIR=./out_stSet/results
OUTFILE=./out_stSet/final

# Preprocessing and generation
./../mtSet/bin/mtSet_preprocess --compute_covariance --bfile $BFILE --cfile $CFILE 
./../mtSet/bin/mtSet_simPheno --bfile $BFILE --cfile $CFILE --pfile $PFILE --chrom 22
./../mtSet/bin/mtSet_preprocess --precompute_windows --fit_null --bfile $BFILE --cfile $CFILE --pfile $PFILE --wfile $WFILE --nfile $NFILE --window_size $WSIZE --plot_windows --trait_idx 0 

# Analysis
# test
./../mtSet/bin/mtSet_analyze --bfile $BFILE --cfile $CFILE --pfile $PFILE --nfile $NFILE --wfile $WFILE --minSnps 4 --resdir $RESDIR --start_wnd 0 --end_wnd 5 --trait_idx 0 
./../mtSet/bin/mtSet_analyze --bfile $BFILE --cfile $CFILE --pfile $PFILE --nfile $NFILE --wfile $WFILE --minSnps 4 --resdir $RESDIR --start_wnd 5 --end_wnd 10 --trait_idx 0 
#permutations
for i in `seq 0 1`;
do
./../mtSet/bin/mtSet_analyze --bfile $BFILE --cfile $CFILE --pfile $PFILE --nfile $NFILE --wfile $WFILE --minSnps 4 --resdir $RESDIR --start_wnd 0 --end_wnd 5 --perm $i --trait_idx 0 
./../mtSet/bin/mtSet_analyze --bfile $BFILE --cfile $CFILE --pfile $PFILE --nfile $NFILE --wfile $WFILE --minSnps 4 --resdir $RESDIR --start_wnd 5 --end_wnd 10 --perm $i --trait_idx 0
done

#postprocess
./../mtSet/bin/mtSet_postprocess --resdir $RESDIR --outfile $OUTFILE 
