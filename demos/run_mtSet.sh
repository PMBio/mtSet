#!/bin/bash 
BFILE=./../data/1000G_chr22/chrom22 #specify here bed basename
CFILE=./out/chrom22
PFILE=./out/pheno
WFILE=./out/windows
NFILE=./out/null
WSIZE=10000
RESDIR=./out/results
OUTFILE=./out/final

# Preprocessing and generation
./../mtSet/bin/mtSet_preprocess --compute_covariance --bfile $BFILE --cfile $CFILE 
./../mtSet/bin/mtSet_simPheno --bfile $BFILE --cfile $CFILE --pfile $PFILE --chrom 22
./../mtSet/bin/mtSet_preprocess --precompute_windows --fit_null --bfile $BFILE --cfile $CFILE --pfile $PFILE --wfile $WFILE --nfile $NFILE --window_size $WSIZE --plot_windows 

# Analysis
# test
./../mtSet/bin/mtSet_analyze --bfile $BFILE --cfile $CFILE --pfile $PFILE --nfile $NFILE --wfile $WFILE --minSnps 4 --resdir $RESDIR --start_wnd 0 --end_wnd 5
./../mtSet/bin/mtSet_analyze --bfile $BFILE --cfile $CFILE --pfile $PFILE --nfile $NFILE --wfile $WFILE --minSnps 4 --resdir $RESDIR --start_wnd 5 --end_wnd 10
#permutations
for i in `seq 0 1`;
do
./../mtSet/bin/mtSet_analyze --bfile $BFILE --cfile $CFILE --pfile $PFILE --nfile $NFILE --wfile $WFILE --minSnps 4 --resdir $RESDIR --start_wnd 0 --end_wnd 5 --perm $i
./../mtSet/bin/mtSet_analyze --bfile $BFILE --cfile $CFILE --pfile $PFILE --nfile $NFILE --wfile $WFILE --minSnps 4 --resdir $RESDIR --start_wnd 5 --end_wnd 10 --perm $i
done

#postprocess
./../mtSet/bin/mtSet_postprocess --resdir $OUTDIR --outfile $OUTFILE 
