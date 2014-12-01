#!/bin/bash 
BFILE=./../data/1000G_chr22/chrom22 #specify here bed basename
CFILE=./out_mtSet1VC/chrom22
PFILE=./out_mtSet1VC/pheno
WFILE=./out_mtSet1VC/windows
NFILE=./out_mtSet1VC/null
WSIZE=10000
RESDIR=./out_mtSet1VC/results
OUTFILE=./out_mtSet1VC/final

# Preprocessing and generation
./../mtSet/bin/mtSet_preprocess --compute_covariance --bfile $BFILE --cfile $CFILE 
./../mtSet/bin/mtSet_simPheno --bfile $BFILE --cfile $CFILE --pfile $PFILE --chrom 22
./../mtSet/bin/mtSet_preprocess --precompute_windows --fit_null --bfile $BFILE --pfile $PFILE --wfile $WFILE --nfile $NFILE --window_size $WSIZE --plot_windows 

# Analysis
# test
./../mtSet/bin/mtSet_analyze --bfile $BFILE --pfile $PFILE --nfile $NFILE --wfile $WFILE --minSnps 4 --resdir $RESDIR --start_wnd 0 --end_wnd 5 
./../mtSet/bin/mtSet_analyze --bfile $BFILE --pfile $PFILE --nfile $NFILE --wfile $WFILE --minSnps 4 --resdir $RESDIR --start_wnd 5 --end_wnd 10 
#permutations
for i in `seq 0 1`;
do
./../mtSet/bin/mtSet_analyze --bfile $BFILE --pfile $PFILE --nfile $NFILE --wfile $WFILE --minSnps 4 --resdir $RESDIR --start_wnd 0 --end_wnd 5 --perm $i 
./../mtSet/bin/mtSet_analyze --bfile $BFILE --pfile $PFILE --nfile $NFILE --wfile $WFILE --minSnps 4 --resdir $RESDIR --start_wnd 5 --end_wnd 10 --perm $i
done

#postprocess
./../mtSet/bin/mtSet_postprocess --resdir $RESDIR --outfile $OUTFILE 
