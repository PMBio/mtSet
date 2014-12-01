#!/bin/bash 
wget http://www.ebi.ac.uk/~casale/1000G_chr22.zip
mkdir ./../data
unzip 1000G_chr22.zip -d ./../data
rm 1000G_chr22.zip
