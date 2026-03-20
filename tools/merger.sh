#!/bin/bash

JOB=$1
WORKDIR=./MERGED_$(basename $JOB)
IMGDIR=images
R2DIR=r2_plots

mkdir -p $WORKDIR
for filename in $JOB/$IMGDIR/*.png; do
  fname=$(b=${filename##*/}; echo ${b%.*})
  convert $JOB/$R2DIR/$fname"_r2".png $filename +append $WORKDIR/$fname.png
done

