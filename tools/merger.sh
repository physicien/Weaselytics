#!/bin/bash

JOB=./noisy_start_8160126
WORKDIR=./MERGED_${JOB##*/}
IMGDIR=images
R2DIR=r2_plots

mkdir -p $WORKDIR
for filename in $JOB/$IMGDIR/*.png; do
  fname=$(b=${filename##*/}; echo ${b%.*})
  convert $JOB/$R2DIR/$fname"_r2".png $filename +append $WORKDIR/$fname.png
done

