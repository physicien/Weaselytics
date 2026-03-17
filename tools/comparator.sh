#!/bin/bash

JOB1=./REF_2026-03-12_8072682
JOB2=./noisy_start_8160126
WORKDIR=./${JOB1##*/}__vs__${JOB2##*/}
IMGDIR=images
R2DIR=r2_plots

mkdir -p $WORKDIR/$IMGDIR
for filename in $JOB1/$IMGDIR/*.png; do
  fname=$(b=${filename##*/}; echo ${b%.*})
  convert $filename $JOB2/$IMGDIR/$fname.png +append $WORKDIR/$IMGDIR/$fname.png
done

mkdir -p $WORKDIR/$R2DIR
for filename in $JOB1/$R2DIR/*.png; do
  fname=$(b=${filename##*/}; echo ${b%.*})
  convert $filename $JOB2/$R2DIR/$fname.png +append $WORKDIR/$R2DIR/$fname.png
done
