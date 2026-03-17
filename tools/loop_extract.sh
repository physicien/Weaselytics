#!/bin/bash

LAUNCHDIR=$PWD;
DATADIR=$LAUNCHDIR/baseline_corrected_data_46770061/dead_time
SOLVDIR=2-Xylene__LPYE__n-Pentane__1-10
pattern=*
x0=4.00
x1=4.50
os=n-Pentane

for filename in $DATADIR/$SOLVDIR/$pattern.txt; do
  python hplc_extract.py $filename -s -nb -x0 $x0 -x1 $x1 -os $os
done
