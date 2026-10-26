#!/bin/bash
# Fit one molecule's peak in every baseline-corrected chromatogram of a
# directory, writing <stem>_<MOL>.csv for each.
#
# Usage:  tools/loop_extract.sh DIR X0 X1 MOL [PATTERN]
#   DIR      directory holding the <stem>_bl.txt files
#   X0, X1   fitting window, in minutes
#   MOL      molecule name, used in the output file name
#   PATTERN  glob on the file name, without .txt (default: *)

[ $# -ge 4 ] || {
  echo "usage: $0 DIR X0 X1 MOL [PATTERN]" >&2
  exit 1
}

DATADIR=$1
x0=$2
x1=$3
os=$4
pattern=${5:-*}

for filename in $DATADIR/$pattern.txt; do
  python -m weaselytics $filename -s -nb -x0 $x0 -x1 $x1 -os $os
done
