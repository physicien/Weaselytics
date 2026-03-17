#!/bin/bash

#SBATCH --job-name=bl_corrected_data
#SBATCH --account=rrg-cotemich-ac
#SBATCH --mail-type=END,FAIL          # Mail events (NONE,BEGIN,END,FAIL,ALL)
#SBATCH --mail-user=youremail@mail.com
#SBATCH --ntasks=1                    # cpus, if across several nodes
#SBATCH --mem-per-cpu=1G              # memory, if across several nodes
##SBATCH --nodes=2                     # nb of whole nodes
##SBATCH --ntasks-per-node=12          # cpus, if whole node
##SBATCH --mem=24G                     # memory, mem=0 for all memory of the node
#SBATCH --time=00-03:00		      # time (DD-HH:MM)
#SBATCH --output=%x_%j.log 	      # output .log file

#######################################
# Job variables and settings:
scriptname=weaselytics.py
scriptdir="${scriptname%.*}"
datadir=data
storage=$SLURM_TMPDIR
flags="-n -e -p"

#######################################
# Files names
BASENAME=$SLURM_JOB_NAME

#######################################
#Create a local working directory for the ORCA calculation
LAUNCHDIR=$PWD;
DATADIR=$LAUNCHDIR/$datadir
WORKDIR=$storage/$BASENAME"_"$SLURM_JOB_ID
mkdir -p $WORKDIR
mkdir -p $WORKDIR/images
mkdir -p $WORKDIR/r2_plots

#Copy the input file and the script to the local working directory
cp $LAUNCHDIR/$scriptdir/*.py $WORKDIR/
cp "$(readlink -f $0)" "$WORKDIR"
cd $WORKDIR

module load scipy-stack
source /lustre09/project/6004866/$USER/virtualenv/HPLC_ENV/bin/activate

for dir in $DATADIR/*; do
  name=${dir##*/}
  mkdir -p $WORKDIR/$name
    for filename in $DATADIR/$name/*.txt; do
      python $scriptname $filename $flags
      fname=$(b=${filename##*/}; echo ${b%.*})
      mv $WORKDIR/$fname"_bl.txt" $WORKDIR/$name/$fname"_bl.txt"
      printf "\n"
    done
done

#Move the files to the local working directory 
mv $LAUNCHDIR/$BASENAME"_"$SLURM_JOB_ID.log $WORKDIR
mv $WORKDIR $LAUNCHDIR
