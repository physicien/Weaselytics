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
#SBATCH --time=00-03:00               # time (DD-HH:MM)
#SBATCH --output=%x_%j.log            # output .log file

#######################################
# Job variables and settings:
datadir=data
storage=$SLURM_TMPDIR
flags="-n -e -p"

#######################################
LAUNCHDIR=$PWD;
DATADIR=$LAUNCHDIR/$datadir
WORKDIR=$storage/$SLURM_JOB_NAME"_"$SLURM_JOB_ID
mkdir -p $WORKDIR
cd $WORKDIR

module load scipy-stack
source /lustre09/project/6004866/$USER/virtualenv/HPLC_ENV/bin/activate

for dir in $DATADIR/*; do
  for filename in $DATADIR/"${dir##*/}"/*.txt; do
    weaselytics "$filename" $flags --output-dir "$WORKDIR"
    printf "\n"
  done
done

mv "$LAUNCHDIR/$SLURM_JOB_NAME"_"$SLURM_JOB_ID.log" "$WORKDIR"
mv "$WORKDIR" "$LAUNCHDIR"
