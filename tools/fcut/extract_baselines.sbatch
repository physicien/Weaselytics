#!/bin/bash

#SBATCH --job-name=EXTRACT_BL
#SBATCH --account=rrg-cotemich-ac
##SBATCH --ntasks=1                    # cpus, if across several nodes
##SBATCH --cpus-per-task=16            # workers, if across several nodes
##SBATCH --mem-per-cpu=500M            # memory, if across several nodes
#SBATCH --nodes=1                     # nb of whole nodes
#SBATCH --ntasks-per-node=16          # cpus, if whole node
#SBATCH --mem=8G                      # memory, mem=0 for all of the node
#SBATCH --time=00-03:00		      # time (DD-HH:MM)
#SBATCH --output=%x_%j.log 	      # output .log file

# Stores the baseline the production selection fits to every real
# chromatogram, for comparison against the analytic baselines the
# synthetic benchmark uses.
#
# Follows tools/slurm_script: work in $SLURM_TMPDIR, run the branch
# checked out in the launch directory, move the results and the log back
# at the end.
#
# The r2 cache travels to node-local disk with the data. A warm cache
# makes this minutes rather than hours, since the autocorrelation sweep
# is the whole cost; it is COPIED because a miss deletes the entries of
# the same stem before writing.
#
# Submit from the repository root:
#     DATA=/scratch/$USER/data CACHE=/scratch/$USER/r2_cache sbatch \
#         tools/fcut/extract_baselines.sbatch

#######################################
# Job variables and settings:
datadir=${DATA:?set DATA to the directory holding Molecule/stem.txt}
cachedir=${CACHE:-}
storage=$SLURM_TMPDIR

#######################################
# Files names
BASENAME=$SLURM_JOB_NAME

#######################################
# Create a local working directory
LAUNCHDIR=$PWD;
WORKDIR=$storage/$BASENAME"_"$SLURM_JOB_ID
mkdir -p $WORKDIR

# Copy this script to the working directory for provenance
cp "$(readlink -f $0)" "$WORKDIR"

# Data and cache travel to node-local disk
cp -r "$datadir" "$WORKDIR/data"
cache_arg=""
if [ -n "$cachedir" ]; then
  cp -r "$cachedir" "$WORKDIR/r2_cache"
  cache_arg="--cache $WORKDIR/r2_cache"
fi
cd $WORKDIR

module load scipy-stack
source /lustre09/project/6004866/$USER/virtualenv/HPLC_ENV/bin/activate

# Run the branch checked out in the launch directory
export PYTHONPATH="$LAUNCHDIR${PYTHONPATH:+:$PYTHONPATH}"

# One BLAS thread per worker process: the parallelism is over signals,
# so threaded BLAS would only oversubscribe.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python $LAUNCHDIR/tools/fcut/extract_baselines.py \
    "$WORKDIR/data" "$WORKDIR/out" $cache_arg \
    --workers ${SLURM_NTASKS_PER_NODE:-${SLURM_CPUS_PER_TASK:-1}}

#Move the files to the local working directory
mv $LAUNCHDIR/$BASENAME"_"$SLURM_JOB_ID.log $WORKDIR
mv $WORKDIR $LAUNCHDIR
