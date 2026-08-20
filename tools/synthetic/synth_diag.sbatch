#!/bin/bash

#SBATCH --job-name=SYNTH_DIAG
#SBATCH --account=rrg-cotemich-ac
##SBATCH --ntasks=1                    # cpus, if across several nodes
##SBATCH --cpus-per-task=16            # workers, if across several nodes
##SBATCH --mem-per-cpu=500M            # memory, if across several nodes
#SBATCH --nodes=1                     # nb of whole nodes
#SBATCH --ntasks-per-node=16          # cpus, if whole node
#SBATCH --mem=8G                      # memory, mem=0 for all of the node
#SBATCH --time=00-03:00		      # time (DD-HH:MM)
#SBATCH --output=%x_%j.log 	      # output .log file

# Scores the synthetic benchmark: tools/synthetic/synth_diag.py over
# every signal of a dataset built by synth_dataset.py.
#
# Follows tools/slurm_script: work in $SLURM_TMPDIR, run the branch
# checked out in the launch directory, move the results and the log back
# at the end. Differences with that script:
#   - one process with a worker pool over signals, rather than workers
#     inside a single sweep. The task count does not confine the pool:
#     Slurm's cgroup grants the job every allocated cpu on the node, and
#     `--mem` pools the memory across them, which suits a pool whose
#     per-signal footprint varies fivefold;
#   - the dataset is copied to node-local disk, because the run writes
#     one npz per signal and the shared filesystem is poor at that;
#   - the r2 cache and the stored error curves travel with it, so a
#     re-run pays for neither the autocorrelation sweep nor the 250
#     BEADS fits of the error curve;
#   - the dataset is a parameter rather than a hardcoded `data`.
#
# Submit from the repository root:
#     DATASET=/scratch/$USER/SYNTH_ERB_2026-08-18 sbatch \
#         tools/synthetic/synth_diag.sbatch

#######################################
# Job variables and settings:
dataset=${DATASET:?set DATASET to the synthetic dataset directory}
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

# The dataset travels to node-local disk; the results come back below
cp -r "$dataset" "$WORKDIR/dataset"
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

python $LAUNCHDIR/tools/synthetic/synth_diag.py "$WORKDIR/dataset" \
    --workers ${SLURM_NTASKS_PER_NODE:-${SLURM_CPUS_PER_TASK:-1}}

#Move the files to the local working directory
mv $LAUNCHDIR/$BASENAME"_"$SLURM_JOB_ID.log $WORKDIR
mv $WORKDIR $LAUNCHDIR
