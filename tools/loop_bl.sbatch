#!/bin/bash

#SBATCH --job-name=SWEEP
#SBATCH --account=rrg-cotemich-ac
#SBATCH --mail-type=END,FAIL          # Mail events (NONE,BEGIN,END,FAIL,ALL)
#SBATCH --mail-user=youremail@mail.com
#SBATCH --ntasks=1                    # cpus, if across several nodes
#SBATCH --cpus-per-task=8             # workers for the r2 sweep (-w)
#SBATCH --mem-per-cpu=256M            # memory, if across several nodes
##SBATCH --nodes=2                     # nb of whole nodes
##SBATCH --ntasks-per-node=12          # cpus, if whole node
##SBATCH --mem=24G                     # memory, mem=0 for all memory of the node
#SBATCH --time=00-03:00               # time (DD-HH:MM)
#SBATCH --output=%x_%j.log            # output .log file

# Submit from the repository root (the directory containing `data`):
#     sbatch -J PROD_2026-10-19 tools/loop_bl.sbatch
#     sbatch -J SYNTH_ERB_2026-10-23 --export=ALL,MODE=synth,\
# DATASET=$PWD/../SYNTH_ERB_2026-10-23 tools/loop_bl.sbatch

#######################################
# Job variables and settings:
mode=${MODE:-prod}
datadir=data
storage=$SLURM_TMPDIR
flags="-n -e -p"
cachedir=r2_cache

#######################################
LAUNCHDIR=$PWD;
DATADIR=$LAUNCHDIR/$datadir
WORKDIR=$storage/$SLURM_JOB_NAME"_"$SLURM_JOB_ID
mkdir -p $WORKDIR
cd $WORKDIR

module load scipy-stack
source /lustre09/project/6004866/$USER/virtualenv/HPLC_ENV/bin/activate

# Run the branch checked out in the launch directory
export PYTHONPATH="$LAUNCHDIR${PYTHONPATH:+:$PYTHONPATH}"

# One BLAS thread per worker process
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -c "import numpy, scipy, pybaselines as p
from pybaselines._compat import _HAS_NUMBA
print('numpy', numpy.__version__, 'scipy', scipy.__version__,
      'pybaselines', p.__version__, 'numba', _HAS_NUMBA)"

if [ "$mode" = synth ]; then
  python $LAUNCHDIR/tools/synthetic/synth_diag.py $DATASET \
      --cache-dir $WORKDIR/$cachedir --diag-dir $WORKDIR/diag \
      --workers ${SLURM_CPUS_PER_TASK:-1}
  python $LAUNCHDIR/tools/synthetic/synth_figs.py $DATASET \
      --cache-dir $WORKDIR/$cachedir --diag-dir $WORKDIR/diag \
      --out-dir $WORKDIR --workers ${SLURM_CPUS_PER_TASK:-1}
else
  for filename in $DATADIR/*/*.txt; do
    python -m weaselytics $filename $flags \
        -od $WORKDIR -cd $WORKDIR/$cachedir \
        -w ${SLURM_CPUS_PER_TASK:-1}
    printf "\n"
  done
fi

mv "$LAUNCHDIR/$SLURM_JOB_NAME"_"$SLURM_JOB_ID.log" "$WORKDIR"
mv "$WORKDIR" "$LAUNCHDIR/runs"
