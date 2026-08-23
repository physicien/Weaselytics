#!/bin/bash
# Refill the r2 caches with the Durbin-Watson statistic.
#
# The sweep now stores dw_val alongside r2_val, and _R2_CHANNEL is bumped
# to "y-baseline-dw", so every existing entry misses and the BEADS sweep
# has to be paid once more. That sweep is 99.9% of the compute, hence the
# job array: each task takes an interleaved shard of the signals.
#
#   sbatch --export=ALL,MODE=prod  tools/fcut/regen_dw.sbatch
#   sbatch --export=ALL,MODE=synth,DATASET=/path/to/SYNTH_ERB_2026-08-18 \
#          tools/fcut/regen_dw.sbatch
#
# Submit from the checkout. On Rorqual that is
#   ~/links/projects/rrg-cotemich-ac/$USER/Git/Weaselytics
# and `data` sits inside it. On a workstation `data` may instead be a
# sibling of the checkout, so both are looked for rather than assumed;
# set DATADIR to override.
#
# DATASET has no default. The synthetic set is not beside the code on
# every machine and guessing its path has already cost one stray output
# directory, so MODE=synth refuses to start without it.
#
# The caches are written to a NEW directory, never on top of an existing
# one: a cache miss deletes every entry of the same stem before writing,
# so pointing this at runs/PROD_2026-08-20/r2_cache would destroy the
# curves it is replacing before knowing the new ones are good.

#SBATCH --job-name=REGEN_DW
#SBATCH --account=rrg-cotemich-ac
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=manubou5@gmail.com
#SBATCH --array=0-11
##SBATCH --ntasks=1                   # cpus, if across several nodes
##SBATCH --mem-per-cpu=2G             # memory, if across several nodes
#SBATCH --nodes=1                     # nb of whole nodes
#SBATCH --ntasks-per-node=1           # one task; the sweep forks its own
#SBATCH --cpus-per-task=16            # workers for the r2 sweep (-w)
#SBATCH --mem=32G                     # mem=0 for all memory of the node
#SBATCH --time=00-06:00               # time (DD-HH:MM)
#SBATCH --output=%x_%A_%a.log

set -euo pipefail

MODE=${MODE:-prod}
NSHARDS=${SLURM_ARRAY_TASK_COUNT:-12}
SHARD=${SLURM_ARRAY_TASK_ID:-0}
CPUS=${SLURM_CPUS_PER_TASK:-1}

REPO=$PWD
[ -d "$REPO/weaselytics" ] || {
  echo "submit from the checkout: no $REPO/weaselytics" >&2; exit 2; }

# `data` lives inside the checkout on the cluster and beside it on some
# workstations. Look, do not assume.
if [ -z "${DATADIR:-}" ]; then
  for c in "$REPO/data" "$REPO/../data"; do
    [ -d "$c" ] && { DATADIR=$(cd "$c" && pwd); break; }
  done
fi
[ -n "${DATADIR:-}" ] && [ -d "$DATADIR" ] || {
  echo "no data directory found; set DATADIR" >&2; exit 2; }

STAMP=${STAMP:-$(date +%Y-%m-%d)}
OUTROOT=${OUTROOT:-$REPO/runs/DW_${MODE}_${STAMP}}
CACHEDIR=$OUTROOT/r2_cache
mkdir -p "$CACHEDIR"

module load scipy-stack
source /lustre09/project/6004866/$USER/virtualenv/HPLC_ENV/bin/activate

# Run the branch checked out in $REPO, not an installed copy.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# One BLAS thread per worker process: the parallelism is at the level of
# the r2 sweep, so threaded BLAS would only oversubscribe.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "mode=$MODE shard=$SHARD/$NSHARDS cpus=$CPUS"
echo "repo=$REPO"
echo "data=$DATADIR"
echo "out =$OUTROOT"
python -c "import weaselytics.baseline as b; print('channel:', b._R2_CHANNEL)"

case "$MODE" in
prod)
  # Interleave rather than take a contiguous block: the files are sorted
  # by molecule and the record lengths cluster, so contiguous shards
  # would differ in wall time.
  mapfile -t FILES < <(ls "$DATADIR"/*/*.txt | sort \
                       | awk -v n="$NSHARDS" -v s="$SHARD" 'NR%n==s')
  echo "${#FILES[@]} signals in this shard"
  for f in "${FILES[@]}"; do
    python -m weaselytics "$f" -n -e -p \
        -od "$OUTROOT" -cd "$CACHEDIR" -w "$CPUS"
    printf "\n"
  done
  ;;
synth)
  # No default: guessing this path is what produced a stray W_synth
  # directory on the cluster once already.
  [ -n "${DATASET:-}" ] || {
    echo "MODE=synth needs DATASET=/path/to/the/synthetic/dataset" >&2
    exit 2; }
  [ -d "$DATASET/signals" ] || { echo "missing: $DATASET/signals" >&2; exit 2; }
  python "$REPO/tools/synth_diag.py" "$DATASET" \
      --cache-dir "$CACHEDIR" --diag-dir "$OUTROOT/diag" \
      --workers "$CPUS" --nshards "$NSHARDS" --shard "$SHARD"
  ;;
*)
  echo "MODE must be prod or synth" >&2
  exit 2
  ;;
esac

echo "shard $SHARD done: $(ls "$CACHEDIR" | wc -l) curves in the cache"
