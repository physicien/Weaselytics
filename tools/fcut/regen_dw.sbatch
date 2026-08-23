#!/bin/bash
# Refill the r2 caches with the Durbin-Watson statistic.
#
# The sweep now stores dw_val alongside r2_val, and _R2_CHANNEL is bumped
# to "y-baseline-dw", so every existing entry misses and the BEADS sweep
# has to be paid once more.
#
# One task, not an array. The sweep already parallelises internally over
# the cutoff grid (-w), and the whole production set costs about 8400 s
# of autocorrelation serially, so on 32 cpus it is minutes. An array
# would add a queue wait per shard and a merge step afterwards to
# subdivide a job that does not need it.
#
#   sbatch -J DW_PROD  --export=ALL,MODE=prod tools/fcut/regen_dw.sbatch
#   sbatch -J DW_SYNTH --export=ALL,MODE=synth,\
#DATASET=/path/to/SYNTH_ERB_2026-08-18 tools/fcut/regen_dw.sbatch
#
# The job name is not optional. Slurm resolves %x in the log filename at
# submission, before this script runs, so the name cannot be derived
# from MODE here: two runs submitted without -J produce two logs called
# the same thing and only the job id tells them apart. The check below
# refuses to start unless the name matches the mode.
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

#SBATCH --job-name=DW_UNNAMED         # override with -J, see above
#SBATCH --account=rrg-cotemich-ac
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=manubou5@gmail.com
##SBATCH --ntasks=1                   # cpus, if across several nodes
##SBATCH --mem-per-cpu=2000M          # memory, if across several nodes
#SBATCH --nodes=1                     # nb of whole nodes
#SBATCH --ntasks-per-node=1           # one task; the sweep forks its own
#SBATCH --cpus-per-task=32            # workers for the r2 sweep (-w)
#SBATCH --mem=64000M                  # mem=0 for all memory of the node
#SBATCH --time=00-02:00               # time (DD-HH:MM)
#SBATCH --output=%x_%j.log

set -euo pipefail

MODE=${MODE:-prod}
WANT="DW_$(echo "$MODE" | tr '[:lower:]' '[:upper:]')"
if [ "${SLURM_JOB_NAME:-$WANT}" != "$WANT" ]; then
  echo "job name is '${SLURM_JOB_NAME}', expected '$WANT'." >&2
  echo "resubmit with:  sbatch -J $WANT --export=ALL,MODE=$MODE ..." >&2
  exit 2
fi
# Sharding is inert here: one task takes everything. The flags stay
# because synth_diag.py carries them and they cost nothing at 1/0.
NSHARDS=${NSHARDS:-1}
SHARD=${SHARD:-0}
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

echo "done: $(ls "$CACHEDIR" | wc -l) curves in the cache"

# The log belongs with the run it describes, not loose in the checkout.
# Slurm still holds the file open, so it is copied now and the original
# is left for Slurm to finish writing and for the user to remove.
LOG=$SLURM_SUBMIT_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log
[ -f "$LOG" ] && cp "$LOG" "$OUTROOT/" && echo "log copied to $OUTROOT/"
