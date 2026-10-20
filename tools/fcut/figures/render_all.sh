#!/bin/bash
# Re-render every figure in tools/fcut/images/ and install it there.
#
# All fifteen are JPEG at 200 dpi, quality 92, progressive. Re-render
# rather than upscale: the sources are vector until the final conversion.
#
# The six hist_* figures come from the package as it stood at five older
# commits, so they need a worktree per commit and the historical
# interpreter. The worktrees are created here if absent and left in place;
# `git worktree prune` after deleting them clears the registrations.
#
# Usage:  ./render_all.sh [method|stages|hist|all]     (default: all)
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
IMG=$REPO/tools/fcut/images
WORK=${WEASELYTICS_FIGURE_WORK:-$HERE/.work}
PY=${WEASELYTICS_PY:-$HOME/virtualenv/DEV_WEASELYTICS_ENV/bin/python}
HPY=${WEASELYTICS_HIST_PY:-$HOME/virtualenv/WEASELYTICS_HIST_ENV/bin/python}
DATA=${WEASELYTICS_DATA:-$REPO/../data}
WHAT=${1:-all}

mkdir -p "$IMG" "$WORK"
for p in "$PY" "$DATA"; do
  [ -e "$p" ] || { echo "missing: $p" >&2; exit 1; }
done

conv () {  # conv <png> <name>
  "$PY" - "$1" "$IMG/$2.jpg" <<'PYX'
import sys
from PIL import Image
im = Image.open(sys.argv[1]).convert("RGB")
im.save(sys.argv[2], "JPEG", quality=92, optimize=True, progressive=True)
print(f"    {sys.argv[2].split('/')[-1]}  {im.size[0]}x{im.size[1]}")
PYX
}

run_py () {  # run_py <script>
  ( cd "$WORK" && "$PY" "$WORK/$1" >/dev/null 2>&1 )
}

# The figure scripts resolve their output directory from __file__, so they
# are run from a copy in $WORK and the PNGs land there rather than in the
# repo next to the sources.
sync_scripts () {
  cp "$HERE"/fig_*.py "$HERE"/hist_render.py "$WORK/" 2>/dev/null
}

if [ "$WHAT" = all ] || [ "$WHAT" = method ]; then
  echo "=== the four method figures ==="
  sync_scripts
  ( cd "$WORK" && "$PY" fig_vocab.py  >/dev/null 2>&1 ) && conv "$WORK/fig_vocab.png"  method_vocabulary
  ( cd "$WORK" && "$PY" fig_stages.py >/dev/null 2>&1 ) && conv "$WORK/fig_stages.png" method_three_stages
  ( cd "$WORK" && "$PY" fig_scut.py   >/dev/null 2>&1 ) && conv "$WORK/fig_scut.png"   method_scut
  # fig_diag drives auto_beads, which writes the diagnostic itself under
  # its own output_dir, so the newest PNG there is the one wanted.
  rm -rf "$WORK/diagout" && mkdir -p "$WORK/diagout"
  sed "s|output_dir=OUT|output_dir='$WORK/diagout'|" "$HERE/fig_diag.py" \
      > "$WORK/fig_diag_run.py"
  ( cd "$WORK" && "$PY" fig_diag_run.py >/dev/null 2>&1 )
  P=$(find "$WORK/diagout" -name '*.png' -printf '%T@ %p\n' 2>/dev/null \
      | sort -rn | head -1 | cut -d' ' -f2)
  [ -n "$P" ] && conv "$P" method_diagnostic \
      || echo "    method_diagnostic: no png produced"
fi

if [ "$WHAT" = all ] || [ "$WHAT" = stages ]; then
  echo "=== the five stage figures ==="
  sync_scripts
  run_py fig_stage1.py
  run_py fig_stage23.py
  for n in stage1_segments stage1_features stage1_fallback \
           stage2_exclusions stage3_select; do
    [ -f "$WORK/$n.png" ] && conv "$WORK/$n.png" "$n" \
        || echo "    $n: no png produced"
  done
fi

if [ "$WHAT" = all ] || [ "$WHAT" = hist ]; then
  echo "=== the six history figures ==="
  [ -x "$HPY" ] || { echo "    need $HPY (WEASELYTICS_HIST_PY)"; exit 1; }
  sync_scripts
  render_hist () {  # render_hist <commit> <signal-stem> <outname>
    local wt=$WORK/hist_$1 out=$WORK/rr_$3
    if [ ! -d "$wt" ]; then
      git -C "$REPO" worktree add --detach "$wt" "$1" >/dev/null 2>&1 \
          || { echo "    $3: cannot check out $1"; return; }
    fi
    rm -rf "$out" && mkdir -p "$out"
    local sig
    sig=$(ls "$DATA"/*/"$2".txt 2>/dev/null | head -1)
    [ -z "$sig" ] && { echo "    $3: signal $2 not found under $DATA"; return; }
    "$HPY" "$WORK/hist_render.py" "$wt" "$sig" "$out" >/dev/null 2>&1
    local p
    p=$(find "$out" -name '*.png' -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -1 | cut -d' ' -f2)
    [ -n "$p" ] && conv "$p" "$3" || echo "    $3: no png produced"
  }
  render_hist 4883e7c 2-Xylene__LPYE__CS2__15                    hist_derivative_tolerances
  render_hist b6ddd57 2-Dichlorobenzene__LPYE__C60__4            hist_rolling_std
  render_hist b6ddd57 2-Chlorotoluene__LPYE__60-100__1           hist_rolling_std_bimodal
  render_hist 5415895 4-Xylene__LPYE__60-100__3                  hist_rolling_mad
  render_hist 75f21df 3-Chlorotoluene__LPYE__MINOR_isomer_C90__1 hist_crossings
  render_hist f9316c3 4-Xylene__LPYE__60-100__3                  hist_hybrid_fragmented
fi

echo "=== installed ==="
ls -l "$IMG"/*.jpg | awk '{printf "  %8d  %s\n", $5, $9}' | sed "s|$IMG/||"
