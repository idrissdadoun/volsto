#!/bin/sh
# Local correlation model (SPEC §8.7, M12 part LC7): the long passes, in the owner's order
# (fourth round, 2026-10-09: the calendar repair of every leg and the unscreened fallback are
# the defaults of configs/studies/dispersion/lcm.yaml).  Run from a frozen worktree of the
# commit to be measured, writing into the worktree that holds outputs/dispersion_lc:
#
#   nohup caffeinate -ims sh scripts/lcm_sweep.sh ~/Code/volsto-lc \
#       >> ~/Code/volsto-lc/outputs/dispersion_lc/logs/driver.log 2>&1 &
#
# Every pass is resumable by date (scripts/disp_lcm.py): running this script again skips what
# the same commit and configuration already wrote.  STEPS selects the steps (default: P F T Y Z).
#   P  3m production pass on the default configuration (8e5 particles and paths, deltas, the
#      variance swap on the reference dates) over the 219 dates, today and the reference dates
#      first; the report at the end
#   F  today's full risk at production budget (the row is risk_full_2026-10-02_3m.json)
#   T  today alone at 12m, then at 24m: development budget (2e5), no risk — the clipped mass at
#      lambda = 0 and at the cap on the repaired DJX surface (decision 5)
#   Y  12m development pass over the monthly dates
#   Z  24m, development budget, on the four reference dates only: indicative (the row of today
#      is T's)
#   N  optional sensitivity, not in the default steps: 3m development pass with no repair and no
#      fallback (lcm_norepair.yaml; rows, logs and table carry the suffix _norepair).  A row of
#      that name written by an earlier commit is run again and replaced.
ROOT=${1:?usage: lcm_sweep.sh <worktree that holds outputs/dispersion_lc> [steps]}
STEPS=${2:-"P F T Y Z"}
PY=${PY:-$ROOT/.venv/bin/python}
WORKERS=${WORKERS:-auto}
cd "$(dirname "$0")/.." || exit 1
# the frozen tree's own package, ahead of the environment's editable install of the live tree
PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH
mkdir -p "$ROOT/outputs/dispersion_lc/logs"
echo "=== $(date '+%Y-%m-%d %H:%M:%S') driver in $(pwd), commit $(git rev-parse --short HEAD), steps: $STEPS"
"$PY" -c "import volsto, sys; print('volsto from', volsto.__file__)"

run() {
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') start: $*"
    "$PY" "$@"
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') exit code $?: $*"
}

for step in $STEPS; do
    case $step in
    P) run scripts/disp_lcm.py --tenor 3m --dates monthly --budget production --risk deltas \
        --varswap --workers "$WORKERS" --root "$ROOT" ;;
    F) run scripts/lcm_price.py --date 2026-10-02 --tenor 3m --budget production --risk full \
        --root "$ROOT" --row-out "$ROOT/outputs/dispersion_lc/risk_full_2026-10-02_3m.json" ;;
    T) run scripts/disp_lcm.py --tenor 12m --dates today --budget development --risk none \
        --workers 1 --root "$ROOT" --no-report
       run scripts/disp_lcm.py --tenor 24m --dates today --budget development --risk none \
        --workers 1 --root "$ROOT" --no-report ;;
    Y) run scripts/disp_lcm.py --tenor 12m --dates monthly --budget development --risk none \
        --workers "$WORKERS" --root "$ROOT" ;;
    Z) run scripts/disp_lcm.py --tenor 24m --dates reference --budget development --risk none \
        --workers "$WORKERS" --root "$ROOT" ;;
    N) run scripts/disp_lcm.py --tenor 3m --dates monthly --budget development --risk none \
        --workers "$WORKERS" --config configs/studies/dispersion/lcm_norepair.yaml --tag norepair \
        --root "$ROOT" ;;
    *) echo "unknown step $step" ;;
    esac
done
echo "=== $(date '+%Y-%m-%d %H:%M:%S') driver finished (steps: $STEPS)"
