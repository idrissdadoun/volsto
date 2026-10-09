#!/bin/sh
# Local correlation model (SPEC §8.7, M12 part LC7): the long passes, in the owner's order
# (third round, 2026-10-08).  Run from a frozen worktree of the commit to be measured, writing
# into the worktree that holds outputs/dispersion_lc:
#
#   nohup caffeinate -ims sh scripts/lcm_sweep.sh ~/Code/volsto-lc \
#       >> ~/Code/volsto-lc/outputs/dispersion_lc/logs/driver.log 2>&1 &
#
# Every pass is resumable by date (scripts/disp_lcm.py): running this script again skips what
# the same commit and configuration already wrote.  STEPS selects the steps (default: all).
#   0  today and the three other reference dates, 3m, production budget, deltas, variance swap
#   1  3m development pass (2e5 particles and paths, no risk) over the 219 dates
#   2  3m production pass (8e5, deltas)
#   3  today's full risk at production budget
#   4  12m development pass          5  24m development pass
ROOT=${1:?usage: lcm_sweep.sh <worktree that holds outputs/dispersion_lc> [steps]}
STEPS=${2:-"0 1 2 3 4 5"}
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
    0) run scripts/disp_lcm.py --tenor 3m --dates reference --budget production --risk deltas \
        --varswap --workers 1 --root "$ROOT" --no-report ;;
    1) run scripts/disp_lcm.py --tenor 3m --dates monthly --budget development --risk none \
        --workers "$WORKERS" --root "$ROOT" ;;
    2) run scripts/disp_lcm.py --tenor 3m --dates monthly --budget production --risk deltas \
        --varswap --workers "$WORKERS" --root "$ROOT" ;;
    3) run scripts/lcm_price.py --date 2026-10-02 --tenor 3m --budget production --risk full \
        --root "$ROOT" --row-out "$ROOT/outputs/dispersion_lc/risk_full_2026-10-02_3m.json" ;;
    4) run scripts/disp_lcm.py --tenor 12m --dates monthly --budget development --risk none \
        --workers "$WORKERS" --root "$ROOT" ;;
    5) run scripts/disp_lcm.py --tenor 24m --dates monthly --budget development --risk none \
        --workers "$WORKERS" --root "$ROOT" ;;
    *) echo "unknown step $step" ;;
    esac
done
echo "=== $(date '+%Y-%m-%d %H:%M:%S') driver finished (steps: $STEPS)"
