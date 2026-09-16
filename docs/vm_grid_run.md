# Running the default grid on a rented VM

This runbook covers running `configs/grids/default.yaml` (131 points, light risk tier on the
3-bucket ladder, 8·10⁵ particles, 4·10⁵ pricing paths) on one or two rented VMs, and bringing
the results store **and** the leverage cache back to the laptop. It also covers sizing later
grids from measured per-point cost. Set the variables in §0 first. Every command meant for the
laptop was run on the laptop (macOS, `/usr/bin/rsync` = openrsync) when this page was written;
the VM commands assume Ubuntu 24.04.

## 0. The answer: expected wall clock of the full light-tier grid

The grid costs **177 process-hours at 1 thread per process, at laptop per-core speed**
(measured per-point costs with the risk step on, §1). A VM divides that by the number of
worker processes it can run at full speed, one per **physical core** (a point gains little from
more threads: 1.7x from 12 threads). Two things are not known until the VM probe (§6) runs:
the VM's per-core speed relative to the laptop (**f**, plausibly 1.0–1.5, slower is larger) and
how much 24–48 concurrent processes slow each other down (**e**, plausibly 0.75–1.0). The
"expected" column takes f = 1.2 and e = 0.9, a factor 1.33 on the laptop-speed figure; the
range takes f/e from 1.0 to 2.0, the model's own −5% / +5% (the spread of three projections
from different stores, §1 — a consistency check, not independent evidence) and up to +3% for
the 2F and marking points charged at the 1F cost.

| layout | laptop-speed lower bound | **expected** | range | memory needed |
|---|---|---|---|---|
| (a) one 48-vCPU / 24-core instance (CCX63, c7i.12xlarge), `--workers 24 --threads-per-worker 1` | 7.4 h | **≈ 10 h** | 7–16 h | 24 × 2.36 GiB = 57 GiB (≥ 71 GiB RAM at the 0.8 rule): fits both |
| (b) one 96-vCPU / 48-core instance (c7i.24xlarge), `--workers 48 --threads-per-worker 1` | 3.7 h | **≈ 5 h** | 3.5–8 h | 48 × 2.36 GiB = 113 GiB (≥ 142 GiB RAM): fits 192 GiB |
| (c) two 48-vCPU instances, `--shard 1/2` and `--shard 2/2`, each `--workers 24 --threads-per-worker 1` | 3.7 h (3.67 / 3.70 h per VM) | **≈ 5 h** | 3.5–8 h | 57 GiB per VM, as (a) |

- The **~3 h target is not reachable on 48 physical cores at laptop per-core speed with
  e = 1**: the floor for (b) and (c) is 3.7 h. About 3 h needs a VM core faster than the
  laptop's, or about 60 physical cores.
- (b) and (c) have the same number of cores. (c) is likely a little faster in practice, because
  24 processes per box contend less for memory bandwidth than 48; (b) is simpler (one cache, one
  store, no merge). The two halves of (c) are balanced (88.1 and 88.9 process-hours).
- On a 48-vCPU box, 24 workers × 2 threads puts the second thread on an SMT sibling. With two
  *real* laptop cores the grid is 154 process-hours (6.4 h floor on 24 workers). An SMT sibling
  gains less, so expect (a) with 2 threads to land between 6.4 h and 7.4 h at laptop speed.
  Probe A measures the sibling's gain. Use 2 threads only if it gains at least 10%.
- The single longest point (an LSV point calibrating 17 times) takes 1.38 h at laptop speed,
  well under every layout's wall clock, so it does not set the tail.
- Without the risk step (`--risk none`) the grid is 24.8 process-hours: 1.0 h floor on 24
  workers.
- Memory is the measured peak RSS of a production light-tier point with the risk step on
  (2.36 GiB, 1F point; the LV point 2.12 GiB). 2F and marking points have not had their memory
  measured. Probe D measures a point on the VM before the worker count is fixed (§7).

What the VM probe pins down (§6): f from probe A, e(24) and e(48) from probe B, the risk-step
costs from probe C, and the production peak RSS from probe D. The wall clock is then
`(parallel line of the §7 dry run) × f / e`, and with `--cost-from` a VM store already
carries f.

```bash
# on the laptop and on the VM
export REPO=volsto                       # repository directory name
export VM=root@203.0.113.10              # ssh target (ubuntu@... on AWS)
export DATA=/data                        # VM working volume (EBS / Hetzner volume on spot)
```

## 1. What the numbers rest on

All measurements below were taken on the owner's laptop: Apple M-series, 12 logical cores
(8 performance + 4 efficiency), 24 GiB. The laptop was shared with other agents, with load
average 2–8 during the runs. Every run calibrated into a scratch cache, never the repository
cache. Wall clocks are timings, not Monte Carlo estimates; run-to-run variation on the shared
laptop was 3–6% (particle pass 191.5 vs 197.1 s, pricing 281.7 vs 265.4 s in two production
runs).

**Per-point cost at the production budget, 1 thread** (placeholder surface, 1F ν 0.5 ρ −0.7
κ 1.5; `--risk light` on the 3-bucket ladder; store `scratchpad/vmfix/prodrisk/store`; run
wall 6211 s; peak RSS 2.36 GiB):

| step | 1F point | LV point |
|---|---|---|
| calibration step (particle pass 197.1 s) | 203.0 s | 2.8 s (Dupire) |
| diagnostics | 36.7 s | — |
| pricing (M4 + 2y→3y + M6) | 265.4 s | 187.5 s |
| analytics (SSR + Var(V)) | 196.6 s | 90.1 s |
| **risk** (17 states, 38 pricings) | **4283.0 s** | **943.5 s** |
| total | 4984.8 s | 1223.9 s |

The risk step is 86% of an LSV point. It breaks down into four parts:

- 16 recalibrations (3045.5 s), since the 17th state is the point's own leverage;
- per state, 4.85 s of serial surface work outside the engine plus a 6.05 s state build (1.04 s
  for LV);
- 38 pricings at 0.104 × the pricing step (0.118 for LV).

The per-state costs and the pricing ratio were separated by measuring the same point at two
budgets (probe C, 10⁵ / 5·10⁴: 1F risk 726.3 s, LV 206.7 s; constants in
`volsto/viewers/precompute.py`, `RISK_STATE_OVERHEAD_S`, `RISK_STATE_BUILD_S` and
`RISK_PRICING_RATIO`). What that does and does not show:

- The outside-engine work per state is measured directly at both budgets and agrees to 1.7%
  (4.91 / 4.83 s on the 1F point, 4.92 / 4.87 s on the LV point). This is the one independent
  check of the decomposition.
- The state build and the pure pricing are *solved* from the two budgets, with the pricing
  assumed to scale exactly with the paths (×8). The model therefore reproduces both risk steps
  by construction. The ratio of the pure pricing to the pricing step agrees at both budgets
  (0.101 / 0.104 LSV, 0.120 / 0.118 LV), but that only says the pricing step itself scaled by
  about 8; it is not evidence for the split. The earlier cost rule
(17 × calibration + 38 × 0.0913 × pricing) gives 4271.5 s on this 1F point (−0.3%), but only by
cancellation. It charges 17 particle passes where 16 were run (3350.7 s against 3045.5 s), and
921 s where the pricings plus the per-state work took 1237.5 s. On the LV point it gives 650 s,
31% short (2.6x short at the probe budget). Scaling probe C by 8 overstates the 1F step by 36%.

**Three projections of the grid from different stores, at 1 thread, agree within 5%.** They
are not independent: all three use `RISK_STATE_BUILD_S`, which was fitted from the first and
third stores together.
- `--cost-from scratchpad/vmfix/prodrisk/store` (production, risk measured): 177.0
  process-hours;
- `--cost-from scratchpad/prod/store_1` (production, risk off, risk from the constants):
  180.6;
- `--cost-from scratchpad/vmverify/P/C/store` (probe C, rescaled ×8 in particles and paths):
  185.7.

A source without an LV point (probe D's `--only` store) charges the LV risk pricing as
`RISK_PRICING_RATIO["lv"]` × the LV pricing step, taken as the LSV pricing × 0.7065
(`LV_TO_LSV_PRICING_RATIO`, 187.5 / 265.4 s in the production run); the dry run names that
fallback. The 2F point and the 36 marking points are charged the 1F point's costs. The only measurement
of 2F cost, `outputs/store` at 12 threads, puts its pricing at 1.14x the 1F pricing, which
adds up to about 3% to the grid.

**Threads.** One 1F point at the probe budget, at 1 / 2 / 4 / 8 / 12 threads
(`scratchpad/scaling`), gave t(n)/t(1) of:

| step | 2 | 4 | 8 | 12 | production, 12 |
|---|---|---|---|---|---|
| particle pass | 0.918 | 0.801 | 0.717 | 0.644 | 0.709 |
| diagnostics | 0.733 | 0.586 | 0.519 | 0.503 | 0.488 |
| pricing (and risk pricing) | 0.752 | 0.616 | 0.554 | 0.545 | 0.540 |
| analytics | 0.754 | 0.621 | 0.557 | 0.552 | 0.546 |

`--cost-from` rescales between thread counts with this table (`MEASURED_THREAD_RATIO`),
interpolated in 1/n. A single Amdahl fraction fitted to all five columns (p = 0.316 / 0.549 /
0.504 / 0.498) fits diagnostics, pricing and analytics to within 0.009. It misfits the
particle pass by +0.076 at 2 threads and −0.066 at 12: the second core gains 8%, not the 16% a
single fraction implies. A whole 1F point runs 1.70x faster on 12 threads than on 1 (720.1 vs
422.8 s): it uses the equivalent of 1.7 of the 12 cores. Throughput therefore comes from many
single-thread processes. The 2-thread column is two real cores; an SMT sibling on a server
gains less (§0). The laptop-speed projections at 1 thread read a 1-thread store directly and
involve no thread rescaling.

| layout (light tier, 3 buckets, `--cost-from scratchpad/vmfix/prodrisk/store`) | process-hours | floor |
|---|---|---|
| 1 process × 12 threads (the laptop) | 111.2 | 111.2 h |
| 24 × 1 | 177.0 | 7.37 h |
| 24 × 2 (two real cores each) | 154.3 | 6.43 h |
| 12 × 4 | 133.4 | 11.12 h |
| 48 × 1 | 177.0 | 3.69 h |
| 48 × 2 (two real cores each) | 154.3 | 3.22 h |
| `--shard 1/2`, `2/2` at 24 × 1 | 88.06 / 88.89 | 3.67 / 3.70 h |
| `--shard i/24`, one process each | 5.81–8.31 per shard | 8.31 h (the slowest shard) |

These dry runs took 0.7–0.9 s each, and nothing was computed. The command is:

```bash
NUMBA_NUM_THREADS=2 .venv/bin/python -m volsto.viewers.precompute --grid configs/grids/default.yaml \
  --dry-run --store /tmp/vmdry --cost-from <store> --workers W --threads-per-worker T [--shard i/n]
```

With `--shard`, the `parallel (shard i/n, ...)` line is that shard's wall clock, not the whole
grid's.

## 2. Provisioning spec

| instance | vCPU | physical cores | RAM | note |
|---|---|---|---|---|
| Hetzner CCX63 | 48 dedicated | 24 (AMD EPYC, SMT) | 192 GB | 960 GB NVMe, no spot |
| AWS c7i.12xlarge | 48 | 24 (Xeon 8488C, SMT) | 96 GiB | spot available |
| AWS c7i.24xlarge | **96** | 48 | 192 GiB | the "c7i.24xlarge" of the brief is 96 vCPU, not 48 |

Instance facts are from the providers' public listings (not re-checked when this page was
written); check the price and availability yourself. Use Ubuntu 24.04 with **at least 100 GB of
disk**. The run adds about 2,150 leverage entries of about 18 MB each (8·10⁵ particles), roughly
39 GB: 123 base calibrations plus 16 risk-bump recalibrations for each of the 127 LSV points
(the 17th state is the point's own leverage). That is on top of today's 5.3 GB cache, the venv
and the store. On spot, put `$DATA` on a persistent volume (EBS) so an interruption keeps the
finished points and leverages. The cache must be on a **local** file system (ext4 / xfs on the
instance or an attached volume): the manifest lock (§13) is a POSIX `flock`, which is not
reliable over NFS.

## 3. On the laptop: pin the commit, package the code, checksum the cache

The VM must run a **committed** tree: cache keys depend on the calibration code tag and the spec.

```bash
cd ~/Code/$REPO
git status --short                                   # must print nothing
export COMMIT=$(git rev-parse HEAD); echo $COMMIT
git bundle create /tmp/$REPO.bundle HEAD             # there is no git remote: ship a bundle
# cache integrity: key count, manifest hash, per-file checksums (the manifest and its lock are
# checked separately: the VM rewrites the manifest; leftover temporaries of interrupted writes,
# '.*.tmp*', are not cache content, §13)
ls cache/*/leverage.npz | wc -l
shasum -a 256 cache/manifest.parquet
(cd cache && find . -type f ! -name '.DS_Store' ! -name 'manifest.parquet*' ! -name '.*.tmp*' -print0 \
  | sort -z | xargs -0 shasum -a 256) > /tmp/cache.sha256
wc -l < /tmp/cache.sha256                            # 2 lines per leverage (npz + spec), more with diagnostics
# the default projection, for the record (reads the cache manifest, computes nothing). Without
# --cost-from it uses the cache manifest and the fallback costs, measured at 12 threads:
# 126.52 process-hours, which is NOT the 1-thread figure of §0. The §0 figures come from --cost-from <a 1-thread store>, §1.
NUMBA_NUM_THREADS=2 .venv/bin/python -m volsto.viewers.precompute --grid configs/grids/default.yaml \
  --dry-run --store /tmp/vmdry --workers 24 --threads-per-worker 1
scp /tmp/$REPO.bundle /tmp/cache.sha256 $VM:/tmp/
```

## 4. VM setup at the exact commit

```bash
ssh $VM
export REPO=volsto DATA=/data COMMIT=<paste the commit>
sudo apt-get update && sudo apt-get install -y git rsync tmux build-essential
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
sudo mkdir -p $DATA && sudo chown $USER $DATA && cd $DATA
git clone /tmp/$REPO.bundle $REPO && cd $REPO && git -c advice.detachedHead=false checkout $COMMIT
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev,viewers]"
.venv/bin/python -c "import volsto, numba; print(numba.__version__, numba.config.NUMBA_NUM_THREADS)"
nproc; lscpu | grep -E 'Model name|Thread|Core|Socket'; free -g
```

The commands below use `.venv/bin/python -m volsto.viewers.precompute` (and
`-m volsto.viewers.app`), which work whether or not the console scripts `volsto-precompute` /
`volsto-viewer` were installed.

## 5. Bring the cache in and check its integrity

```bash
# on the laptop (macOS openrsync: -a --partial --progress are portable; --info=progress2 is not)
rsync -a --partial --progress --stats \
  --exclude 'manifest.parquet.lock' --exclude '.*.tmp*' \
  ~/Code/$REPO/cache/ $VM:$DATA/$REPO/cache/
# on the VM
cd $DATA/$REPO
ls cache/*/leverage.npz | wc -l                      # = the laptop's count
sha256sum cache/manifest.parquet                     # = the laptop's hash
(cd cache && sha256sum --quiet -c /tmp/cache.sha256) && echo CACHE OK
.venv/bin/python -c "from volsto.calibration.cache import check_guard; check_guard(); print('code tag OK')"
```

## 6. Probe the VM before choosing the layout

The laptop numbers transfer only once the VM's per-core speed, its concurrency efficiency and
its memory per process are known. Probes A–C use the reduced budget (10⁵ particles, 5·10⁴
paths), whose linear rescaling to production was checked on the laptop (§1). Probe D is one
point at the production budget. Every probe calibrates **into `$DATA/probe`, never into
`cache/`**.

```bash
cd $DATA/$REPO && export P=$DATA/probe && mkdir -p $P
cat > $P/grid.yaml <<'EOF'
# VM probe: one 1F point + the LV point on the placeholder surface, reduced budget.
name: vm_probe
reference_spec: configs/studies/lsv_reference_2f.yaml
surfaces:
  - {name: placeholder, kind: placeholder, one_factor: true, two_factor: false, marking: false}
particle: {n_particles: 100000, horizon: 3.0}
one_factor: {nu: [0.5], rho: [-0.7], kappa: [1.5]}
two_factor_presets: []
include_degenerate: false
products: {m4: true, m6: true, conditional: true, cliquet_maturities: [1.0, 2.0]}
pricing: {n_paths: 50000, seed: 2024}
risk: {tier: none, products: ["autocall 3y", "cliquet 1y"], light_pillars: [0.25, 1.0, 3.0], fwd_var_buckets: 3}
EOF
sed -e 's/n_paths: 50000/n_paths: 400000/' -e 's/n_particles: 100000/n_particles: 800000/' \
  -e 's/name: vm_probe/name: vm_probe_production/' $P/grid.yaml > $P/grid_prod.yaml
PC=volsto.viewers.precompute            # used as: .venv/bin/python -m $PC ...

# D (start first, runs alongside A-C, ~1.4 h at laptop speed): one production light-tier 1F
#   point at 1 thread, with its peak RSS sampled — THE memory number the worker count needs
ONE_F=$(.venv/bin/python -c "from volsto.viewers.grid import load_grid, enumerate_points
print(next(p.id for p in enumerate_points(load_grid('$P/grid_prod.yaml')) if p.mode == 'one_factor'))")
NUMBA_NUM_THREADS=1 nohup .venv/bin/python -m $PC --grid $P/grid_prod.yaml --store $P/D/store --cache $P/D/cache \
  --risk light --only $ONE_F > $P/D.log 2>&1 &
DPID=$!
(while kill -0 $DPID 2>/dev/null; do echo "$(date +%s) $(ps -o rss= -p $DPID)"; sleep 5; done) > $P/D.rss &

# A. per-core speed and the second thread: one process at 1 and at 2 threads (~2 min each)
for t in 1 2; do
  NUMBA_NUM_THREADS=$t .venv/bin/python -m $PC --grid $P/grid.yaml --store $P/A$t/store --cache $P/A$t/cache \
    --risk none > $P/A$t.log 2>&1
done
# laptop reference, same probe: 1F point 93.1 s at 1 thread (particle pass 25.2 / diag 4.6 /
# pricing 33.2 / analytics 24.2), 75.8 s at 2 real cores (23.2 / 3.4 / 24.9 / 18.2)
for t in 1 2; do .venv/bin/python -m $PC report --store $P/A$t/store --no-write | grep -E '\| one_factor \| calibrated \| (calibration|pricing|total) '; done
# B. concurrency: K simultaneous single-thread processes, K = cores - 1 then vCPUs - 1 (one
#    slot is D's): 23 and 47 on a 48-vCPU box, 47 and 95 on a 96-vCPU box
for K in $(( $(nproc) / 2 - 1 )) $(( $(nproc) - 1 )); do
  mkdir -p $P/B$K; PIDS=()
  for i in $(seq 1 $K); do
    NUMBA_NUM_THREADS=1 nohup .venv/bin/python -m $PC --grid $P/grid.yaml --store $P/B$K/s$i \
      --cache $P/B$K/c$i --risk none > $P/B$K/log$i.txt 2>&1 &
    PIDS+=($!)
  done; wait "${PIDS[@]}"
  .venv/bin/python - $P/B$K $P/A1/store <<'EOF'
import sys, pathlib
import pandas as pd
from volsto.viewers.precompute import load_cost_records
runs = sorted(pathlib.Path(sys.argv[1]).glob("s*"))
pts = pd.concat([load_cost_records(s).points for s in runs])
one = load_cost_records(sys.argv[2]).points
t_k = pts.loc[pts.how == "calibrated", "total"].median()
t_1 = one.loc[one.how == "calibrated", "total"].median()
e = t_1 / t_k
print(f"K={len(runs)}: 1F point {t_k:.1f} s vs {t_1:.1f} s alone -> efficiency e(K) {e:.2f}, "
      f"effective workers {len(runs) * e:.1f}; peak RSS {pts.peak_rss_bytes.max() / 2**30:.2f} GiB")
EOF
done
# C. the risk step at 1 thread, reduced budget (~18 min on the laptop): the risk costs by mode
NUMBA_NUM_THREADS=1 .venv/bin/python -m $PC --grid $P/grid.yaml --store $P/C/store --cache $P/C/cache \
  --risk light > $P/C.log 2>&1
.venv/bin/python -m $PC report --store $P/C/store --out $P/C/report
# D, when it has finished: its wall split and its peak RSS
wait $DPID; .venv/bin/python -m $PC report --store $P/D/store --no-write | grep -E 'one_factor \| calibrated \| (risk|total) '
awk 'm<$2{m=$2} END{printf "D peak RSS %.2f GiB\n", m/2^20}' $P/D.rss
```

What each probe pins down, against the laptop:

| probe | measures | laptop value | used for |
|---|---|---|---|
| A, 1 thread | per-core speed f = VM 1F total / 93.1 s | 1 | every wall clock scales by f |
| A, 2 threads | the second thread's gain on an SMT sibling | 75.8 / 93.1 = 0.81 on real cores | whether `--threads-per-worker 2` is worth it |
| B | concurrency efficiency e(K) (e = the lone 1F point's time / its time among K) and RSS at the probe budget | not measurable on 12 cores | effective workers K·e(K) |
| C | risk costs by mode at the probe budget | 1F risk 726.3 s, LV 206.7 s | `--cost-from $P/C/store` |
| D | a production light-tier point and **its peak RSS** | 1F point 4984.8 s (risk 4283.0 s); peak RSS 2.36 GiB | the worker count; validates C's rescaling |

## 7. Choose the layout

1. Memory first. Take D's peak RSS (laptop: 2.36 GiB per process) and keep
   `workers × peak RSS ≤ 0.8 × RAM`, with RAM in GiB (`free -g` prints GiB). A CCX63
   (192 GB = 178.8 GiB) holds 60 workers on that basis, a c7i.12xlarge (96 GiB) holds 32, and
   a c7i.24xlarge (192 GiB) holds 65.
2. Workers. On a 48-vCPU box (24 cores) run 24 × 1, unless probe B shows `47 × e(47)` at
   least 15% above `23 × e(23)`. In that case the SMT siblings do real work: run 47–48 × 1 if
   memory allows. Use 24 × 2 only if probe A's second thread gained at least 10%. On a 96-vCPU
   box, run probe B with K = 47 and 95 instead, and apply the same rule.
3. Project with the VM's own measured costs. Probe D's store (production budget, risk on,
   1 thread) is the best source, and probe C's store is the fallback while D is still running.
   A VM store already carries the per-core speed f, so only e(K) remains to apply. D holds only
   the 1F point, so the LV risk pricing comes from the named fallback ratio (§1); the four LV
   points are under 1% of the grid.

```bash
cd $DATA/$REPO
NUMBA_NUM_THREADS=1 .venv/bin/python -m volsto.viewers.precompute --grid configs/grids/default.yaml \
  --dry-run --store $DATA/store --cache cache --cost-from $P/D/store --workers 24 --threads-per-worker 1
```

The `parallel (shard i/n, ...)` line is the process-hours of the points that invocation would
compute, divided by the workers (or its longest point, if that is longer). Divide it by `e(K)`
from probe B to get the expected wall clock.

With probe D as the source, that division is **conservative**. D ran alongside probes A–C, so
its costs already include some contention (at worst the vCPUs − 1 load of B's second pass, with
D's own SMT sibling busy), and dividing by `e(K)` charges that contention a second time. The
range is therefore from `parallel line` (D's contention taken as the run's) to
`parallel line / e(K)` (the conservative end); plan on the latter. In the other direction,
`e(K)` is measured at the probe budget, where each process holds far less memory than the
2.36 GiB of production, so it may overstate the efficiency of a production run. The ETA after
the first hour (§9) is the check. With `--shard i/n` it is that shard's figure, not
the grid's. A projection from a *laptop* store must also be multiplied by f from probe A.

## 8. Run

Run inside `tmux new -s grid`, with the store and the cache both on `$DATA`.

**One VM, one process pool (recommended: a single queue, so no shard straggles):**

```bash
cd $DATA/$REPO
.venv/bin/python -m volsto.viewers.precompute --grid configs/grids/default.yaml --store $DATA/store \
  --cache cache --workers 24 --threads-per-worker 1 --resume 2>&1 | tee -a $DATA/grid.log
```

**Two VMs** (layout (c) of §0): on VM 1 run the same line with `--shard 1/2`, on VM 2 with
`--shard 2/2`, each with its own copy of the cache and its own `$DATA/store`. The dry runs give
88.06 and 88.89 process-hours for the two halves at laptop speed, 1 thread. Bring both
back (§11, once per VM, merging each VM's manifest rows).

**One VM, n independent shard processes** (a crash only takes out its own shard; the shards
straggle: at laptop speed the 24 shards of the light tier take 5.81–8.31 process-hours
each, so the slowest sets the wall clock):

```bash
cd $DATA/$REPO && N=24 && mkdir -p $DATA/logs
for i in $(seq 1 $N); do
  NUMBA_NUM_THREADS=1 nohup .venv/bin/python -m volsto.viewers.precompute \
    --grid configs/grids/default.yaml --store $DATA/store --cache cache \
    --shard $i/$N --resume > $DATA/logs/shard_$i.log 2>&1 &
done
```

## 9. Monitor

```bash
tail -f $DATA/grid.log | grep --line-buffered -E '^\[|FAILED|done:'   # [k/n] ... elapsed, ETA
ls $DATA/store/results/points | wc -l                                  # points stored so far
ls cache/*/leverage.npz | wc -l                                        # leverages so far
.venv/bin/python -m volsto.viewers.precompute report --store $DATA/store --no-write | head -20
ps -u $USER -o rss= | awk '{s+=$1} END {printf "%.1f GiB total RSS\n", s/2^20}'; free -g; uptime
```

The running ETA after each point is the mean point time so far multiplied by the points left;
the first points finish out of order, so trust it after about 30 points. The run record
(`results/runs/*.json`) is written when a run **ends**, so the report only covers finished runs.
Watch the total RSS through the first hour (the first risk steps) and stop the run if it passes
0.85 × RAM (`--resume` picks it up with fewer workers).

## 10. Spot interruption and `--resume`

An interrupted point is lost; everything stored before the interruption is kept, including every
calibrated leverage (written as soon as its particle pass finishes, with its manifest row).
Every cache write is atomic (a temporary file in the same directory, fsync'd, then renamed:
`atomic_write` in `volsto/calibration/cache.py`), so a kill mid-write leaves the previous file or
no file, never a torn `leverage.npz` that `--resume` would take for a hit. `has()` also rejects
a `leverage.npz` that is not a complete archive. A kill can leave a `.<name>.<hex>.tmp.<ext>`
temporary, which nothing reads (§13). Restart
the same command line with `--resume` (already present above). `--resume` skips stored points,
turns a stored point whose leverage exists into a cache hit, and refreshes a point stored below
the requested tier for the risk step only. Failed points are listed at the end of the log and in
the run record, the CLI exits 1, and a later `--resume` picks them up. On AWS, watch for the
two-minute notice (`curl -s http://169.254.169.254/latest/meta-data/spot/instance-action`) and
rsync back when it appears (§11).

## 11. Rsync back: the store and the cache

The new leverages are the expensive part (about 2,150 calibrations, roughly 39 GB). Bring both
back.

```bash
# on the laptop
mkdir -p ~/Code/$REPO/outputs/store ~/Code/$REPO/outputs/vm_run
rsync -a --partial --progress --stats $VM:$DATA/store/ ~/Code/$REPO/outputs/store/
rsync -a --partial --progress --stats --exclude 'manifest.parquet' \
  --exclude 'manifest.parquet.lock' --exclude '.*.tmp*' \
  $VM:$DATA/$REPO/cache/ ~/Code/$REPO/cache/
rsync -a $VM:$DATA/$REPO/cache/manifest.parquet /tmp/vm_manifest.parquet
rsync -a --partial $VM:$DATA/grid.log $VM:$DATA/probe ~/Code/$REPO/outputs/vm_run/
```

The cache manifest is not rsync'd over the laptop's. Merge the VM's new rows into it with
`LeverageCache.merge_manifest`, which takes the cache's own lock and publishes atomically (with
the ordinary file mode), so it is safe even with a precompute running:

```bash
cd ~/Code/$REPO && .venv/bin/python -c "
import pandas as pd
from volsto.calibration.cache import LeverageCache
before, added = LeverageCache('cache').merge_manifest(pd.read_parquet('/tmp/vm_manifest.parquet'))
print(before, '+', added, '=', before + added, 'rows')"
```

The rows added are the VM's *new* keys, not the VM manifest's row count: a key the laptop
already had (a base leverage the VM reused) is not added again. On the laptop, merging the
17-leverage cache of the production measurement into the 288-row manifest added 16 rows
(288 + 16 = 304), because its base 1F leverage was already there. The added count must equal the
growth of `ls cache/*/leverage.npz | wc -l` from the rsync.

## 12. Check on arrival

```bash
cd ~/Code/$REPO && export NUMBA_NUM_THREADS=2
ls outputs/store/results/points | wc -l                                  # 131
ls cache/*/leverage.npz | wc -l                                          # before + the VM's new ones
.venv/bin/python -c "import pandas as pd, pathlib; m = pd.read_parquet('cache/manifest.parquet'); \
k = {p.parent.name for p in pathlib.Path('cache').glob('*/leverage.npz')}; \
print(len(k), 'leverages,', m.key.nunique(), 'manifest keys, missing from manifest:', len(k - set(m.key)))"
.venv/bin/python -c "from volsto.calibration.cache import LeverageCache as C; c = C('cache'); \
print('unlisted', c.unlisted_keys(), 'temporaries', len(c.temporaries()))"
.venv/bin/python -c "from volsto.viewers.store import ResultsStore; \
print(ResultsStore('outputs/store').refresh_manifest())"               # after two VMs, §13
.venv/bin/python -m volsto.viewers.precompute --grid configs/grids/default.yaml --dry-run --resume 2>&1 | grep '^resume:'
.venv/bin/python -m volsto.viewers.precompute report --store outputs/store   # -> outputs/store/cost_report/
.venv/bin/python -m volsto.viewers.app --check                             # 8 pages rendered
```

The dry run must print `resume: 131 points already done, 0 to compute`. Its projection table
still counts the 36 marking points as misses, because they are re-fitted when computed and have
no cache key at enumeration; that count is not a defect. "missing from manifest" must be 0
(and `unlisted` empty). A non-empty list names complete leverages whose manifest row was lost to
a kill between the leverage's rename and the manifest append. Those leverages are sound and are
used as hits; only the cost report lacks their calibration wall time. Leftover temporaries can be
deleted when no writer is running (`find cache -name '.*.tmp*' -delete`). Keep
`outputs/store/cost_report/summary.csv`: it holds the measured per-point cost by mode and step
(median, p90, core-seconds). Size the next grid from it with `--cost-from outputs/store`.

## 13. Known hazards

- **Concurrent manifest writers — fixed.** `LeverageCache._append_manifest`
  (`volsto/calibration/cache.py`) now takes an exclusive `flock` on `cache/manifest.parquet.lock`
  around its read-modify-write and publishes the new manifest by writing a temporary file in
  the cache root and `os.replace`-ing it, so no row is lost and no reader sees a torn file
  (`tests/test_cache_concurrency.py`: 12 writer processes × 30 rows with 3 readers polling —
  0 failures, 360/360 rows; the previous code, run three times on the same test, failed
  571–1,188 operations on torn or empty reads each time, and in the run whose final manifest
  could be read at all, 306 of the 360 rows were lost). The format is unchanged. The lock needs
  a local file system (§2).
- **Torn leverages on a kill — fixed.** Every cache file (`leverage.npz`, `spec.json`,
  `diagnostics.json`, the manifest) is written through `atomic_write`, with the ordinary
  `0666 & ~umask` mode. `leverage.npz` is the entry's commit point: `spec.json` is written before
  it, `diagnostics.json` and the manifest row after it. A kill leaves the old entry or none, plus
  at most one `.<name>.<hex>.tmp.<ext>` temporary in the cache root or the entry directory. The
  temporary is harmless and matched by `'.*.tmp*'`; delete it when no writer is running.
  `tests/test_cache_concurrency.py` SIGKILLs a writer halfway through its leverage. The pre-fix
  code, on the same scenario, reported the torn file as a hit whose load failed with
  `BadZipFile`.
- The VM's `cache/` is a copy. Never rsync its manifest over the laptop's (§11).
- A code-tag bump or a change in particle count makes a new grid, and every point misses.
- The run record keeps no absolute path. The store and cache are relocatable as a pair.
- `results/manifest.json` is a derived view, rewritten atomically at the end of every run. Two
  VMs write separate stores. The rsyncs of §11 merge their per-point files and run records (the
  names are distinct), but `results/manifest.json` ends up as the last-rsynced VM's copy.
  Nothing in volsto reads it; rebuild it from the merged files with
  `ResultsStore('outputs/store').refresh_manifest()` (§12).
