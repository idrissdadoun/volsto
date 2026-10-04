# Download-day runbook — the ORATS archive (M11)

The exact commands, in order, for the day the ORATS "Near End-of-Day" archive is bought. The
S3 credentials expire 14 days after purchase: steps 1–5 are the ones that need them, and they
fit in one day. Everything below was built and measured on the free one-day sample
(2024-01-03, 66.8 MB zipped); SPEC §18 holds the measurements.

State on 2026-10-03 (M11 Part 1): steps 0–5 are built. Steps 6–8 (`convert`, `verify`,
`extract`) land with Part 2b; their commands, times and disk are filled in then and are marked
**pending** here — do not buy before that part is merged.

Conventions:

- Run from `~/Code/volsto` with the project environment (`.venv/bin/volsto-data`, or activate
  `.venv`).
- Long steps run under `caffeinate -i` so the Mac does not sleep.
- `volsto-data` never sees a credential: it passes the *name* of an AWS profile to the AWS
  CLI. Never paste a key into a command, a file of this repository or a chat.
- Exit codes: 0 clean, 1 a check found something (read the `FINDING` lines), 2 a refusal.
- Sizes: the archive is estimated at 200–350 GB zipped (±40 %, extrapolated from one day);
  step 2 prints the exact byte count before anything is downloaded.

## 0. Once, before the purchase

```bash
brew install awscli
```

The AWS CLI is not installed on this Mac as of 2026-10-03 (`volsto-data fetch` refuses with
this hint when it is absent).

Decide where the two roots live and export them in the shell you will use (add them to
`~/.zshrc` to make them permanent). Raw may sit on an external volume; the store should stay
on the internal disk (it is read at random).

```bash
export VOLSTO_DATA_RAW=/Volumes/<external>/volsto-raw     # default: data/raw
export VOLSTO_DATA_STORE=$HOME/Code/volsto/data/store      # default: data/store
```

```bash
.venv/bin/volsto-data status
```

Prints both roots, where each came from (default / env / flag) and the free space on their
volumes.

## 1. Configure the profile (your step)

```bash
aws configure --profile orats
```

Type the access key and secret ORATS sends you at the prompts. This writes
`~/.aws/credentials`; nothing in volsto reads that file.

## 2. Dry run — list the bucket, write nothing

Bucket, prefix and file naming come with the purchase; fill them in.

```bash
.venv/bin/volsto-data fetch --vendor orats --profile orats --bucket <bucket> --prefix <prefix> --dry-run
```

Prints the object count, the total bytes, the first and last keys, the bytes still to download
and the free space on the raw volume, and warns if the sync would be refused for lack of space
(10 GiB margin). Expected: about 4,970 objects (one per trading day 2007-01-03 → today) and
200–350 GB. Time: seconds. Disk: none.

Check before going on:

- the keys look like `ORATS_SMV_Strikes_YYYYMMDD.zip`. If the naming differs, note the regular
  expression for step 4 (`--pattern`, group 1 = `YYYYMMDD`);
- if the prefix holds other products, restrict the sync with `--include '*.zip'` (or a narrower
  glob);
- if the bucket is requester-pays, append `-- --request-payer requester`.

## 3. Sync

```bash
caffeinate -i .venv/bin/volsto-data fetch --vendor orats --profile orats --bucket <bucket> --prefix <prefix>
```

Lists again, refuses if the raw volume is short, then runs
`aws s3 sync s3://<bucket>/<prefix>/ $VOLSTO_DATA_RAW/orats/` (never `--delete`). Interrupted or
failed: run the same command again; objects already complete are skipped.

- Time: bounded by the line. Measured here on the sample over public HTTPS, one stream:
  66.8 MB in 3.05 s (21.9 MB/s) → 2.5–4.4 h for 200–350 GB at that rate; `aws s3 sync` runs 10
  requests in parallel by default, so expect less if the line allows. Not measured against the
  real bucket.
- Disk: the listed total on the raw volume (200–350 GB).

## 4. Verify raw

```bash
caffeinate -i .venv/bin/volsto-data verify-raw --vendor orats
```

Hashes every zip, streams every CSV from inside its zip (nothing is unzipped to disk), writes
`$VOLSTO_DATA_RAW/orats/raw_manifest.json` and prints: files, bytes, rows, the schema versions
seen, every missing / unexpected / duplicated trading day against `data/history/SPX.csv`, every
`trade_date` mismatch, every schema drift with the columns named, every file without OPRA
symbols on SPX, and the OPRA coverage per year. Exit 0 only when all of it is clean.

- Time: measured 0.45 s per sample-sized file on one core, 3.1 s for 32 files on 8 workers
  (10 files/s; internal SSD) → about 8 min for 4,970 files if the disk keeps up; an external
  spinning disk at 150 MB/s needs 25–40 min to be read once. Peak memory 0.4–0.5 GB in total.
- Disk: the manifest only (a few MB).
- Re-running reuses entries whose size and modification time are unchanged; `--rehash` reads
  everything again.

Expect findings on the real archive (this is the step that answers the open questions of
SPEC §18): send me the output. In particular **schema drift** (older years with fewer columns)
and **files without OPRA symbols on SPX** stop the importer for those dates until a schema
version or the AM/PM fallback is agreed. If the calendar file ends before the last file
("files dated beyond the calendar"), refresh `data/history/SPX.csv` and re-run.

## 5. Second copy of raw on another disk

```bash
caffeinate -i rsync -a --progress "$VOLSTO_DATA_RAW/orats/" /Volumes/<other>/volsto-raw/orats/
```

Then verify the copy by content, not by size or date — every file is hashed again and compared
by name and sha256 with the manifest of step 4:

```bash
caffeinate -i .venv/bin/volsto-data --raw /Volumes/<other>/volsto-raw verify-raw --vendor orats --against "$VOLSTO_DATA_RAW/orats/raw_manifest.json"
```

Exit 0 and "identical to the reference manifest" means the copy is good. A missing, extra or
different file is listed as a `FINDING`. (The manifest travels with the files: `rsync` copied
it too, and the scan reads `*.zip` only, so it is never taken for a raw file.)

- Time: the copy is bounded by the slower disk (300 GB at 150 MB/s ≈ 35 min; at 500 MB/s ≈
  10 min); the verification as step 4.
- Disk: the archive size again, on the other disk.

After this step the credentials are no longer needed.

## 6. Convert — pending Part 2b

`volsto-data convert --vendor orats`: one Parquet file per trading day under
`$VOLSTO_DATA_STORE/orats/strikes/year=YYYY/`, read straight from the zips. Layout, time, peak
memory and disk are measured in Part 2a/2b (Part 0 measured 99 MB per day at zstd-3, 1.5 × the
zip, which is what Part 2a sets out to reduce).

## 7. Verify the store — pending Part 2b

`volsto-data verify --vendor orats`: row counts, per-column null counts and sums equal between
raw and Parquet, for every file.

## 8. Extract — pending Part 2b

`volsto-data extract --tickers SPX,…`: one Parquet per ticker across all dates, for backtests.
