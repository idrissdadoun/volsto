# Download-day runbook — the ORATS archive (M11)

The exact commands, in order, for the day the ORATS "Near End-of-Day" archive is bought. The
S3 credentials expire 14 days after purchase: steps 1–5 are the ones that need them, and they
fit in one day. Everything below was built and measured on the free one-day sample
(2024-01-03, 66.8 MB zipped); SPEC §18 holds the measurements.

State on 2026-10-03 (M11 Part 2b): every step below is built and measured on the sample. The
read API (Part 3) is merged; the SPX importer (Part 4) and the stored fit records are built and
await review, and the fit records must be merged before any backtest calibration on ORATS data
— the archive can be downloaded, verified and converted before those.

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

## 6. Convert

```bash
caffeinate -i .venv/bin/volsto-data convert --vendor orats
```

Reads each day's CSV straight from its zip and writes
`$VOLSTO_DATA_STORE/orats/strikes/year=YYYY/YYYY-MM-DD.parquet` (zstd 9, dictionary off, row
groups of 131,072, every vendor column, float64, the full universe), then
`store_manifest.json`. Needs the raw manifest of step 4; each zip's sha256 is checked again as
it is read. Refuses before writing if the store volume is short of 1.14 × the zips to convert
plus 10 GiB. Interrupted: run it again, finished days are skipped.

- Time (measured, 16 sample-sized days): 2.3 days/s on the default 8 workers → about 36 min
  for 4,970 days; 16 workers 29 min, 4 workers 55 min. If raw is on a spinning disk the read
  (300 GB at 150 MB/s ≈ 35 min) is the floor.
- Memory: 0.8–1.1 GiB per worker at the sample's size → 7–9 GiB on 8 workers.
- Disk: 1.14 × the zips: 229–400 GB for the 200–350 GB estimate. If raw and store share one
  disk, they need 430–750 GB together.
- Exit 1 lists every file that failed (`FAILED <file>: <reason>`) — schema drift, a value that
  does not parse, a `trade_date` mismatch, an SPX row without an OPRA symbol — and every day
  whose `cOpra` is not unique. A failed day is not in the store; send me the list.

## 7. Verify the store

```bash
caffeinate -i .venv/bin/volsto-data verify --vendor orats
```

Re-reads every raw file and its Parquet file: the raw sha256 against the manifest, the row
count, per column the null count and the sum, then the whole tables for equality. Lists raw
dates not converted and Parquet files without a manifest entry. Exit 0 only when every file is
a faithful copy.

- Time (measured): 5.3 days/s on 8 workers → about 16 min for 4,970 days.
- Memory: about 1.2 GiB per worker. Disk: none.

## 8. Extract

```bash
caffeinate -i .venv/bin/volsto-data extract --tickers SPX
```

One Parquet file per ticker across all dates, `$VOLSTO_DATA_STORE/orats/by_ticker/SPX.parquet`
(one row group per trade date), bound to the store manifest: after any later `convert` the
extract reads as stale in `volsto-data status` and the same command rebuilds it.

- Time (measured): 0.08 s per store day for three tickers → about 7 min for 4,970 days.
- Disk: SPX is about 1.1 MiB per day at the sample's size → about 5 GB for the archive.

## Afterwards

```bash
.venv/bin/volsto-data status
```

```bash
.venv/bin/volsto-data sql "select year, count(*) as rows, count(distinct ticker) as tickers from strikes group by year order by year"
```

`sql` is optional (DuckDB, the `data` extra); the view `strikes` is every day file plus `year`.

## External volumes

Raw and store may both sit on external volumes; nothing assumes the internal disk. Temporary
files are written beside their destination (same volume) and renamed. If a root points below
`/Volumes/<name>` and that volume is not mounted, every command refuses instead of creating the
directory on the internal disk. macOS's `._*` companion files on non-APFS volumes are ignored.
