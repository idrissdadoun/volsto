# M11 Part 0 — the ORATS one-day sample, inspected (2026-10-03)

Sample: `ORATS_SMV_Strikes_20240103.zip` (trade date 2024-01-03), downloaded into
`data/orats_sample/` (git-ignored). No library code written. The analysis ran in a cloud
session on throw-away scripts (parity regression as `import_hdn.implied_forward`, Black
inversion as `volsto.market.bs.implied_vol`); every number below is reproducible from the
sample with those two functions. Wall clock for the whole part: ≈ 45 min (download 12 s; each
analysis pass 10–30 s). No Monte Carlo, no calibration. The owner's decisions are recorded at
the end.

**Branch state.** At the time of writing `origin/fix/test-portability` (fd48063, "Tests:
portable across machines") was not yet merged into `main`; `feature/m11-orats-store` was
started from it, so it becomes a descendant of `main` once the merge lands (owner's decision 1).

**Where the sample contradicts the ORATS page**
1. The file has 39 columns, not 36: `cOpra`, `pOpra` (OPRA symbols, after `ticker`) and
   `cMidIv` (between `cBidIv` and `cAskIv`) are extra; the published 36 are all present, in order.
2. `stkPx` for an index is **not** the implied futures price. It is the parity forward
   discounted at `iRate`: `stkPx = F_parity · exp(−iRate · yte)` within ±3 bp to 1y on SPX.
3. `spot_px` is empty on 94.7 % of rows (5740 of 5787 tickers; every single stock), and where
   present on indices it is not the close and not the snapshot level (SPX 4674.89 against a
   close of 4704.81 and a snapshot level of 4709.55, below the day's low; VIX 16.18 against a
   VIX close of 14.04). Unusable as a spot.
4. `divRate` is 0 on every row. `residualRateData` carries whatever carry ORATS fitted.
5. Snapshot time: nothing in the file states it (no timestamp column; the zip member is dated
   2024-01-04 02:11). "14 minutes before the close" cannot be verified from the sample.

## 1. Structure

| item | value |
|---|---|
| zip | 66,808,802 B (63.7 MiB), one member, deflate; sha256 `e04a3731…b514570` |
| unzipped | 222,935,113 B (212.6 MiB), `ORATS_SMV_Strikes_20240103.csv` |
| rows | 716,822 (+ header); 1,433,644 contracts (call and put per row) |
| tickers | 5,787; expiries 80 (Fri 41, Wed 14, Thu 11, Tue 8, Mon 6); (ticker, expiry) groups 31,250 |
| delimiter | comma, no quoting needed, LF line ends, no BOM, header row |
| naming | `ORATS_SMV_Strikes_YYYYMMDD.{zip,csv}` |
| dates | `expirDate`, `trade_date` as `M/D/YYYY` (no zero padding) |
| strings | `ticker` (1–5 upper-case letters; share classes as `BRK_B`, `BF_A`, …), `cOpra`, `pOpra` (OSI symbols, 21 characters) |
| text precision | strike 2 dp; prices 2 dp; `yte` 5 dp; IVs 6 dp; Greeks 8 dp; `iRate` 3–4 dp; `extPTheo` in scientific notation on some rows |

Inferred dtypes: float64 for `stkPx yte strike cBidPx cValue cAskPx pBidPx pValue pAskPx cBidIv
cMidIv cAskIv smoothSmvVol pBidIv pMidIv pAskIv iRate residualRateData delta gamma theta vega
rho phi driftlessTheta extVol extCTheo extPTheo spot_px`; int64 for `cVolu cOi pVolu pOi divRate`
(divRate only because every value is 0 — store it as float64); strings for the rest.

Nulls: only `spot_px` (678,951 null). Zero counts (of 716,822):

| column | zeros | note |
|---|---|---|
| cBidPx / pBidPx | 150,883 / 191,986 | both zero 7,144; either 335,725 (46.8 %) |
| cAskPx / pAskPx | 0 / 10 | the 10 are penny names (ATNF, EAR); one put bid > ask (ATNF) |
| cBidIv / pBidIv | 387,949 / 417,371 | zero bid → zero IV (54 % / 58 %) |
| cMidIv / pMidIv | 14,867 / 18,852 | |
| cVolu / pVolu | 599,189 / 621,057 | |
| cOi / pOi | 354,006 / 419,702 | both zero 283,954 (39.6 %) |
| yte | 1,334 | 0DTE rows (13 tickers, incl. SPX/SPXW, NDX, RUT, VIX); 392 have cMidIv 0 |
| residualRateData | 149,609 | 405,901 negative, 161,312 positive |
| delta / gamma | 41,947 / 98,666 | |
| smoothSmvVol | 0 | min 0.01, 7 rows at the 9.999999 cap |
| extPTheo | 18,275 | one row at 2.3e15 (CHRS 2026-01-16, strike 2) |

Volumes/OI range 0–462,484 / 0–405,736 (int64). Prices: stkPx 0.02–16,444.7, strike 0.25–22,000.

Per ticker: expiries median 4 (max 55 SPX), rows median 71 (SPX 10,519; NDX 9,658; XSP 5,393;
QQQ 4,255; SPY 3,894; RUT 3,608). 20 tickers carry a per-expiry `stkPx` (the cash-settled
indices: SPX, XSP, NDX, RUT, XEO, OEX, VIX, DJX, BKX, OSX, XAU, RUI, XD* FX indices, UTY); every
other ticker has one `stkPx` (the stock print at the snapshot).

**Archive extrapolation.** 2007-01-03..2026-10-03 ≈ 4,970 trading days. At this day's size the
archive is 332 GB zipped / 1.11 TB of CSV. Older years are smaller (2007: no weeklies on most
names, no 0DTE, far fewer strikes), recent days larger than this early-January day. Estimate:
**200–350 GB zipped, 0.65–1.2 TB unzipped**, uncertain by ±40 %; the `aws s3 ls --summarize`
dry run on download day gives the exact byte count before anything is fetched. Never keep the
CSVs unzipped: the raw layer stays zipped (as delivered). Parquet of this day, typed and sorted:
zstd-3 99.1 MB, zstd-9 91.2 MB, snappy 121 MB (1.4–1.5× the zip; the two OPRA string columns are
the cost — Part 2 will measure dictionary encoding). Disk plan: raw ≈ 200–350 GB on the external
volume plus the second copy; store ≈ 300–500 GB internal.

## 2. S&P 500 index options and AM/PM

- Tickers: `SPX` (10,519 rows, 55 expiry dates, 60 (root, expiry) slices) and `XSP` (mini,
  5,393 rows, 52 expiries, root XSP). No `SPXW`, `SPXPM` ticker. `ES` is Eversource (stkPx
  64.03), not the future. SPX strikes 200–7,000, step 5 near the money (364 strikes on the
  2024-02-16 PM slice); longest expiry 2029-12-21.
- **The ticker column merges the roots**: ticker `SPX` has `cOpra` roots `SPX` (3,278 rows)
  and `SPXW` (7,241 rows). The OPRA root is the only AM/PM marker in the file.
- Duplicates on (ticker, expirDate, strike): **1,590 keys, 3,180 rows, all SPX**, exactly on
  the five third Fridays where both roots list (2024-01-19, 02-16, 03-15, 04-19, 05-17). On
  (root, expirDate, strike): **0** duplicates in the whole file.
- `yte` per (ticker, expirDate): **one value everywhere** (0 groups with two). The AM root
  (SPX) and the PM root (SPXW) on 2024-01-19 both carry `yte` = 0.04384 = 16/365.
- The quotes differ as they should: on 2024-01-19 at the same strikes (±3 %) the PM call mid is
  0.95 above the AM and the PM put mid 0.90 above; the ATM straddle ratio squared gives
  T_AM / T_PM = 0.949, i.e. the AM root prices as **0.81 days shorter** on 16 days. ORATS's
  own put IVs confirm it: inverting the AM root's put mids with T = yte − 1 day reproduces
  `pMidIv` to 0.08 vp, with T = yte to 0.49 vp; for the PM root T = yte reproduces them to
  0.08 vp (see §3). ORATS fits a separate forward per root (`stkPx` 4709.40 AM vs 4708.93 PM
  on 01-19) and a separate smooth vol (0.1128 vs 0.1120 ATM). Liquidity 01-19: AM 106k
  volume / 2.67 M OI, PM 23k / 398k.
- **How AM/PM is recovered:** by the OPRA root in `cOpra`/`pOpra`, exactly as the HDN path
  groups by (root, expiration): `SPX` → AM (T = calendar days − 1), `SPXW` → PM (T = calendar
  days), the 2022 H2 HDN convention. It **cannot** be recovered from `ticker`, `expirDate` or
  `yte`. Risk: the published column list omits `cOpra`/`pOpra`; whether every year of the
  archive carries them, and what root the pre-2010 weeklies used, must be asked (§6). The
  importer must refuse a file without the OPRA columns rather than guess by weekday.

## 3. `yte`

`yte = round(calendar days from trade_date to expirDate / 365, 5)` on all 80 expiries
(max |yte − days/365| = 4.8e-6, i.e. rounding), the same value for every ticker and both
roots, 0 on the trade date itself. No time-of-day fraction, no business days, no AM
adjustment, no snapshot-time offset. Internally ORATS uses one day less for the AM root (§2)
but prints the same `yte`. Decision for Part 4 (proposed): compute T ourselves from the dates
and the root, as HDN does — `yte` is a cross-check only.

Vendor IV convention (median |our Black inversion − vendor IV|, vol points, near the money):
no single convention reproduces both sides. With S = `stkPx`, r = `iRate`, q = 0, T = `yte`:
puts 0.08–0.13 vp (≤ 2 m, PM root), 0.22–0.26 vp at 6–12 m; calls 0.24–0.49 vp at ≤ 2 m rising
to 1.3 vp (6 m) and 2.1 vp (1 y) — and `cMidIv − pMidIv` at the same strike is −0.24 vp (2 w)
to −1.95 vp (1 y), so the vendor's call and put IVs are not parity-consistent with each other.
With S = `spot_px` the error is 2–5 vp. Consequence: the vendor IVs cannot define the bid/ask
spread filter at the HDN precision; Part 4 should invert bid and ask prices itself (the HDN
code already does when `iv_bid`/`iv_ask` are absent) and keep `cBidIv`…`pAskIv` as
cross-checks. Open question for the owner.

Vendor Greeks against Black–Scholes(S = stkPx, r = iRate, q = 0, σ = smoothSmvVol or cMidIv):
delta within 0.002–0.03, gamma within 1e-5, vega 3–10 % off — their own model, cross-checks only.

## 4. `stkPx` against the cash and the parity forward; rates

Our parity forward (regression of C − P on K within ±10 %, mids, both bids > 0; as
`import_hdn.implied_forward`) versus `stkPx`, SPX:

| expiry (root) | days | pairs | stkPx | F_parity ± se | F / stkPx − 1 | F / (stkPx·e^{rT}) − 1 |
|---|---|---|---|---|---|---|
| 2024-01-04 (W) | 1 | 82 | 4709.55 | 4710.74 ± 0.005 | +2.5 bp | +1.0 bp |
| 2024-01-19 (SPX) | 16 | 176 | 4709.40 | 4720.01 ± 0.010 | +22.5 | −1.8 |
| 2024-01-19 (W) | 16 | 180 | 4708.93 | 4720.13 ± 0.012 | +23.8 | −0.6 |
| 2024-02-16 (SPX) | 44 | 180 | 4705.83 | 4735.55 ± 0.005 | +63.2 | −2.9 |
| 2024-03-15 (SPX) | 72 | 166 | 4700.25 | 4749.95 ± 0.005 | +105.7 | −2.5 |
| 2024-06-21 (SPX) | 170 | 112 | 4689.65 | 4801.13 ± 0.010 | +237.7 | −9.1 |
| 2024-12-20 (SPX) | 352 | 38 | 4663.18 | 4882.47 ± 0.029 | +470.3 | −3.4 |
| 2025-12-19 (SPX) | 716 | 9 | 4580.36 | 5006.97 ± 0.18 | +931 | +41 |
| 2028-12-15 (SPX) | 1808 | 9 | 4280.77 | 5399.51 ± 0.63 | +2613 | +382 |

So `stkPx` = parity forward × e^{−iRate·yte} to within 3 bp through 1 y (the −9 bp at 6 m and
the 2024-12-31/2025-01-17 slices at +11/+20 bp sit where SPXW strikes are sparse), and
diverges beyond 2 y where the chain has 8–9 two-sided pairs. Same on XSP (±3 bp to 3 m, −8 to
−35 bp at 6–12 m with 60–90 pairs), NDX (±1 bp to 1 m, −9 to −18 bp at 4–7 m), RUT (±2 bp to
2 m, −8 to −33 bp at 5–12 m). For VIX, `stkPx` per expiry is the futures discounted
approximately (−1 bp at 7 d to −98 bp at 231 d). Spot at the snapshot: SPX 0DTE `stkPx`
4709.55, 1-day parity forward 4710.74 ± 0.005 (82 pairs) — +12.6 bp over the official close
4704.81 (close from memory; the owner's `data/history/SPX.csv` is the authority). The 16:00
close against a 15:46 snapshot is the mirror image of the HDN asynchrony (16:00 close against
16:15 quotes), so the importer's spot-asynchrony check applies with the sign of the last 14
minutes.

`spot_px`: present on 47 index tickers only, one value per ticker. SPX 4674.89 (−63.6 bp vs the
close, −76 bp vs the 1-day forward), XSP 469.54 (−33 bp), NDX 16379.63 (−3.5 bp), RUT 1950.59
(−55 bp), VIX 16.18 (+15 % against a 14.04 close; the 0DTE stkPx is 13.85). Not a usable spot;
its definition goes to ORATS (§6).

Single stocks (one `stkPx` per ticker = the stock print): parity forward against
`stkPx·e^{iRate·yte}`, in bp —

| ticker | stkPx | 2 d | 16 d | 44 d | 72 d | 170 d | 261–380 d | note |
|---|---|---|---|---|---|---|---|---|
| AAPL | 184.24 | +1.9 | −2.9 | −11.6 | −19.4 | −41.3 | −45.5 (261 d) | dividend yield ≈ 0.5 % |
| JPM | 171.55 | −37.9 | −58.6 | −65.9 | −75.7 | −135.5 | −232 (380 d) | $1.05 ex-div 2024-01-05 (61 bp) — the 2-day parity already sits 35 bp below stkPx: early-exercise/dividend |
| XOM | 103.52 | +1.8 | −1.3 | −59.3 | −95.2 | −179 | −169 (198 d) | $0.95 ex-div 2024-02-13 (92 bp) appears between 37 d and 44 d |

`iRate` is a step function of `yte` with eight values that are the 2024-01-03 Treasury
constant-maturity yields (1 m 5.55, 3 m 5.46, 6 m 5.24, 1 y 4.80, 2 y 4.33, 3 y 4.09, 5 y 3.93,
7 y 3.95 %); the step boundaries are consistent with "the first CMT tenor ≥ yte" (yte 0–0.077 →
1 m, 0.079–0.233 → 3 m, 0.288–0.485 → 6 m, 0.537–0.995 → 1 y, 1.04–1.96 → 2 y, 2.04–2.96 →
3 y, 3.96–4.95 → 5 y, 5.97 → 7 y). The parity-implied funding rate (`−ln DF / T`) on SPX is
5.0–6.0 % at 1–6 m against 5.46–5.55 % Treasury, i.e. the usual box spread. `divRate` = 0 on
every row. `residualRateData` is constant within 21,333 of the 31,250 (ticker, expiry) groups
and varies by strike in the rest (mostly long expiries); on SPX it is −0.9 % to −1.4 % per year
from 2 weeks to 1 y (0 on three slices), the sign and size of a dividend yield but not equal to
the one implied against any spot in the file (−q_impl against the snapshot spot is −0.7 % at
44 d, −1.1 % at 1 y). Its definition goes to ORATS. Neither column is needed: Part 4 takes the
funding curve and the carry from parity, as HDN does.

## 5. Coverage

Present: SPX (55 expiries), XSP (52), NDX (47, one root), RUT (27, one root), VIX (13 Wednesday
expiries, one root; VIX 0DTE stkPx 13.85), SPY (34), QQQ (31), IWM (29), DJX (10), and all ten
basket names of `data/history/` (AAPL 20 expiries, MSFT 19, AMZN 17, NVDA 19, JPM 17, XOM 16,
JNJ 14, PG 14, HD 15, UNH 13). Absent (no listed options): VIX3M, VVIX, SKEW, COR1M, COR3M.
No NDXP / RUTW / VIXW roots. `data/history/` itself is not in this container (it is
git-ignored and lives in the owner's local archive); the names come from
`scripts/fetch_history.py` on `origin/barrier-vanilla-dispersion`.

## 6. What the sample cannot answer — questions for ORATS

1. Column history since 2007: when `cOpra`/`pOpra`/`cMidIv` appear, whether any column was
   added, renamed or re-typed, and whether `spot_px` is populated in other years (Part 2's
   schema-version registry depends on this).
2. The OPRA root of pre-2010 SPX weeklies (pre-OSI symbology) and of the 2011–2015 `SPXPM`
   third-Friday PM series; whether the ticker column has always merged the roots.
3. The S3 layout: bucket, prefix, one object per day or per year, naming (`ORATS_SMV_Strikes_
   YYYYMMDD.zip`?), object count and total bytes; whether holidays/half-days have files.
4. Whether delisted tickers remain in the daily files of their listed period (survivorship),
   and ticker renames (share classes as `BRK_B`).
5. The snapshot time (14 minutes before the close? one timestamp for all rows? same on
   half-days?) and the clock source of the stock prints (`stkPx`); what `spot_px` is and why
   it is below the day's range for SPX and above for VIX.
6. Definitions of `residualRateData`, `cValue`/`pValue` (equal to our mids within 0.00–0.01 on
   SPX), `extVol`/`extCTheo`/`extPTheo`, the 9.999999 IV cap, and the `iRate` tenor rule.
7. Early-exercise handling in their IVs for American names (the JPM 2-day parity gap).

## Owner decisions on Part 0 (2026-10-03)

1. `fix/test-portability` is merged before Part 1 starts.
2. **Time to expiry:** from (trade date, expiry date, OPRA root) with the HDN rule — calendar
   days / 365, one day less for the AM root (`SPX`); `yte` is a cross-check only.
3. **Spread filter:** invert bid and ask prices ourselves with our implied forward and discount
   factor; the vendor IVs stay cross-checks. The HDN path keeps its current filter
   (byte-identical rule). The SPEC section records the difference between the two vendors'
   filters.
4. **Types:** the store schema is declared in code, column by column; nothing is inferred from
   a file. `divRate` is float64.
5. **Spot:** never use `spot_px`. Spot is the option-implied spot the importer already
   computes. The official close in `data/history/SPX.csv` is the fixing reference only. The
   10 bp / 3 se asynchrony warning was written for HDN; for ORATS, implied spot minus close is
   reported as a measurement, not as a data problem.
6. **`stkPx`:** cross-check only. Compare our parity forward with `stkPx·exp(iRate·T)` per
   expiry, in bp.
7. **Raw stays zipped.** `verify-raw` hashes the zip; the converter reads from it.
8. **Store size:** Parquet at 1.5× the zip is too large for a full-universe store. In Part 2,
   before fixing the layout, measure on the sample day and report size and read time for zstd
   levels 3, 9 and 15, BYTE_STREAM_SPLIT on float columns, and dictionary on and off; and for
   the full universe against an "indices + ETFs + chosen names" subset. Every column and exact
   values are kept: no float32, no rounding. The owner chooses after seeing the table.
9. **OPRA dependency:** the converter fails loudly on any file where `cOpra` or `pOpra` is
   missing or empty for SPX, and `verify` reports per year whether those columns are
   populated. The fallback (AM/PM from the expiry calendar) is written as a plan only.
10. **Location:** Parts 1 to 4 run on the owner's Mac in `~/Code/volsto`, not in a cloud
    container; the local session re-downloads the sample
    (`https://s3.amazonaws.com/assets.orats.com/ORATS_SMV_Strikes_20240103.zip` →
    `data/orats_sample/`, sha256 `e04a37310b0f453fedc03a000c55adfc517a119279150173916ebdc5fb514570`).
