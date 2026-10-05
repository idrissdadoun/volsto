"""Dispersion study: the daily one-month at-the-money vol of the S&P 500 (spec §6, regime).

    python scripts/disp_spx.py

Reads the SPX per-ticker extract of the ORATS store one trading day at a time (one row group
per day) and writes ``outputs/dispersion/spx_vol.parquet``: ``sigma_spx_1m`` (the vendor's
smoothed vol at the strike nearest the spot, total variance interpolated to 30 calendar days
between the two bracketing expiries) and the SPX spot.  Regime indicator only; VIX is the
cross-check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from volsto.studies import disp_data as dd


def main() -> None:
    pf = pq.ParquetFile(dd.STORE / "orats" / "by_ticker" / "SPX.parquet")
    rows = []
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(
            g, columns=["trade_date", "expirDate", "strike", "stkPx", "smoothSmvVol"]
        ).to_pandas()
        if t.empty:
            continue
        day = str(t["trade_date"].iloc[0])[:10]
        spot = float(t["stkPx"].iloc[0])
        t["days"] = (
            pd.to_datetime(t["expirDate"].astype(str).str[:10]) - pd.Timestamp(day)
        ).dt.days
        t = t[(t["days"] >= 5) & (t["smoothSmvVol"] > 0)]
        if t.empty:
            continue
        t["dist"] = (t["strike"] - spot).abs()
        atm = t.loc[t.groupby("days")["dist"].idxmin(), ["days", "smoothSmvVol"]].sort_values(
            "days"
        )
        d, v = atm["days"].to_numpy(float), atm["smoothSmvVol"].to_numpy(float)
        w = np.interp(30.0, d, v * v * d)
        rows.append(
            {
                "date": day,
                "spx": spot,
                "sigma_spx_1m": float(np.sqrt(w / 30.0)) if d[0] <= 30 else float(v[0]),
            }
        )
    out = pd.DataFrame(rows).drop_duplicates("date").set_index("date").sort_index()
    out.to_parquet(dd.OUT / "spx_vol.parquet")
    med = out["sigma_spx_1m"].median()
    print(f"spx_vol: {len(out)} days {out.index[0]} to {out.index[-1]}; median {med:.3f}")


if __name__ == "__main__":
    main()
