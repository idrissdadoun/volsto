"""Add the ``sabrw`` section (the day's SABRW fits, SPEC §15 Part 3) to committed snapshots
imported before the importer stored it, leaving every other byte of the file unchanged.

For each ``spx_<date>.yaml`` of ``--dirs``: the day's quotes are filtered again from the vendor
file under ``--root`` (:func:`volsto.market.import_hdn.to_grid_surface`, the snapshot's own
filters), checked against the snapshot's provenance (the filters, the number of points and
expiries and the drop counts must be equal — otherwise the file was imported under other
numerics and the script stops), fitted per expiry
(:func:`~volsto.market.import_hdn.sabrw_fits`, ``t_min`` :data:`SABRW_T_MIN` to the last
retained expiry, as :func:`~volsto.market.import_hdn.import_day` does) and the section
(:func:`~volsto.market.import_hdn.sabrw_section`) appended.  A file that already has one is
left alone.  No surface, no calibration."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import yaml

from volsto.config import to_mapping
from volsto.market import import_hdn as ih
from volsto.market.vendor import HdnSource

ROOT = Path(__file__).resolve().parents[1]
DIRS = (
    ROOT / "configs" / "surfaces" / "snapshots",
    ROOT / "configs" / "surfaces" / "snapshots" / "hdn_2022H2",
    ROOT / "configs" / "surfaces" / "snapshots" / "hdn_2022H2_ssvi",
)


SECTION_START = "\nsabrw:\n"


def add_section(path: Path, source: HdnSource, *, replace: bool = False) -> int | None:
    """Append the section to ``path``; the number of fits, ``None`` when it already has one.
    With ``replace`` an existing section — which must be the file's last one — is fitted again
    and rewritten, every byte before it left as it is (after a change of the fitter:
    ``IMPORTER_TAG`` is bumped and the stored fits are regenerated, the market and surface
    sections are not)."""
    text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if "sabrw" in raw:
        if not replace:
            return None
        cut = text.rindex(SECTION_START) + 1
        head = yaml.safe_load(text[:cut])
        if "sabrw" in head or head != {k: v for k, v in raw.items() if k != "sabrw"}:
            raise SystemExit(f"{path}: the sabrw section is not the last one: not rewritten")
        text = text[:cut]
        raw = head
    prov = raw["provenance"]
    f = ih.HdnFilters()
    if to_mapping(asdict(f)) != prov["filters"]:
        raise SystemExit(f"{path}: imported under other filters: {prov['filters']}")
    chain = source.load_chain(prov["quote_date"], prov["underlying"])
    if source.day_file(prov["quote_date"]).name != prov["file"]:
        raise SystemExit(f"{path}: imported from {prov['file']}, not the source's day file")
    fwds = ih.implied_forwards(chain, max_years=f.max_years, band=f.near_atm_band)
    _, points = ih.to_grid_surface(chain, fwds, f)
    got = (len(points.table), int(points.table["expiry"].nunique()), dict(points.dropped))
    want = (prov["n_points"], prov["n_expiries"], dict(prov["dropped"]))
    if got != want:
        raise SystemExit(f"{path}: the quotes filter again to {got}, the snapshot has {want}")
    t_max = float(points.table["T"].max())
    fits = ih.sabrw_fits(points, t_min=ih.SABRW_T_MIN, t_max=t_max)
    block = yaml.safe_dump(to_mapping({"sabrw": ih.sabrw_section(fits)}), sort_keys=False)
    path.write_text(text + block, encoding="utf-8")
    return len(fits)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=[str(d) for d in DIRS])
    ap.add_argument("--root", default=str(ROOT / "data/hdn_sample/options_sample_2022H2"))
    ap.add_argument(
        "--replace",
        action="store_true",
        help="fit again and rewrite the sections that exist (after a change of the fitter)",
    )
    a = ap.parse_args()
    root = Path(a.root)
    source = HdnSource(root)
    t0 = time.perf_counter()
    done = skipped = 0
    for d in a.dirs:
        for path in sorted(Path(d).glob("spx_*.yaml")):
            n = add_section(path, source, replace=a.replace)
            if n is None:
                skipped += 1
                continue
            done += 1
            print(f"{path}: {n} fits", flush=True)
    wall = time.perf_counter() - t0
    print(f"{done} sections added, {skipped} files already had one; wall {wall:.0f} s")


if __name__ == "__main__":
    main()
