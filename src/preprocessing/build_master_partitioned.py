from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\felip\camelscl-hydrology-ml")
PRO  = ROOT / "data" / "processed"
OUT_DIR = PRO / "master_parts"
OUT_DIR.mkdir(exist_ok=True, parents=True)

def _load(name: str) -> pd.DataFrame:
    p = PRO / name
    df = pd.read_parquet(p)
    if "basin_id" in df.columns:
        df["basin_id"] = df["basin_id"].astype(str)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def main():
    q    = _load("streamflow_mm.parquet")         # target
    p    = _load("precip_cr2met.parquet")
    tmin = _load("tmin_cr2met.parquet")
    tmax = _load("tmax_cr2met.parquet")
    pet  = _load("pet_hargreaves.parquet")
    attr = pd.read_parquet(PRO / "attributes.parquet")
    attr["basin_id"] = attr["basin_id"].astype(str)

    basins = q["basin_id"].unique().tolist()
    print(f"Basins: {len(basins)}")

    for i, b in enumerate(basins, 1):
        qb = q[q["basin_id"] == b]
        if qb.empty:
            continue
        dfb = (qb.merge(p[p["basin_id"] == b],    on=["basin_id","date"], how="left")
                 .merge(tmin[tmin["basin_id"]==b],on=["basin_id","date"], how="left")
                 .merge(tmax[tmax["basin_id"]==b],on=["basin_id","date"], how="left")
                 .merge(pet[pet["basin_id"]==b],  on=["basin_id","date"], how="left")
                 .merge(attr[attr["basin_id"]==b],on="basin_id",          how="left"))
        # limpieza mínima
        for c in ["prcp_mm_cr2met","pet_mm_hargreaves"]:
            if c in dfb.columns:
                dfb.loc[dfb[c] < 0, c] = pd.NA

        dfb = dfb.dropna(subset=["discharge_mm"]).sort_values("date")
        if dfb.empty:
            continue

        outp = OUT_DIR / f"master_{b}.parquet"
        dfb.to_parquet(outp, index=False)
        if i % 25 == 0:
            print(f"  {i}/{len(basins)} wrote {outp.name} rows={len(dfb):,}")

    print("✔ master parts escritos en", OUT_DIR)

if __name__ == "__main__":
    main()
