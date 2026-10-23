#!/usr/bin/env python3
"""Build aircraft mappings and coverage from the package's BTS snapshots."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FLIGHTS = ROOT / "raw" / "flights"
REFERENCES = ROOT / "references"
MONTHS = ROOT / "tables" / "flight_months.csv"

def read_flights(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        csv_name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        with archive.open(csv_name) as stream:
            return pd.read_csv(stream, usecols=["Reporting_Airline", "Tail_Number"], dtype=str)


def main() -> None:
    month_index = pd.read_csv(MONTHS, dtype={"month": str})
    monthly = []
    for _, record in month_index.iterrows():
        month = record["month"]
        path = FLIGHTS / f"{month.replace('-', '_')}.zip"
        flights = read_flights(path)
        flights["month"] = month
        monthly.append(flights)
    flights = pd.concat(monthly, ignore_index=True)
    flights["Tail_Number"] = flights["Tail_Number"].str.strip()
    flights["Reporting_Airline"] = flights["Reporting_Airline"].str.strip()

    maps: dict[int, pd.DataFrame] = {}
    for year in (2024, 2025):
        ref = pd.read_csv(REFERENCES / f"aircraft_reference_{year}.csv", dtype=str).fillna("")
        ref = ref.drop_duplicates(["airline", "tail_number"], keep="last")
        maps[year] = ref[["airline", "tail_number", "aircraft_model", "manufacturer"]]

    output_rates = []
    unmatched = []
    used_models: set[str] = set()
    for year, ref in maps.items():
        joined = flights.merge(ref, left_on=["Reporting_Airline", "Tail_Number"], right_on=["airline", "tail_number"], how="left", validate="many_to_one")
        matched = joined["aircraft_model"].fillna("").str.strip().ne("")
        for month, group in joined.groupby("month", sort=True):
            valid_tail = group["Tail_Number"].fillna("").str.strip().ne("")
            match = group["aircraft_model"].fillna("").str.strip().ne("")
            output_rates.append({
                "inventory_year": year, "month": month, "flight_rows": len(group),
                "rows_with_tail": int(valid_tail.sum()), "matched_flight_rows": int(match.sum()),
                "flight_row_match_rate": round(match.sum() / len(group), 6),
                "unique_airline_tail_pairs": int(group.loc[valid_tail, ["Reporting_Airline", "Tail_Number"]].drop_duplicates().shape[0]),
                "matched_unique_airline_tail_pairs": int(group.loc[match, ["Reporting_Airline", "Tail_Number"]].drop_duplicates().shape[0]),
            })
        miss = joined.loc[~matched & joined["Tail_Number"].fillna("").str.strip().ne(""), ["month", "Reporting_Airline", "Tail_Number"]].drop_duplicates()
        miss.insert(0, "inventory_year", year)
        unmatched.append(miss)
        used_models.update(joined.loc[matched, "aircraft_model"].str.strip().unique())

    aliases = pd.read_csv(REFERENCES / "aircraft_type_aliases.csv", dtype=str).fillna("")
    if aliases.aircraft_model_raw.duplicated().any():
        raise ValueError("aircraft_type_aliases.csv contains duplicate raw model labels")
    missing = sorted(used_models - set(aliases.aircraft_model_raw))
    if missing:
        raise ValueError(f"Add new observed labels to aircraft_type_aliases.csv first: {missing}")
    model_table = aliases.loc[aliases.aircraft_model_raw.isin(used_models)].copy()
    model_table = model_table.rename(columns={"aircraft_model_raw": "aircraft_model"})
    model_table.to_csv(REFERENCES / "aircraft_model_mapping.csv", index=False)
    pd.DataFrame(output_rates).to_csv(REFERENCES / "aircraft_monthly_match_rates.csv", index=False)
    pd.concat(unmatched, ignore_index=True).to_csv(REFERENCES / "aircraft_unmatched_tails.csv", index=False)
    print(f"models={len(model_table)} rows={len(flights)} rates={len(output_rates)} unmatched={sum(len(x) for x in unmatched)}")


if __name__ == "__main__":
    main()
