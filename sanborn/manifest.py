"""Read and write data/manifest.csv, one row per sheet (guide §3.3, Table 6)."""
import csv

from sanborn import ROOT

PATH = ROOT / "data" / "manifest.csv"
COLUMNS = [
    "sheet_id", "source", "item_id", "resource_id", "city", "county", "state", "lat", "lon",
    "year", "date_range_flag", "publisher", "sheet_kind", "jp2_url", "iiif_base",
    "width", "height", "bytes", "sha256", "color_frac", "scale_guess", "split",
    "n_buildings", "n_party_walls", "local_path", "status", "fetched_at",
]


def read(path=PATH):
    """All rows as dicts of strings (pd.read_csv(path) works too for analysis)."""
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def upsert(rows, path=PATH):
    """Add rows (dicts) or update existing ones with the same sheet_id; unknown keys are ignored."""
    table = {r["sheet_id"]: r for r in read(path)}
    for row in rows:
        rec = table.setdefault(row["sheet_id"], dict.fromkeys(COLUMNS, ""))
        rec.update({k: "" if v is None else v for k, v in row.items() if k in COLUMNS})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        w.writerows(table.values())
    tmp.replace(path)  # never leave a half-written manifest
