"""`uv run sheets`: every sheet, whether its image is in place, and how far its labeling is."""
import json

from sanborn import ROOT, manifest
from sanborn.labels import LABELS


def progress(sheet):
    """(status text, n_buildings or "") for one sheet, from its label folder."""
    d = LABELS / sheet
    edits_path, summary_path = d / "edits.json", d / "summary.json"
    if not (d / "regions.png").exists():
        return "no regions (not set up)", ""
    if not edits_path.exists():
        return "not started", ""
    edits = json.loads(edits_path.read_text())
    n_regions = len(json.loads((d / "regions.geojson").read_text())["features"])
    checked = len(set(edits.get("checked", [])) | set(map(int, edits.get("groups", {}))) | set(edits.get("drop", [])))
    minutes = round(edits.get("active_seconds", 0) / 60)
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        stale = edits_path.stat().st_mtime > summary_path.stat().st_mtime
        state = "dissolved, but edited since: run dissolve again" if stale else "dissolved"
        return f"{state} ({minutes} min)", summary.get("buildings", "")
    if not edits.get("extents"):
        return f"started, no extent yet ({minutes} min)", ""
    return f"in progress: ~{min(checked, n_regions)} of {n_regions} regions touched ({minutes} min)", ""


def main():
    rows = manifest.read()
    print(f"{'sheet':26s} {'city':16s} {'year':8s} {'image':8s} {'buildings':9s}  status")
    for r in rows:
        sheet = r["sheet_id"]
        image = "ok" if (ROOT / "data" / "raw" / f"{sheet}.jp2").exists() else "MISSING"
        status, n = progress(sheet)
        print(f"{sheet:26s} {r['city']:16s} {r['year'][:7]:8s} {image:8s} {str(n):9s}  {status}")
    if any(not (ROOT / "data" / "raw" / f"{r['sheet_id']}.jp2").exists() for r in rows):
        print("\nMISSING image: put that sheet's .jp2 file into the data/raw folder (README.md, step 3).")


if __name__ == "__main__":
    main()
