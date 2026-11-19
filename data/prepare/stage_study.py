"""Materialize the archived research scripts in a new, configurable working tree."""
import argparse
import csv
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path, help="New directory for the study layout")
args = parser.parse_args()
out = args.directory.resolve()
if out.exists():
    parser.error("Specify a new directory")
root = Path(__file__).resolve().parents[2]
with (Path(__file__).with_name("study_layout.tsv")).open() as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
for row in rows:
    dst = out / row["study_file"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    original = (root / row["package_file"]).read_text()
    dst.write_text(original.replace("@STUDY@", str(out)))
print(f"Staged {len(rows)} study files in {out}")
