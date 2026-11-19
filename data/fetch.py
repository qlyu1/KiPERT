"""List raw data records or fetch a single checksummed Zenodo file."""
import argparse
import csv
import hashlib
import tempfile
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", help="Dataset ID from datasets.tsv")
parser.add_argument("--file", help="Exact raw filename to download")
parser.add_argument("--output", type=Path, default=Path("downloads"))
a = parser.parse_args()
with (Path(__file__).parent / "manifests/datasets.tsv").open() as f:
    records = [
        r
        for r in csv.DictReader(f, delimiter="\t")
        if not a.dataset or r["dataset"] == a.dataset
    ]
if not a.file:
    for r in records:
        print(r["dataset"], r["file_or_subset"], r["url"], sep="\t")
else:
    found = [r for r in records if r["file_or_subset"] == a.file and r["checksum"]]
    if len(found) != 1:
        parser.error("Choose one exact filename with a checksum in datasets.tsv")
    r = found[0]
    a.output.mkdir(parents=True, exist_ok=True)
    dst = a.output / r["file_or_subset"]
    if dst.exists():
        parser.error("Destination file exists")
    with tempfile.NamedTemporaryFile(dir=a.output, delete=False) as f:
        tmp = Path(f.name)
    try:
        algorithm, expected = r["checksum"].split(":")
        h = hashlib.new(algorithm)
        with urllib.request.urlopen(r["url"], timeout=120) as source, tmp.open(
            "wb"
        ) as target:
            while chunk := source.read(1 << 20):
                h.update(chunk)
                target.write(chunk)
        if h.hexdigest() != expected or tmp.stat().st_size != int(r["bytes"]):
            raise ValueError(
                "Downloaded file differs from the recorded checksum or size"
            )
        tmp.replace(dst)
        print(dst)
    finally:
        tmp.unlink(missing_ok=True)
