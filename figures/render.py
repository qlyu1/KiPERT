"""Render one paper figure or all ten quantitative figures as PDFs."""
from pathlib import Path
import argparse
import subprocess
import sys

names = ("fig2", "fig3", "fig4", "fig5", "extended_data")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
parser.add_argument("--only", nargs="+", choices=names)
args = parser.parse_args()
for name in args.only or names:
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent / name / "plot.py"),
            str(args.output.resolve() / name),
        ],
        check=True,
    )
