"""Run figure-specific numerical checks and optionally save a fresh report."""
import argparse
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import scipy

from tutorial import run as tutorial
from evaluation import paper, transfer_interval
from experiments.fig3_double import run as double
from experiments.fig4_transfer import run as transfer


def single():
    pred, weights = tutorial.refit()
    return dict(
        shape=list(pred.shape),
        contexts=weights.context.nunique(),
        max_abs_error=tutorial.verify(pred),
    )


CHECKS = {
    "tutorial": single,
    "double": double.verify,
    "transfer": transfer.verify,
    "fig2": paper.figure2,
    "fig3": paper.figure3,
    "fig3h_refit": double.refit_figure3h,
    "fig4": paper.figure4,
    "fig4_interval": transfer_interval.check,
    "fig5": paper.figure5,
    "extended": paper.extended,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=CHECKS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = {}
    for name in args.only or CHECKS:
        start = perf_counter()
        results[name] = {
            "values": CHECKS[name](),
            "seconds": round(perf_counter() - start, 3),
        }
        label = (
            "PARTIAL"
            if name == "fig4_interval"
            and not all(
                results[name]["values"]["ratio_endpoints_match_at_printed_precision"]
            )
            else "PASS"
        )
        print(f"{label}  {name}  {results[name]['seconds']} s", file=sys.stderr)
    report = dict(
        status=(
            "CHECKS_COMPLETED_WITH_INTERVAL_DIFFERENCE"
            if "fig4_interval" in results
            and not all(
                results["fig4_interval"]["values"][
                    "ratio_endpoints_match_at_printed_precision"
                ]
            )
            else "PASS"
        ),
        system=platform.system(),
        machine=platform.machine(),
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        pandas=pd.__version__,
        checks=results,
    )
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
