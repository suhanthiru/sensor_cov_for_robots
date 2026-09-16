"""Draw every figure in the report from the sweeps already on disk.

    python -m sensorcov.run_figures

Reads results/*.npz, so it is cheap to re-run while iterating on the plots and
does not touch the ray casting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .machine import MachineSpec
from .sensors import Layout
from .study import LAYOUTS, RESULTS, markdown_table
from .sweep import SweepResult
from .viz import (FIGURES, blind_maps, coverage_distributions, detection_polar,
                  machine_view, pareto_plot)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--figures", default=str(FIGURES))
    ap.add_argument("--layouts", nargs="*", default=list(LAYOUTS))
    args = ap.parse_args(argv)

    results = Path(args.results)
    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)

    sweeps = {}
    for name in args.layouts:
        path = results / f"{name}.npz"
        if not path.exists():
            print(f"  {name}: no sweep on disk, skipping")
            continue
        sweeps[name] = SweepResult.load(path)
    if not sweeps:
        print("nothing to draw; run python -m sensorcov.study first")
        return 1

    with open(results / "study.json") as fh:
        rows = json.load(fh)["rows"]

    print("drawing:")
    pareto_plot(rows, figures / "pareto.png")
    coverage_distributions(sweeps, figures / "coverage_distributions.png")
    detection_polar(sweeps, figures / "detection.png")
    blind_maps(sweeps, figures / "blind_maps.png")

    m = MachineSpec.load()
    # The two layouts the report argues about: the one with the most persistent
    # blind volume and the one that trades it for transient.
    for name, kind in (("a_cab_corners", "persistent"), ("d_cab_and_boom", "transient")):
        if name in sweeps:
            machine_view(m, Layout.load(name), sweeps[name],
                         path=figures / f"machine_{name}_{kind}.png", mask_kind=kind)

    with open(figures.parent / "results" / "table.md", "w") as fh:
        fh.write(markdown_table(rows))
    print(f"  wrote {results / 'table.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
