"""The trade study: four layouts, one table, and an explicit Pareto front.

The four layouts vary on two axes that are usually confounded, count and
placement, so that the front means something.  A and D both carry two sensors
and differ only in where the second one goes; B and C carry more.  If the answer
were simply "more sensors is better" the front would be a straight line and
there would be nothing to trade.

The axis that is easy to leave out of a plot like this is what each arrangement
costs to keep working.  A fixed extrinsic is solved once at build; an extrinsic
that is a function of joint state is solved continuously, by the encoders, for
the life of the machine.  Those are not the same engineering commitment and the
report says so in words, because no scalar on a Pareto plot carries it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .detect import DetectionConfig
from .metrics import band_breakdown, compute, pareto_front, worst_poses
from .sweep import SweepResult, run_sweep

LAYOUTS = ("a_cab_corners", "b_roof_and_cameras", "c_corner_solid_state", "d_cab_and_boom")
RESULTS = Path(__file__).resolve().parent.parent / "results"


def load_or_run(layout: str, results_dir: Path, **kw) -> SweepResult:
    """Reuse a completed sweep if one is on disk, otherwise run it.

    The sweeps are the expensive part and the figures get iterated on far more
    often than the physics does, so they are cached as plain npz.
    """
    path = results_dir / f"{layout}.npz"
    if path.exists() and not kw.pop("force", False):
        print(f"{layout}: reusing {path.name}")
        return SweepResult.load(path)
    kw.pop("force", None)
    res = run_sweep(layout, **kw)
    res.save(path)
    return res


def markdown_table(rows) -> str:
    """The headline table, in the form the README carries it."""
    front = {r["layout"] for r in pareto_front(rows)}
    head = ("| Layout | Sensors | Coverage | Worst pose | Person band | Dual | "
            "Persistent | Transient | Detected | Never seen |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n")
    body = ""
    for r in rows:
        mark = " *" if r["layout"] in front else ""
        body += (f"| {r['layout'][0].upper()}{mark} | {r['n_sensors']} | "
                 f"{r['coverage_mean']:.1f}% | {r['coverage_worst']:.1f}% | "
                 f"{r['person_band_mean']:.1f}% | {r['dual_coverage_mean']:.1f}% | "
                 f"{r['persistent_blind_m3']:.0f} m3 | "
                 f"{r['transient_blind_m3']:.0f} m3 | "
                 f"{r['detect_mean']:.1f}% | {r['positions_never']} |\n")
    return head + body


def run(layouts=LAYOUTS, results_dir: Path = None, **kw) -> dict:
    results_dir = Path(results_dir or RESULTS)
    results_dir.mkdir(parents=True, exist_ok=True)

    sweeps, rows, detail = {}, [], {}
    for name in layouts:
        res = load_or_run(name, results_dir, **kw)
        sweeps[name] = res
        met = compute(res)
        rows.append(met.row())
        detail[name] = {"metrics": met.row(),
                        "bands": band_breakdown(res),
                        "worst_poses": worst_poses(res)}

    front = pareto_front(rows)
    summary = {"rows": rows, "pareto": [r["layout"] for r in front], "detail": detail}
    with open(results_dir / "study.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return {"summary": summary, "sweeps": sweeps, "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run sweeps already on disk")
    ap.add_argument("--layouts", nargs="*", default=list(LAYOUTS))
    ap.add_argument("--results", default=None)
    args = ap.parse_args(argv)

    out = run(args.layouts, results_dir=args.results, workers=args.workers,
              quick=args.quick, force=args.force, det_cfg=DetectionConfig())
    rows = out["rows"]

    print()
    print(markdown_table(rows))
    print("* on the Pareto front over coverage up, sensor count down, "
          "transient blind volume down")
    print()
    for r in rows:
        print(f"  {r['layout']:<22} {r['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
