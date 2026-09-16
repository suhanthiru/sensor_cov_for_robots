"""Run the articulation sweep for one sensor layout.

    python -m sensorcov.run_sweep --layout a_cab_corners --workers 12
    python -m sensorcov.run_sweep --layout d_cab_and_boom --quick
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .detect import DetectionConfig
from .metrics import band_breakdown, compute, worst_poses
from .sweep import run_sweep

RESULTS = Path(__file__).resolve().parent.parent / "results"


def report(res, indent: str = "") -> dict:
    """Print the headline numbers for one sweep and return them as a dict."""
    met = compute(res)
    bands = band_breakdown(res)
    worst = worst_poses(res)

    print(f"{indent}{met.title}")
    print(f"{indent}  {met.n_sensors} sensors, {met.n_poses:,} poses, "
          f"{met.seconds / 60:.1f} min")
    print(f"{indent}  coverage        mean {met.coverage_mean:5.1f}%   "
          f"worst {met.coverage_worst:5.1f}%   best {met.coverage_best:5.1f}%")
    print(f"{indent}  person band     mean {met.person_band_mean:5.1f}%   "
          f"worst {met.person_band_worst:5.1f}%")
    print(f"{indent}  dual coverage   mean {met.dual_coverage_mean:5.1f}%")
    print(f"{indent}  blind volume    persistent {met.persistent_blind_m3:8.1f} m3   "
          f"transient {met.transient_blind_m3:8.1f} m3  "
          f"(severe {met.transient_severe_m3:.1f})")
    print(f"{indent}  free envelope   {met.envelope_free_m3:8.1f} m3   "
          f"always seen {met.always_seen_m3:8.1f} m3")
    print(f"{indent}  detection       mean {met.detect_mean:5.1f}%   "
          f"worst pose {met.detect_worst:5.1f}%   "
          f"never-detected positions {met.positions_never}")
    print(f"{indent}  by band:")
    for b in bands:
        print(f"{indent}    {b['band']:>10}  free {b['free_m3']:7.1f} m3   "
              f"persistent {b['persistent_pct']:5.1f}%   transient {b['transient_pct']:5.1f}%")
    print(f"{indent}  worst configuration: "
          f"swing {worst[0]['swing']:.0f}, boom {worst[0]['boom']:.0f}, "
          f"stick {worst[0]['stick']:.0f}, bucket {worst[0]['bucket']:.0f} deg "
          f"-> {worst[0]['coverage']:.1f}% coverage")
    return {"metrics": met.row(), "bands": bands, "worst_poses": worst}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--quick", action="store_true",
                    help="coarse joint steps and 50 cm cells, for checking the pipeline")
    ap.add_argument("--cell", type=float, default=0.25, help="voxel edge, m")
    ap.add_argument("--radius", type=float, default=15.0, help="envelope radius, m")
    ap.add_argument("--z-hi", type=float, default=6.0, help="envelope top, m")
    ap.add_argument("--out", default=None, help="where to write the .npz")
    args = ap.parse_args(argv)

    res = run_sweep(args.layout, quick=args.quick, workers=args.workers,
                    cell=args.cell, radius=args.radius, z_hi=args.z_hi,
                    det_cfg=DetectionConfig())

    out = Path(args.out) if args.out else RESULTS / f"{args.layout}.npz"
    res.save(out)
    print()
    summary = report(res)
    print()
    print(f"written to {out}")
    with open(out.with_suffix(".json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
