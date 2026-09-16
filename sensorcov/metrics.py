"""Turning the sweep counters into the numbers the study reports.

The split that matters is between the two kinds of blind volume, and it falls
straight out of the counters.

A cell that was free space at some point in the sweep and was never seen by
anything, in any configuration, is **persistent** blind volume.  No amount of
articulation will help; it is a consequence of where the sensors are, and the
only fix is to move or add one.

A cell that was seen in some configurations and missed in others is
**transient**.  It is the more dangerous class, for a reason that has nothing to
do with its size: a persistent blind spot is discoverable.  Anyone who runs a
static coverage analysis, or walks around the machine once, will find it, and it
can be trained around.  A transient one is covered when you look and blind when
the boom comes down, so it is invisible to exactly the analysis most people run,
and it teaches operators to trust a region that is not always there.

So transient volume is reported alongside a severity, the fraction of the poses
in which a cell was free and unseen, rather than as a flag.  A cell blind in one
pose in a thousand and a cell blind in half of them are not the same finding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .envelope import Envelope
from .sweep import PERSON_BAND, SweepResult


def classify(res: SweepResult):
    """Split the envelope into solid, always seen, transient and persistent.

    Returns ``(persistent, transient, severity, ever_free)`` as per-cell arrays.
    ``severity`` is the fraction of free poses in which a cell went unseen, and
    is zero where a cell was always seen.
    """
    free = res.free_count.astype(np.float64)
    ever_free = res.free_count > 0

    persistent = ever_free & ~res.ever_seen
    transient = res.ever_seen & (res.seen_count < res.free_count)

    severity = np.zeros(len(free), dtype=np.float64)
    np.divide(res.free_count - res.seen_count, np.maximum(free, 1.0), out=severity)
    severity[~ever_free] = 0.0
    return persistent, transient, severity, ever_free


@dataclass
class Metrics:
    """The reported numbers for one layout."""

    layout: str
    title: str
    n_sensors: int
    moves_with_the_boom: bool
    n_poses: int

    coverage_mean: float          # % of free cells seen, averaged over poses
    coverage_worst: float         # % in the worst configuration
    coverage_best: float
    person_band_mean: float       # the same over 0 to 2 m
    person_band_worst: float
    dual_coverage_mean: float     # % seen by two or more sensors

    persistent_blind_m3: float
    transient_blind_m3: float
    transient_severe_m3: float    # blind in at least a quarter of poses
    always_seen_m3: float
    envelope_free_m3: float

    detect_mean: float            # % of standing positions detected, over poses
    detect_worst: float           # % in the worst configuration
    positions_always: int         # standing positions detected in every pose
    positions_never: int
    worst_position_rate: float    # best any position manages across the sweep

    seconds: float

    def row(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def compute(res: SweepResult, env: Envelope = None) -> Metrics:
    env = env or res.envelope
    persistent, transient, severity, ever_free = classify(res)
    band = env.band(*PERSON_BAND)

    n_poses = len(res.poses)
    det = res.detect_count.astype(np.float64) / max(n_poses, 1)

    return Metrics(
        layout=res.layout, title=res.title, n_sensors=res.n_sensors,
        moves_with_the_boom=bool(res.moves_with_the_boom), n_poses=n_poses,

        coverage_mean=100.0 * float(res.per_pose_cov.mean()),
        coverage_worst=100.0 * float(res.per_pose_cov.min()),
        coverage_best=100.0 * float(res.per_pose_cov.max()),
        person_band_mean=100.0 * float(res.per_pose_person.mean()),
        person_band_worst=100.0 * float(res.per_pose_person.min()),
        dual_coverage_mean=100.0 * float(res.per_pose_dual.mean()),

        persistent_blind_m3=env.volume_of(persistent),
        transient_blind_m3=env.volume_of(transient),
        transient_severe_m3=env.volume_of(transient & (severity >= 0.25)),
        always_seen_m3=env.volume_of(ever_free & (res.seen_count == res.free_count)),
        envelope_free_m3=env.volume_of(ever_free),

        detect_mean=100.0 * float(res.per_pose_detect.mean()),
        detect_worst=100.0 * float(res.per_pose_detect.min()),
        positions_always=int((det >= 1.0).sum()),
        positions_never=int((det <= 0.0).sum()),
        worst_position_rate=100.0 * float(det.min()),

        seconds=float(res.seconds),
    )


def band_breakdown(res: SweepResult, env: Envelope = None, edges=(0.0, 2.0, 4.0, 6.0)):
    """Persistent and transient volume by height band.

    A single envelope-wide figure hides the thing that matters.  A layout can
    look excellent overall while being blind exactly in the band a person stands
    in, because that band is a sixth of the volume and most of the envelope is
    open air above head height.
    """
    env = env or res.envelope
    persistent, transient, severity, ever_free = classify(res)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = env.band(lo, hi)
        free = sel & ever_free
        out.append({
            "band": f"{lo:.0f} to {hi:.0f} m",
            "free_m3": env.volume_of(free),
            "persistent_m3": env.volume_of(sel & persistent),
            "transient_m3": env.volume_of(sel & transient),
            "persistent_pct": 100.0 * env.volume_of(sel & persistent) / max(env.volume_of(free), 1e-9),
            "transient_pct": 100.0 * env.volume_of(sel & transient) / max(env.volume_of(free), 1e-9),
        })
    return out


def worst_poses(res: SweepResult, n: int = 5):
    """The configurations with the lowest coverage, in degrees, for the report.

    Reported because "worst case 41 percent" is not actionable on its own.  The
    joint angles say whether the worst case is a pose the machine spends its
    working day in or a corner of the envelope it passes through.
    """
    idx = np.argsort(res.per_pose_cov)[:n]
    return [{"swing": float(np.degrees(res.poses[i, 0])),
             "boom": float(np.degrees(res.poses[i, 1])),
             "stick": float(np.degrees(res.poses[i, 2])),
             "bucket": float(np.degrees(res.poses[i, 3])),
             "coverage": 100.0 * float(res.per_pose_cov[i]),
             "person_band": 100.0 * float(res.per_pose_person[i]),
             "detect": 100.0 * float(res.per_pose_detect[i])} for i in idx]


def pareto_front(rows, maximise=("coverage_mean",), minimise=("n_sensors",)):
    """Which layouts are not beaten on every axis at once.

    Coverage up, sensor count down.  A layout is on the front when nothing else
    is at least as good on both and strictly better on one.

    Transient blind volume is deliberately *not* a third axis, although it is the
    colour on the plot and it was the obvious candidate.  Adding it makes the
    front degenerate, and the reason is worth stating: a layout that sees very
    little has almost no transient blind volume, because a cell that is never
    covered in any configuration is persistent, not transient.  Minimising
    transient volume on its own therefore rewards being uniformly bad, and the
    first version of layout C, whose four windows left a gap straight ahead,
    made the front purely by scoring zero on it.  Transient volume is a way of
    reading a layout that already covers the envelope, not a goal by itself.
    """
    def dominates(a, b):
        ge = (all(a[k] >= b[k] for k in maximise)
              and all(a[k] <= b[k] for k in minimise))
        gt = (any(a[k] > b[k] for k in maximise)
              or any(a[k] < b[k] for k in minimise))
        return ge and gt

    return [r for r in rows if not any(dominates(o, r) for o in rows if o is not r)]
