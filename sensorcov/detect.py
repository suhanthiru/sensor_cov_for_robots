"""Can a person standing in the work area actually be detected, pose by pose.

This is the metric that matters, and it is a different computation from coverage.
Coverage asks whether a sight line exists.  Detection asks whether enough of the
sensor's real, discrete beams come back off a target of a particular size at a
particular range, which is a question a field-of-view cone cannot answer: under
a solid cone every target in range is struck by infinitely many rays.

The target is the one from the brief: a cylinder 1.7 m tall and 0.4 m across,
standing on grade, which is a person.  It has to be hit by at least three beams
from at least 5 m out, in every configuration the machine can reach.

The obvious implementation adds the cylinder to the scene and re-casts, and it
does not finish: that is a BVH rebuild per candidate position per pose.  Instead
the beams are cast once against the machine and the ground, and then intersected
analytically with each candidate cylinder.  A beam counts as a return when its
cylinder root is nearer than whatever it actually struck, which is exactly "the
cylinder was the first thing this beam hit" and gets occlusion by the machine for
free.

Cameras are scored on pixels rather than returns, because they do not emit
anything.  The equivalence is stated in the config rather than hidden here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Only rays whose azimuth lands near the target can possibly strike it.  A 0.2 m
# radius cylinder subtends about 4.6 degrees at 5 m and 1.5 degrees at 15 m, so
# this cull is worth about two orders of magnitude.  The factor is a safety
# margin on that half-angle; it does not need to cover a tilted mount smearing
# the target across azimuth, because the band is widened by the measured foot to
# head azimuth spread separately.  A test compares the culled answer against
# casting every beam at every target and requires them to be identical.
AZ_MARGIN = 1.35

# The candidate rings sit exactly on the ends of the standoff band, and a radius
# rebuilt through cos, sin and hypot lands a few ulps either side of the radius
# it was built from.  Without this slack, whole rings of candidates drop out of
# the band at azimuths that happen to round the wrong way, which looks exactly
# like a blind sector.
RANGE_EPS = 1e-6


@dataclass
class DetectionConfig:
    """The target, where it is tried, and what counts as detecting it."""

    height_m: float = 1.7        # a standing person
    diameter_m: float = 0.4
    min_range_m: float = 5.0     # the brief asks about targets from 5 m out
    max_range_m: float = 15.0    # the edge of the analysed envelope
    min_returns: int = 3         # lidar beams that must come back off the target
    min_pixels: float = 60.0     # the camera equivalent of min_returns, see below
    ring_radii: tuple = (5.0, 7.5, 10.0, 12.5, 15.0)
    n_azimuth: int = 72          # every 5 degrees

    @property
    def radius_m(self) -> float:
        return 0.5 * self.diameter_m

    def targets_W(self) -> np.ndarray:
        """Candidate standing positions, as ``(P, 2)`` ground coordinates."""
        th = np.arange(self.n_azimuth) * (2 * np.pi / self.n_azimuth)
        r = np.asarray(self.ring_radii, dtype=float)
        R, TH = np.meshgrid(r, th, indexing="ij")
        return np.stack([(R * np.cos(TH)).ravel(), (R * np.sin(TH)).ravel()], axis=1)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def lidar_returns(scene, T_WS, spec, targets_xy, cfg: DetectionConfig) -> np.ndarray:
    """Beams that come back off each candidate cylinder, as ``(P,)`` counts.

    One cast of the whole beam table, then an analytic cylinder test per
    candidate.  Candidates are evaluated independently, each as though it were
    the only obstacle present, which is the right question: we are asking
    whether *a* person standing there would be seen, not what happens when the
    site is full of them.
    """
    dirs_S, az, el = spec.beam_table()
    n_el, n_az = len(el), len(az)
    R_WS = np.asarray(T_WS[:3, :3], dtype=np.float64)
    o_W = np.asarray(T_WS[:3, 3], dtype=np.float64)

    # Kept in float64 throughout and narrowed to float32 only for the cast, so
    # the whole table is never copied just to change its width.
    dirs_W = dirs_S @ R_WS.T
    flat32 = dirs_W.astype(np.float32)
    t_hit = scene.cast(np.broadcast_to(o_W.astype(np.float32), flat32.shape),
                       flat32).t_hit.astype(np.float64)    # metres: dirs are unit
    t_hit = np.where(np.isfinite(t_hit), t_hit, np.inf).reshape(n_el, n_az)
    dirs_W = dirs_W.reshape(n_el, n_az, 3)

    # Beam azimuths are wrapped into the same interval arctan2 returns.  A
    # spinning unit generates them over [0, 2pi) while a target azimuth comes
    # back in (-pi, pi], and comparing the two unwrapped silently selects no
    # beams at all for every target on one side of the machine.
    az_w = _wrap(az)
    rad = cfg.radius_m
    targets = np.asarray(targets_xy, dtype=float)
    n_t = len(targets)
    out = np.zeros(n_t, dtype=np.int32)

    # Range is measured horizontally, because the brief's "from 5 m out" is a
    # standoff on the ground rather than a slant range.
    dxy = np.hypot(targets[:, 0] - o_W[0], targets[:, 1] - o_W[1])
    live = ((dxy >= cfg.min_range_m - RANGE_EPS) & (dxy <= cfg.max_range_m + RANGE_EPS)
            & (dxy > rad))
    if not live.any():
        return out

    # Which azimuth columns could possibly strike each target.  Both sensor
    # kinds sample azimuth uniformly, so the band is a contiguous run of column
    # indices and can be computed arithmetically.  The obvious version builds a
    # targets-by-azimuths boolean instead, and that array is the entire cost of
    # this function: a few hundred targets against 1800 columns dwarfs the ray
    # cast, which is about a millisecond.
    #
    # The foot and the head of the target sit at slightly different azimuths
    # under a pitched mount, so the band is centred between them and widened by
    # half their separation, which covers both in one run.  Over-selecting is
    # free because the intersection below is exact; under-selecting would lose
    # real returns.
    idx = np.flatnonzero(live)
    tg_live = targets[idx]
    wraps = spec.h_fov_deg >= 359.999
    az0 = 0.0 if wraps else float(az_w[0])
    daz = (2 * np.pi / n_az) if wraps else float((az_w[-1] - az_w[0]) / (n_az - 1))

    a = []
    for cz in (0.0, cfg.height_m):
        v_S = (np.column_stack([tg_live[:, 0], tg_live[:, 1],
                                np.full(len(idx), cz)]) - o_W) @ R_WS
        a.append(np.arctan2(v_S[:, 1], v_S[:, 0]))
    delta = _wrap(a[1] - a[0])
    a_mid = a[0] + 0.5 * delta
    half = (np.arcsin(np.minimum(rad / dxy[idx], 1.0)) * AZ_MARGIN
            + spec.h_res + 0.5 * np.abs(delta))

    k_lo = np.ceil((a_mid - half - az0) / daz)
    k_hi = np.floor((a_mid + half - az0) / daz)
    if wraps:
        n_cols = np.minimum(k_hi - k_lo + 1.0, n_az)
    else:
        k_lo = np.maximum(k_lo, 0.0)
        k_hi = np.minimum(k_hi, n_az - 1.0)
        n_cols = k_hi - k_lo + 1.0
    n_cols = np.maximum(n_cols, 0.0).astype(np.int64)
    if n_cols.sum() == 0:
        return out

    # Expand the ragged runs into flat (column, target) pairs without a loop.
    starts = np.repeat(k_lo.astype(np.int64), n_cols)
    ends = np.cumsum(n_cols)
    offs = np.arange(int(ends[-1])) - np.repeat(ends - n_cols, n_cols)
    cols = starts + offs
    cols = np.mod(cols, n_az) if wraps else cols
    tid = np.repeat(idx, n_cols)

    d = np.ascontiguousarray(dirs_W[:, cols, :]).reshape(-1, 3)
    tm = t_hit[:, cols].ravel()
    tid_f = np.tile(tid, n_el)
    ox = o_W[0] - targets[tid_f, 0]
    oy = o_W[1] - targets[tid_f, 1]

    # Ray against an infinite vertical cylinder, then clipped to its height.
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = 2.0 * (d[:, 0] * ox + d[:, 1] * oy)
    c = ox * ox + oy * oy - rad * rad
    disc = b * b - 4.0 * a * c
    ok = (disc > 0.0) & (a > 1e-12)
    if not ok.any():
        return out

    sq = np.sqrt(disc[ok])
    a_l, b_l, d_l, tm_l = a[ok], b[ok], d[ok], tm[ok]
    got = np.zeros(int(ok.sum()), dtype=bool)
    for t in ((-b_l - sq) / (2.0 * a_l), (-b_l + sq) / (2.0 * a_l)):
        z = o_W[2] + t * d_l[:, 2]
        got |= (t > 0.0) & (t < tm_l) & (z >= 0.0) & (z <= cfg.height_m)

    hit_tid = tid_f[ok][got]
    return np.bincount(hit_tid, minlength=n_t).astype(np.int32)


def camera_pixels(scene, T_WS, spec, targets_xy, cfg: DetectionConfig) -> np.ndarray:
    """Pixels each candidate covers, zero where it is out of frame or occluded.

    A camera has no beams to count, so the stand-in for a return count is how
    much of the image the target actually lands on.  Occlusion is tested on a
    short vertical line of samples up the target rather than on its centroid
    alone, so a target half hidden behind the boom is not scored as if it were
    fully visible.
    """
    targets_xy = np.asarray(targets_xy, dtype=float)
    n_samp = 5
    zs = np.linspace(0.15 * cfg.height_m, 0.85 * cfg.height_m, n_samp)
    pts = np.stack([np.repeat(targets_xy[:, 0], n_samp),
                    np.repeat(targets_xy[:, 1], n_samp),
                    np.tile(zs, len(targets_xy))], axis=1).astype(np.float32)

    visible = scene.line_of_sight(T_WS, spec, pts).reshape(-1, n_samp)
    frac = visible.mean(axis=1)

    o_W = np.asarray(T_WS[:3, 3], dtype=float)
    dxy = np.hypot(targets_xy[:, 0] - o_W[0], targets_xy[:, 1] - o_W[1])
    px = spec.pixels_on_target(dxy, cfg.diameter_m, cfg.height_m)
    in_band = (dxy >= cfg.min_range_m - RANGE_EPS) & (dxy <= cfg.max_range_m + RANGE_EPS)
    return np.where(in_band, px * frac, 0.0)


def detected(scene, layout_poses, cfg: DetectionConfig, targets_xy) -> np.ndarray:
    """Which candidate positions any sensor in the layout detects, as ``(P,)`` bool.

    Redundancy is not required here: one sensor seeing a person is the
    difference between a near miss and an incident, so the metric is any-sensor.
    """
    out = np.zeros(len(targets_xy), dtype=bool)
    for mo, T_WS in layout_poses:
        if mo.sensor.is_lidar:
            out |= lidar_returns(scene, T_WS, mo.sensor, targets_xy, cfg) >= cfg.min_returns
        else:
            out |= camera_pixels(scene, T_WS, mo.sensor, targets_xy, cfg) >= cfg.min_pixels
    return out


def sampling_limit(spec, cfg: DetectionConfig, ranges):
    """Where beam sampling, rather than occlusion, is what limits detection.

    The companion to the coverage numbers, and the reason the beam-accurate
    model is worth having even though the brief's threshold turns out not to
    bind.  A 1.7 m person inside 15 m is struck by tens of beams, so "at least
    three returns" is a visibility question in this study and not a sampling
    one.  Saying that plainly is more useful than reporting a metric that is
    satisfied everywhere it is not blocked.

    Two different heights are returned, because the obvious one is misleading.

    ``expected`` is the target height at which the expected return count first
    reaches the threshold, counting rows times columns.  It is optimistic: half
    a beam row times six columns averages three returns, but a target spanning
    half a row either catches that row or falls between two, so the real answer
    is six returns or none depending on where it happens to stand.

    ``guaranteed`` is the height that spans one whole beam row, ``2 r tan(v/2)``.
    At or above it a row must cross the target whatever its alignment, so
    detection stops depending on luck.  Vertical spacing is what runs out first
    on every sensor here, by an order of magnitude: a 32 channel unit puts
    1.45 degrees between rows and 0.2 degrees between columns.

    The sweep itself models the real beam elevations against the real cylinder,
    so it already gets the alignment right; this is the analytic companion for
    the report.
    """
    ranges = np.atleast_1d(np.asarray(ranges, dtype=float))
    heights = np.linspace(0.005, 3.0, 4000)

    expected = np.full(len(ranges), np.nan)
    for i, r in enumerate(ranges):
        cols = 2.0 * np.arctan(0.5 * cfg.diameter_m / r) / spec.h_res
        rows = 2.0 * np.arctan(0.5 * heights / r) / spec.v_res
        ok = np.flatnonzero(rows * cols >= cfg.min_returns)
        if ok.size:
            expected[i] = float(heights[ok[0]])
    guaranteed = 2.0 * ranges * np.tan(0.5 * spec.v_res)
    return {"range_m": ranges, "expected_m": expected, "guaranteed_m": guaranteed}
