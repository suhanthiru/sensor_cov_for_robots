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
# this cull is worth about two orders of magnitude.  The margin is generous
# because a tilted mount smears the target across azimuth a little.
AZ_MARGIN = 3.0

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

    dirs_W = (dirs_S @ R_WS.T).astype(np.float32)
    origins = np.broadcast_to(o_W.astype(np.float32), dirs_W.shape)
    t_hit = scene.cast(origins, dirs_W).t_hit.astype(np.float64)   # metres: dirs are unit
    t_hit = np.where(np.isfinite(t_hit), t_hit, np.inf).reshape(n_el, n_az)
    dirs_W = dirs_W.reshape(n_el, n_az, 3).astype(np.float64)

    # Beam azimuths are wrapped into the same interval arctan2 returns.  A
    # spinning unit generates them over [0, 2pi) while a target azimuth comes
    # back in (-pi, pi], and comparing the two unwrapped silently selects no
    # beams at all for every target on one side of the machine.
    az_w = _wrap(az)
    rad = cfg.radius_m
    out = np.zeros(len(targets_xy), dtype=np.int32)

    for i, (cx, cy) in enumerate(np.asarray(targets_xy, dtype=float)):
        # Range is measured to the target, in the horizontal plane, because the
        # brief's "from 5 m out" is a standoff on the ground, not a slant range.
        dxy = float(np.hypot(cx - o_W[0], cy - o_W[1]))
        if not (cfg.min_range_m - RANGE_EPS <= dxy <= cfg.max_range_m + RANGE_EPS)                 or dxy <= rad:
            continue

        # Azimuth band, in the sensor frame, taken over both the foot and the
        # head of the target so a pitched mount cannot slide it out of the band.
        # Over-selecting here costs a little time and nothing else, because the
        # analytic test below is exact; under-selecting would lose real returns.
        half = np.arcsin(min(rad / dxy, 1.0)) * AZ_MARGIN + spec.h_res
        keep = np.zeros(n_az, dtype=bool)
        for cz in (0.0, cfg.height_m):
            v_S = R_WS.T @ (np.array([cx, cy, cz]) - o_W)
            a_t = np.arctan2(v_S[1], v_S[0])
            keep |= np.abs(_wrap(az_w - a_t)) <= half
        cols = np.flatnonzero(keep)
        if cols.size == 0:
            continue

        d = dirs_W[:, cols, :].reshape(-1, 3)
        t_max = t_hit[:, cols].ravel()
        ox, oy = o_W[0] - cx, o_W[1] - cy

        # Ray against an infinite vertical cylinder, then clipped to its height.
        a = d[:, 0] ** 2 + d[:, 1] ** 2
        b = 2.0 * (d[:, 0] * ox + d[:, 1] * oy)
        c = ox * ox + oy * oy - rad * rad
        disc = b * b - 4.0 * a * c
        live = (disc > 0.0) & (a > 1e-12)
        if not live.any():
            continue
        sq = np.sqrt(disc[live])
        a_l, b_l = a[live], b[live]
        t_near = (-b_l - sq) / (2.0 * a_l)
        t_far = (-b_l + sq) / (2.0 * a_l)

        d_l, tm = d[live], t_max[live]
        oz = o_W[2]
        got = np.zeros(live.sum(), dtype=bool)
        for t in (t_near, t_far):
            z = oz + t * d_l[:, 2]
            got |= (t > 0.0) & (t < tm) & (z >= 0.0) & (z <= cfg.height_m)
        out[i] = int(got.sum())
    return out


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
