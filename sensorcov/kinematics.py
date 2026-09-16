"""The four joint chain, and the grid of configurations the study sweeps.

Swing, boom, stick, bucket.  Everything downstream depends on this being right,
and a kinematics bug is the kind that produces plausible looking pictures and
wrong numbers, so the chain is checked against the manufacturer rather than
against itself: :func:`working_ranges` reconstructs the published working range
diagram from the link lengths and the joint limits, and the test asserts it
lands on the sheet.

Two forms of the same chain live here.  :func:`link_poses` builds the full 4x4
stack and is what places meshes and sensors.  :func:`tip_xz` is the planar
version, closed form and vectorised over whole joint grids at once, used
wherever only the bucket tip matters.  They are required to agree, and a test
says so.
"""

from __future__ import annotations

import numpy as np

from .frames import make_T, rot_y, rot_z, trans
from .machine import MachineSpec, boom_foot_T

JOINTS = ("swing", "boom", "stick", "bucket")


def link_poses(m: MachineSpec, q) -> dict:
    """World pose of every link frame for one configuration.

    ``q`` is ``(swing, boom, stick, bucket)`` in radians.  Returns the 4x4 pose
    of each frame named in :data:`machine.LINKS`, which is exactly what the
    scene builder and the sensor mounts need.
    """
    swing, boom, stick, bucket = (float(v) for v in q)

    T_chassis = np.eye(4)
    T_turret = T_chassis @ trans(z=m.deck_height) @ rot_z(swing)
    T_boom = T_turret @ boom_foot_T(m) @ rot_y(-boom)
    T_stick = T_boom @ trans(x=m.boom_len) @ rot_y(-stick)
    T_bucket = T_stick @ trans(x=m.stick_len) @ rot_y(-bucket)
    return {"chassis": T_chassis, "turret": T_turret, "boom": T_boom,
            "stick": T_stick, "bucket": T_bucket}


def bucket_tip(m: MachineSpec, q) -> np.ndarray:
    """World position of the tooth tip."""
    return (link_poses(m, q)["bucket"] @ np.array([m.bucket_len, 0.0, 0.0, 1.0]))[:3]


# ---------------------------------------------------------------------------
# planar closed form, vectorised over a whole grid
# ---------------------------------------------------------------------------


def tip_xz(m: MachineSpec, boom, stick, bucket):
    """Bucket tip radius and height, in the turret plane, for arrays of angles.

    The joint angles compose additively in this plane because all three pitch
    about the same axis: the stick points along ``boom + stick`` and the bucket
    along ``boom + stick + bucket``.  Radius is measured from the slew axis, so
    it is directly comparable to the published maximum reach, and height is
    above grade, so it is directly comparable to the published dig depth.
    """
    boom = np.asarray(boom, dtype=float)
    stick = np.asarray(stick, dtype=float)
    bucket = np.asarray(bucket, dtype=float)
    a1 = boom
    a2 = boom + stick
    a3 = boom + stick + bucket
    x = (float(m.boom_foot["x"]) + m.boom_len * np.cos(a1)
         + m.stick_len * np.cos(a2) + m.bucket_len * np.cos(a3))
    z = (m.deck_height + float(m.boom_foot["z"]) + m.boom_len * np.sin(a1)
         + m.stick_len * np.sin(a2) + m.bucket_len * np.sin(a3))
    return x, z


def working_ranges(m: MachineSpec, step_deg: float = 0.5) -> dict:
    """Reconstruct the published working ranges from the chain.

    Sweeps boom, stick and bucket on a fine grid and reads the extremes off the
    reachable set.  Maximum reach is taken at grade, within half a voxel of it,
    because that is how the sheet defines "maximum reach at ground line"; taking
    the unconstrained maximum radius instead would quietly report a number from
    a pose with the bucket in the air and overstate the machine by a metre.
    """
    grids = []
    for j in ("boom", "stick", "bucket"):
        lo, hi = m.limits_rad(j)
        grids.append(np.arange(lo, hi + 1e-9, np.radians(step_deg)))
    B, S, K = (g.ravel() for g in np.meshgrid(*grids, indexing="ij"))
    x, z = tip_xz(m, B, S, K)

    at_grade = np.abs(z) <= 0.125
    return {
        "max_reach_ground": float(x[at_grade].max()) if at_grade.any() else float("nan"),
        "max_dig_depth": float(-z.min()),
        "max_cut_height": float(z.max()),
        "max_reach_any": float(x.max()),
        "n_samples": int(x.size),
    }


# ---------------------------------------------------------------------------
# the articulation grid
# ---------------------------------------------------------------------------


def joint_grid(m: MachineSpec, quick: bool = False) -> np.ndarray:
    """The full ``(N, 4)`` grid of joint angles in radians, before filtering.

    Steps come from the config.  ``quick`` doubles every step, which cuts the
    grid by roughly sixteen and exists so the pipeline can be exercised end to
    end in under a minute while developing.
    """
    g = m.joint_grid
    scale = 2.0 if quick else 1.0
    axes = []
    for j, key in (("swing", "swing_step"), ("boom", "boom_step"),
                   ("stick", "stick_step"), ("bucket", "bucket_step")):
        lo, hi = m.limits_rad(j)
        stp = np.radians(float(g[key]) * scale)
        if j == "swing":
            # Swing wraps, so the last sample would duplicate the first.
            axes.append(np.arange(0.0, 2 * np.pi - 1e-9, stp))
        else:
            axes.append(np.arange(lo, hi + 1e-9, stp))
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([a.ravel() for a in mesh], axis=1)


def _box_corners(size, T) -> np.ndarray:
    sx, sy, sz = size
    c = np.array([[i, j, k] for i in (-0.5, 0.5) for j in (-0.5, 0.5) for k in (-0.5, 0.5)])
    c = c * np.array([sx, sy, sz])
    return c @ np.asarray(T, dtype=float)[:3, :3].T + np.asarray(T, dtype=float)[:3, 3]


def _inside_box(pts, size, T) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    local = (np.asarray(pts, dtype=float) - T[:3, 3]) @ T[:3, :3]
    half = 0.5 * np.asarray(size, dtype=float)
    return np.all(np.abs(local) <= half, axis=1)


def boxes_overlap(size_a, T_a, size_b, T_b, margin: float = 0.0) -> bool:
    """Whether two oriented boxes intersect, by separating axes.

    The obvious cheaper test - are any of B's corners inside A - is wrong in a
    way that matters here, and it took a buried sensor to notice.  A bucket
    1.15 m across can engulf a slice of the cab without putting a single one of
    its own corners inside it, so the corner test reports no interference while
    the two solids plainly overlap, and anything bolted to the cab front ends up
    inside the bucket.  Fifteen axes is the exact answer: three faces from each
    box and the nine edge cross products.
    """
    A = np.asarray(T_a, dtype=float)[:3, :3]
    B = np.asarray(T_b, dtype=float)[:3, :3]
    a = 0.5 * np.asarray(size_a, dtype=float) + margin
    b = 0.5 * np.asarray(size_b, dtype=float)
    t = np.asarray(T_b, dtype=float)[:3, 3] - np.asarray(T_a, dtype=float)[:3, 3]

    R = A.T @ B
    tA = A.T @ t
    # The epsilon keeps parallel edges, whose cross product is degenerate, from
    # reporting a spurious separating axis.
    absR = np.abs(R) + 1e-9

    for i in range(3):
        if abs(tA[i]) > a[i] + float(b @ absR[i, :]):
            return False
    for j in range(3):
        if abs(float(tA @ R[:, j])) > float(a @ absR[:, j]) + b[j]:
            return False
    for i in range(3):
        for j in range(3):
            i1, i2 = (i + 1) % 3, (i + 2) % 3
            j1, j2 = (j + 1) % 3, (j + 2) % 3
            ra = a[i1] * absR[i2, j] + a[i2] * absR[i1, j]
            rb = b[j1] * absR[i, j2] + b[j2] * absR[i, j1]
            if abs(tA[i2] * R[i1, j] - tA[i1] * R[i2, j]) > ra + rb:
                return False
    return True


def pose_is_valid(m: MachineSpec, q, grade_tol: float = 0.20, clearance: float = 0.20):
    """Whether one configuration is a state this study can score.

    Two rejections, both of them about the scene rather than about the machine.
    A pose that drives the bucket below grade is a real machine state but not
    one this study models, because the ground here is an unbroken flat plane
    with no trench cut into it; scoring visibility against a bucket buried in
    solid ground would be meaningless.  And a pose that folds the stick or the
    bucket into the cab or the engine housing is not reachable at all, since the
    real machine stops on its interference limits before it gets there.

    The house is inflated by ``clearance`` before that second test rather than
    being taken at its own surface.  A real machine keeps the bucket a hand's
    width off the glass, and without the margin the grid contains poses with the
    bucket millimetres from the cab.  Those are not merely unrealistic: anything
    bolted to the cab front is then buried inside the bucket, and the layout
    gets scored on a sensor that a real installer would never have let the
    bucket reach.

    Returns ``(ok, reason)`` with reason ``""`` when the pose is fine.  Failures
    come back as data rather than exceptions so the sweep can count them and
    report an auditable denominator.
    """
    poses = link_poses(m, q)
    moving = [s for s in m.solids if s.link in ("stick", "bucket")]
    static = [s for s in m.solids if s.link == "turret" and s.name in ("cab", "house", "counterweight")]

    # Below grade, judged on the whole bucket rather than just the tooth tip.
    lowest = np.inf
    for s in moving:
        pts = _box_corners(s.size, poses[s.link] @ s.T_link)
        lowest = min(lowest, float(pts[:, 2].min()))
    if lowest < -grade_tol:
        return False, "below_grade"

    # Folded into the house, with the house inflated by the clearance margin.
    T_turret = poses["turret"]
    for s in moving:
        T_s = poses[s.link] @ s.T_link
        for t in static:
            if boxes_overlap(t.size, T_turret @ t.T_link, s.size, T_s, margin=clearance):
                return False, "self_collision"
    return True, ""


def valid_poses(m: MachineSpec, grid: np.ndarray = None, quick: bool = False):
    """Filter a joint grid to the poses the study scores.

    Returns ``(poses, counts)`` where counts records how many configurations
    each rejection reason removed, so the denominator of every later metric can
    be traced back to a number of poses and a reason.
    """
    grid = joint_grid(m, quick=quick) if grid is None else grid
    keep = np.zeros(len(grid), dtype=bool)
    counts = {"total": int(len(grid)), "below_grade": 0, "self_collision": 0}

    # Swing does not change either test, so score one swing slice and tile it.
    # This is not an optimisation of convenience: it is true because both tests
    # are between parts that all rotate together with the turret.
    swing = grid[:, 0]
    uniq = np.unique(swing)
    base = swing == uniq[0]
    idx_base = np.flatnonzero(base)
    verdict = {}
    for i in idx_base:
        ok, why = pose_is_valid(m, grid[i])
        verdict[tuple(np.round(grid[i, 1:], 9))] = (ok, why)

    for i in range(len(grid)):
        ok, why = verdict[tuple(np.round(grid[i, 1:], 9))]
        keep[i] = ok
        if not ok:
            counts[why] += 1
    counts["kept"] = int(keep.sum())
    return grid[keep], counts
