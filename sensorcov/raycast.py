"""Ray casting against the machine, and the two visibility questions asked of it.

The scene for one configuration is the machine's primitives placed by forward
kinematics plus a flat ground plane.  It is rebuilt per pose, which sounds
expensive and is not: a dozen boxes build their BVH in well under a millisecond,
which is the whole reason primitive geometry was chosen over a production mesh.

Two details of the Open3D interface are load bearing, and both were established
by experiment rather than read off the documentation.

``t_hit`` is in units of the direction vector length, not in metres.  So passing
an *unnormalised* direction from the sensor to a target with ``tfar`` just under
one is an exact per-ray segment query: it asks whether anything stands between
those two specific points, and nothing else.  This matters because
``test_occlusions`` only takes scalar ``tnear`` and ``tfar``, so there is no
other way to give every ray its own range limit, and the any-hit query it runs
is roughly twice the speed of a closest-hit cast.

``cast_rays`` reports ``geometry_ids``, with ``INVALID_ID`` for a miss.  That is
what lets a beam be attributed to the thing it struck first, which the detection
metric needs and a pure occlusion test cannot answer.

Occupancy is computed analytically from the primitives rather than with
``compute_occupancy``.  The Open3D routine counts ray crossings, so a point
inside two overlapping solids - the cab sitting on the deck, say - crosses an
even number of surfaces and is reported as empty.  The machine is built out of
deliberately overlapping boxes, so that failure would be silent and everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .machine import LINKS, MachineSpec, ground_mesh, solid_mesh

# Just inside the segment at both ends: clear of the sensor housing at the near
# end, and stopping short of the target cell at the far end so the cell's own
# contents cannot occlude it.
T_NEAR = 1e-3
T_FAR = 1.0 - 1e-3


@dataclass
class RayHits:
    """Result of casting a batch of rays.  ``t_hit`` is inf where nothing was hit."""

    t_hit: np.ndarray        # (N,) float32, in units of the direction vector length
    geom_id: np.ndarray      # (N,) uint32, INVALID_ID on a miss
    prim_id: np.ndarray      # (N,) int64

    @property
    def hit(self) -> np.ndarray:
        return np.isfinite(self.t_hit)


class MachineScene:
    """The machine in one configuration, plus the ground, ready to be cast against."""

    def __init__(self, m: MachineSpec, link_T: dict, links=LINKS, ground: bool = True):
        import open3d as o3d
        import open3d.core as o3c

        self.m = m
        self.link_T = link_T
        self._solids = [s for s in m.solids if s.link in links]
        self._scene = o3d.t.geometry.RaycastingScene()
        self.invalid_id = o3d.t.geometry.RaycastingScene.INVALID_ID

        for s in self._solids:
            V, F = solid_mesh(s)
            V = V @ np.asarray(link_T[s.link], dtype=np.float32)[:3, :3].T \
                + np.asarray(link_T[s.link], dtype=np.float32)[:3, 3]
            self._scene.add_triangles(o3c.Tensor(np.ascontiguousarray(V, np.float32)),
                                      o3c.Tensor(np.ascontiguousarray(F, np.uint32)))
        self.ground_id = None
        if ground:
            V, F = ground_mesh()
            self.ground_id = int(self._scene.add_triangles(
                o3c.Tensor(np.ascontiguousarray(V, np.float32)),
                o3c.Tensor(np.ascontiguousarray(F, np.uint32))))

    # -----------------------------------------------------------------------

    def line_of_sight(self, T_WS: np.ndarray, spec, pts_W: np.ndarray) -> np.ndarray:
        """Which of ``pts_W`` this sensor can see, ignoring angular sampling.

        Range and field of view are tested first and only the survivors are cast.
        That is not a micro-optimisation: a 30 m camera against a 15 m envelope
        discards most of the cells before any ray exists, and the cull is what
        makes a sixty thousand pose sweep finish in minutes.
        """
        import open3d.core as o3c

        pts_W = np.asarray(pts_W, dtype=np.float32)
        visible = np.zeros(len(pts_W), dtype=bool)

        origin = np.asarray(T_WS[:3, 3], dtype=np.float32)
        v_S = (pts_W - origin) @ np.asarray(T_WS[:3, :3], dtype=np.float32)
        keep = spec.in_fov_fast(v_S)
        if not keep.any():
            return visible

        v = pts_W[keep] - origin
        rays = np.empty((len(v), 6), dtype=np.float32)
        rays[:, :3] = origin
        rays[:, 3:] = v                      # unnormalised: t = 1 is the target
        occluded = self._scene.test_occlusions(
            o3c.Tensor(np.ascontiguousarray(rays)), tnear=T_NEAR, tfar=T_FAR).numpy()

        visible[keep] = ~occluded
        return visible

    def cast(self, origins_W: np.ndarray, dirs_W: np.ndarray) -> RayHits:
        """Cast ``(N, 3)`` origins along ``(N, 3)`` unit directions, world frame."""
        import open3d.core as o3c

        rays = np.concatenate([np.asarray(origins_W, np.float32),
                               np.asarray(dirs_W, np.float32)], axis=1)
        ans = self._scene.cast_rays(o3c.Tensor(np.ascontiguousarray(rays)))
        return RayHits(t_hit=ans["t_hit"].numpy().astype(np.float32),
                       geom_id=ans["geometry_ids"].numpy(),
                       prim_id=ans["primitive_ids"].numpy().astype(np.int64))

    def distance_to(self, pts_W: np.ndarray) -> np.ndarray:
        """Unsigned distance from points to the nearest surface in the scene."""
        import open3d.core as o3c

        return self._scene.compute_distance(
            o3c.Tensor(np.ascontiguousarray(np.asarray(pts_W, np.float32)))).numpy()


# ---------------------------------------------------------------------------
# occupancy, analytically
# ---------------------------------------------------------------------------


def occupied_mask(m: MachineSpec, link_T: dict, pts_W: np.ndarray,
                  links=LINKS, margin: float = 0.0) -> np.ndarray:
    """Which points lie inside the machine in this configuration.

    This is the difference between a cell being blind and a cell being solid,
    and the distinction is not cosmetic.  The boom sweeps bodily through the
    work envelope.  Count the cells it currently fills as unseen and a low boom
    reports an enormous transient blind volume that is really just the boom
    being where the boom is, which would make the headline result of this study
    an artefact.
    """
    pts_W = np.asarray(pts_W, dtype=float)
    inside = np.zeros(len(pts_W), dtype=bool)

    placed = []
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for s in m.solids:
        if s.link not in links:
            continue
        T = np.asarray(link_T[s.link], dtype=float) @ s.T_link
        reach = 0.5 * np.abs(T[:3, :3]) @ np.asarray(s.size, dtype=float) + margin
        lo = np.minimum(lo, T[:3, 3] - reach)
        hi = np.maximum(hi, T[:3, 3] + reach)
        placed.append((s, T))

    # One box test against the whole machine first.  The machine occupies a few
    # percent of a 15 m envelope, so this discards almost every cell before any
    # per-solid work happens.
    near = np.flatnonzero(np.all((pts_W >= lo) & (pts_W <= hi), axis=1))
    if near.size == 0:
        return inside
    sub = pts_W[near]
    hits = np.zeros(near.size, dtype=bool)

    for s, T in placed:
        local = (sub - T[:3, 3]) @ T[:3, :3]
        if s.shape == "box":
            half = 0.5 * np.asarray(s.size, dtype=float) + margin
            hits |= np.all(np.abs(local) <= half, axis=1)
        elif s.shape == "cylinder":
            radius, height = s.size
            hits |= ((np.hypot(local[:, 0], local[:, 1]) <= radius + margin)
                     & (np.abs(local[:, 2]) <= 0.5 * height + margin))
    inside[near] = hits
    return inside
