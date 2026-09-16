"""The machine as a small pile of primitive solids, and where they are bolted.

Primitives rather than a production mesh, on purpose.  Self occlusion is driven
by gross volume, not by surface detail: a boom that is roughly the right box in
roughly the right place casts the shadow that matters, and a coarse mesh builds
its BVH in under a millisecond, which is what makes a sixty thousand pose sweep
finish.  The one shape detail that is not optional is the gooseneck.  A real
excavator boom bends, and since boom shadow is the entire subject of this study,
modelling it as one straight box would put the shadow in the wrong place.

Every solid is attached to a named link and posed in that link frame.  The
kinematics puts the links in the world; nothing here knows about joint angles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .frames import make_T, pose_from_rpy

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

LINKS = ("chassis", "turret", "boom", "stick", "bucket")


@dataclass
class Solid:
    """One convex primitive, posed in the frame of the link it belongs to."""

    name: str
    link: str
    shape: str              # box | cylinder
    size: tuple             # box: (sx, sy, sz) m; cylinder: (radius, height) m
    T_link: np.ndarray      # pose of the solid centre in the link frame


@dataclass
class MachineSpec:
    """The Cat 320 GC, as loaded from configs/machine/cat320.yaml."""

    name: str
    source: str
    links: dict
    undercarriage: dict
    upperframe: dict
    boom_foot: dict
    working_ranges: dict
    joint_limits: dict
    joint_grid: dict
    inferred: dict
    solids: list = field(default_factory=list)

    @classmethod
    def load(cls, path=None) -> MachineSpec:
        path = Path(path) if path else CONFIG_DIR / "machine" / "cat320.yaml"
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        spec = cls(**raw)
        spec.solids = build_solids(spec)
        return spec

    # Convenience accessors, so call sites read like the spec sheet.
    @property
    def boom_len(self) -> float:
        return float(self.links["boom"])

    @property
    def stick_len(self) -> float:
        return float(self.links["stick"])

    @property
    def bucket_len(self) -> float:
        return float(self.links["bucket_tip_radius"])

    @property
    def deck_height(self) -> float:
        return float(self.inferred["deck_height"])

    def limits_rad(self, joint: str):
        lo, hi = self.joint_limits[joint]
        return float(np.radians(lo)), float(np.radians(hi))


def build_solids(m: MachineSpec) -> list:
    """Derive the collision primitives from the spec numbers.

    Written as a derivation rather than a table so that correcting a published
    dimension moves the geometry with it.  The positions the sheet fixes
    exactly, the tail swing radius, the top of the cab, the counterweight
    clearance, are solved for here rather than typed in, and the tests check
    that they land where the sheet says.
    """
    u, f, inf = m.undercarriage, m.upperframe, m.inferred
    deck = float(inf["deck_height"])
    out = []

    def box(name, link, sx, sy, sz, xyz, pitch_deg=0.0):
        out.append(Solid(name, link, "box", (sx, sy, sz),
                         pose_from_rpy(xyz, pitch_deg=pitch_deg)))

    # --- chassis: tracks and car body, fixed to the world -------------------
    half_gauge = 0.5 * float(u["track_gauge"])
    th = float(inf["track_height"])
    for side, y in (("left", +half_gauge), ("right", -half_gauge)):
        box("track_" + side, "chassis",
            float(u["track_length"]), float(u["shoe_width"]), th, (0.0, y, 0.5 * th))
    cbh = float(inf["car_body_height"])
    box("car_body", "chassis", float(inf["car_body_length"]), float(inf["car_body_width"]),
        cbh, (0.0, 0.0, float(u["ground_clearance"]) + 0.5 * cbh))

    # --- turret: deck, engine bay, counterweight, cab -----------------------
    # Turret frame origin is on the slew axis at deck height, so z here is
    # height above the deck and x is forward.
    box("deck", "turret", 3.40, float(f["width"]), 0.20, (0.0, 0.0, -0.10))

    hh = float(inf["house_height"])
    box("house", "turret", float(inf["house_length"]), float(f["width"]), hh,
        (-0.90, 0.0, 0.5 * hh))

    # The counterweight rear face is the tail swing radius and its underside is
    # the published counterweight clearance.  Both are solved, not guessed.
    cwl, cww = float(inf["counterweight_length"]), float(inf["counterweight_width"])
    cw_bottom = float(f["counterweight_clearance"]) - deck
    cw_top = hh - 0.05
    box("counterweight", "turret", cwl, cww, cw_top - cw_bottom,
        (-float(f["tail_swing_radius"]) + 0.5 * cwl, 0.0, 0.5 * (cw_bottom + cw_top)))

    # Cab sits on the deck; its roof is the published top-of-cab height.
    cab_h = float(f["cab_height"]) - deck
    box("cab", "turret", float(inf["cab_length"]), float(inf["cab_width"]), cab_h,
        (0.75, float(inf["cab_y"]), 0.5 * cab_h))
    fogs_t = float(f["fogs_height"]) - float(f["cab_height"])
    box("fogs", "turret", float(inf["cab_length"]), float(inf["cab_width"]), fogs_t,
        (0.75, float(inf["cab_y"]), cab_h + 0.5 * fogs_t))

    # --- boom: two segments meeting at the gooseneck knee --------------------
    # The pin-to-pin length is the kinematic boom length; the knee stands off
    # that line, which is what gives the boom its bend and its real shadow.
    L, kx, kz = m.boom_len, float(inf["boom_knee_x"]), float(inf["boom_knee_z"])
    bw, bd = float(inf["boom_width"]), float(inf["boom_depth"])
    for name, p0, p1 in (("boom_lower", (0.0, 0.0), (kx, kz)),
                         ("boom_upper", (kx, kz), (L, 0.0))):
        dx, dz = p1[0] - p0[0], p1[1] - p0[1]
        seg = float(np.hypot(dx, dz))
        # pitch_deg is the ZYX pitch and Ry(+a) tips +x down, so the sign flips.
        box(name, "boom", seg, bw, bd,
            (0.5 * (p0[0] + p1[0]), 0.0, 0.5 * (p0[1] + p1[1])),
            pitch_deg=-float(np.degrees(np.arctan2(dz, dx))))

    # --- stick and bucket ----------------------------------------------------
    box("stick", "stick", m.stick_len, float(inf["stick_width"]), float(inf["stick_depth"]),
        (0.5 * m.stick_len, 0.0, 0.0))
    box("bucket", "bucket", float(inf["bucket_length"]), float(inf["bucket_width"]),
        float(inf["bucket_depth"]), (0.5 * m.bucket_len, 0.0, -0.25))
    return out


def solid_mesh(s: Solid):
    """Triangulate one primitive in its link frame, as ``(V, F)``.

    Open3D is imported here rather than at module scope so the spec, the
    kinematics and the tests that only need transforms all import cleanly on a
    machine with no Open3D.
    """
    import open3d as o3d

    if s.shape == "box":
        sx, sy, sz = s.size
        mesh = o3d.geometry.TriangleMesh.create_box(sx, sy, sz)
        mesh.translate((-0.5 * sx, -0.5 * sy, -0.5 * sz))
    elif s.shape == "cylinder":
        radius, height = s.size
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius, height, resolution=16)
    else:
        raise ValueError("unknown primitive shape " + repr(s.shape))
    mesh.transform(np.asarray(s.T_link, dtype=float))
    return (np.asarray(mesh.vertices, dtype=np.float32),
            np.asarray(mesh.triangles, dtype=np.uint32))


def ground_mesh(extent: float = 40.0):
    """A flat ground plane, large enough that no ray in the study leaves it."""
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh.create_box(2 * extent, 2 * extent, 0.10)
    mesh.translate((-extent, -extent, -0.10))
    return (np.asarray(mesh.vertices, dtype=np.float32),
            np.asarray(mesh.triangles, dtype=np.uint32))


def boom_foot_T(m: MachineSpec) -> np.ndarray:
    """Pose of the boom foot pin in the turret frame."""
    return make_T(np.eye(3), (float(m.boom_foot["x"]), 0.0, float(m.boom_foot["z"])))
