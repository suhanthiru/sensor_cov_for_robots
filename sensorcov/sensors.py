"""Sensor families, their beam tables, and where they are bolted to the machine.

Two different models of the same sensor live here, and keeping them apart is the
point of the module.

*Line of sight* asks whether a sensor could see into a region at all: is the
point in range, inside the field of view, and unoccluded.  It says nothing about
angular sampling, which is the right abstraction for a coverage percentage.

*Beam accurate* generates the discrete rays the device actually emits.  This
matters more than it looks.  A 32 beam lidar spread over 45 degrees of elevation
puts about 1.4 degrees between channels, which at 15 m is a 37 cm vertical gap,
wider than the 25 cm voxels the envelope is cut into.  Targets genuinely fall
between beams.  Score the "at least three returns" safety metric under the line
of sight model instead and every target in range is struck by infinitely many
rays, so the metric reads 100 percent everywhere and means nothing.

Cameras do not emit beams, so they are not pretended into the beam model.  A
camera detects a target when the target is unoccluded, in range, inside the
frustum, and subtends enough pixels; :func:`pixels_on_target` is the equivalent
of a return count and the threshold is stated in the detection config rather
than smuggled in here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .frames import beam_dirs, pose_from_rpy
from .machine import CONFIG_DIR

LIDAR_KINDS = ("spinning", "solid_state")


@dataclass
class SensorSpec:
    """One sensor family, parameterised rather than hardcoded.

    Angles in degrees, ranges in metres.  ``v_center_deg`` tilts the middle of
    the vertical field of view, which is how a roof lidar gets aimed down at the
    ground instead of at the horizon.
    """

    name: str
    kind: str                  # spinning | solid_state | camera
    h_fov_deg: float
    v_fov_deg: float
    v_center_deg: float
    min_range_m: float
    max_range_m: float
    n_beams: int               # vertical channels; for a camera, image rows
    h_samples: int             # horizontal samples across the h_fov; camera columns
    notes: str = ""

    @classmethod
    def load(cls, name: str) -> SensorSpec:
        with open(Path(CONFIG_DIR) / "sensors" / (name + ".yaml")) as fh:
            return cls(**yaml.safe_load(fh))

    @property
    def is_lidar(self) -> bool:
        return self.kind in LIDAR_KINDS

    @property
    def h_res(self) -> float:
        """Horizontal angular resolution, radians per sample."""
        return np.radians(self.h_fov_deg) / float(self.h_samples)

    @property
    def v_res(self) -> float:
        """Vertical angular resolution, radians between channels."""
        n = max(int(self.n_beams) - 1, 1)
        return np.radians(self.v_fov_deg) / float(n)

    def beam_table(self):
        """Directions the device emits, in the sensor frame, as ``(N, 3)`` unit rays.

        A spinning lidar sweeps the full circle, so its azimuths wrap and the
        last sample is dropped to avoid doubling up on the seam.  A solid state
        unit and a camera span a bounded window and keep both edges.

        Cached, because it is constant in the sensor frame and the sweep asks for
        it once per sensor per pose: rebuilding a 32 by 1800 table of directions
        tens of thousands of times costs more than casting the rays does.
        """
        hit = getattr(self, "_beams", None)
        if hit is None:
            hit = self._build_beam_table()
            object.__setattr__(self, "_beams", hit)
        return hit

    def _build_beam_table(self):
        v_half = 0.5 * np.radians(self.v_fov_deg)
        v_c = np.radians(self.v_center_deg)
        el = np.linspace(v_c - v_half, v_c + v_half, int(self.n_beams))

        if self.h_fov_deg >= 359.999:
            az = np.arange(int(self.h_samples)) * (2 * np.pi / float(self.h_samples))
        else:
            h_half = 0.5 * np.radians(self.h_fov_deg)
            az = np.linspace(-h_half, h_half, int(self.h_samples))
        return beam_dirs(az, el).reshape(-1, 3), az, el

    def in_fov(self, r, az, el):
        """Line of sight predicate: in range and inside the field of view.

        Occlusion is deliberately not part of this.  Separating the two lets the
        field of view be tested against an empty scene, where the answer is
        known in closed form, independently of any ray casting.
        """
        ok = (r >= self.min_range_m) & (r <= self.max_range_m)
        if self.h_fov_deg < 359.999:
            ok &= np.abs(az) <= 0.5 * np.radians(self.h_fov_deg)
        d_el = np.abs(el - np.radians(self.v_center_deg))
        return ok & (d_el <= 0.5 * np.radians(self.v_fov_deg))

    def in_fov_fast(self, v_S: np.ndarray) -> np.ndarray:
        """The same predicate as :meth:`in_fov`, on raw sensor-frame vectors.

        Identical in meaning and roughly four times the speed, because it never
        evaluates an arctangent or an arcsine.  Both bounds are monotone in the
        angle, so the comparison can be made on the coordinates directly:
        elevation against ``r sin(bound)``, azimuth against ``x tan(bound)``.
        The hot loop calls this one and :meth:`in_fov` stays as the readable
        statement of what is meant; a test asserts they agree.
        """
        x, y, z = v_S[:, 0], v_S[:, 1], v_S[:, 2]
        r2 = x * x + y * y + z * z
        ok = (r2 >= self.min_range_m ** 2) & (r2 <= self.max_range_m ** 2)
        if not ok.any():
            return ok
        r = np.sqrt(r2)

        v_c, v_h = np.radians(self.v_center_deg), 0.5 * np.radians(self.v_fov_deg)
        ok &= (z >= r * np.sin(v_c - v_h)) & (z <= r * np.sin(v_c + v_h))

        if self.h_fov_deg < 359.999:
            half = 0.5 * np.radians(self.h_fov_deg)
            if half < 0.5 * np.pi:
                ok &= (x > 0.0) & (np.abs(y) <= x * np.tan(half))
            else:
                ok &= (x >= 0.0) | (np.abs(y) >= -x * np.tan(np.pi - half))
        return ok

    def pixels_on_target(self, r, width_m: float, height_m: float):
        """How many pixels a target of this size covers at this range.

        The camera stand-in for a lidar return count.  A target that is in
        frame, in range and unoccluded is still undetectable if it lands on four
        pixels, and this is the quantity that says so.
        """
        r = np.maximum(np.asarray(r, dtype=float), 1e-6)
        return (2.0 * np.arctan(0.5 * width_m / r) / self.h_res) * \
               (2.0 * np.arctan(0.5 * height_m / r) / self.v_res)


@dataclass
class Mount:
    """One sensor bolted to one link, posed in that link frame.

    The link matters as much as the pose.  A sensor on ``turret`` swings with
    the house and is rigid relative to the cab; a sensor on ``boom`` moves with
    the very thing that does most of the occluding, which is what makes layout D
    interesting and what makes its extrinsic a function of joint state.
    """

    sensor: SensorSpec
    link: str
    T_link: np.ndarray
    label: str = ""


@dataclass
class Layout:
    """A candidate sensor arrangement: the unit of the trade study."""

    name: str
    title: str
    mounts: list = field(default_factory=list)
    rationale: str = ""
    calibration_notes: str = ""

    @classmethod
    def load(cls, name: str) -> Layout:
        with open(Path(CONFIG_DIR) / "layouts" / (name + ".yaml")) as fh:
            raw = yaml.safe_load(fh)
        specs = {}
        mounts = []
        for entry in raw["mounts"]:
            key = entry["sensor"]
            if key not in specs:
                specs[key] = SensorSpec.load(key)
            mounts.append(Mount(
                sensor=specs[key],
                link=entry["link"],
                T_link=pose_from_rpy(entry["xyz"],
                                     roll_deg=entry.get("roll", 0.0),
                                     pitch_deg=entry.get("pitch", 0.0),
                                     yaw_deg=entry.get("yaw", 0.0)),
                label=entry.get("label", key)))
        return cls(name=raw["name"], title=raw["title"], mounts=mounts,
                   rationale=raw.get("rationale", ""),
                   calibration_notes=raw.get("calibration_notes", ""))

    @property
    def n_sensors(self) -> int:
        return len(self.mounts)

    @property
    def moves_with_the_boom(self) -> bool:
        return any(mo.link in ("boom", "stick", "bucket") for mo in self.mounts)

    def poses_W(self, link_T: dict):
        """World pose of every sensor, given the link poses for one configuration.

        Nothing special happens for a boom-mounted sensor: the same forward
        kinematics that puts the boom in the world puts the sensor on it.  That
        is the whole reason the mount carries a link name.
        """
        return [(mo, np.asarray(link_T[mo.link], dtype=float) @ mo.T_link)
                for mo in self.mounts]
