"""Frame conventions for the coverage study.

This module is the single source of truth; everything else imports from here.
The conventions are deliberately the same as ``lidar_demo/frames.py`` so that a
sensor model written for one project means the same thing in the other.

World ``W``
    Local ENU, right handed, ``z`` up.  The origin sits on the ground directly
    under the slew axis, so ``z`` is height above grade and the sign of a
    digging depth is unambiguous.

Machine ``M``
    The frame the study reports in.  Origin on the slew axis at grade, exactly
    like the world, but yaw-aligned with the turret: ``+x`` is whichever way the
    upper structure is currently facing.  It is the world frame with the swing
    angle taken out, ``R_MW = Rz(-swing)``.

    Coverage is a machine-relative property and has to be measured in a
    machine-relative frame.  Measured in the world instead, swing swamps
    everything: a cab-mounted sensor sweeps past every fixed point as the
    machine slews, so almost every cell is seen in some configurations and
    missed in others, and the persistent-versus-transient split degenerates to
    "nearly all transient" no matter where the sensors are.  What that would be
    measuring is that the machine slews, which is not in question.  In ``M`` the
    swing drops out for anything bolted to the turret, and what is left is the
    articulation effect the study is actually about.

Chassis ``C``
    Tracks and car body.  Fixed to the world in this study: the machine is
    parked and only the upper structure moves.  ``x`` forward along the tracks,
    ``y`` left, ``z`` up.

Turret ``T``
    The upper structure, rotated from the chassis by the swing angle about
    ``z``.  Origin on the slew axis at deck height.  Everything bolted to the
    house - cab, counterweight, engine bay, and any cab-mounted sensor - is
    rigid in this frame.

Boom ``B``, stick ``K``, bucket ``U``
    Each link's origin is its inboard pin, with ``x`` along the link toward the
    outboard pin.  A joint angle rotates about ``-y``, so a *positive* boom
    angle raises the boom: ``Ry(-a)`` sends ``+x`` to ``(cos a, 0, sin a)``.
    That sign is stated here because the opposite convention produces a machine
    that digs into the sky and still looks plausible in a plot.

Sensor ``S``
    ``z`` is the spin axis, ``x`` points at azimuth zero, ``y`` left.  A beam at
    azimuth ``az`` and elevation ``el`` points along
    ``[cos el cos az, cos el sin az, sin el]``; azimuth increases counter
    clockwise about ``+z``.  A mount is given as ``T_LS``, the pose of the
    sensor in the frame of the link it is bolted to.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# rotations
# ---------------------------------------------------------------------------


def Rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def Ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def Rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rpy_zyx_to_R(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """``Rz(yaw) Ry(pitch) Rx(roll)`` from degrees."""
    r, p, y = np.radians([float(roll_deg), float(pitch_deg), float(yaw_deg)])
    return Rz(y) @ Ry(p) @ Rx(r)


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Nearest rotation matrix; numerical hygiene after long chains of products."""
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=float))
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0.0:
        U = U.copy()
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


# ---------------------------------------------------------------------------
# SE(3) as 4x4 homogeneous matrices
# ---------------------------------------------------------------------------


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def T_inv(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    Rt = T[:3, :3].T
    out = np.eye(4)
    out[:3, :3] = Rt
    out[:3, 3] = -Rt @ T[:3, 3]
    return out


def trans(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


def rot_y(a: float) -> np.ndarray:
    return make_T(Ry(a), np.zeros(3))


def rot_z(a: float) -> np.ndarray:
    return make_T(Rz(a), np.zeros(3))


def pose_from_rpy(xyz, roll_deg: float = 0.0, pitch_deg: float = 0.0,
                  yaw_deg: float = 0.0) -> np.ndarray:
    """A mount pose from a translation and ZYX Euler angles in degrees."""
    return make_T(rpy_zyx_to_R(roll_deg, pitch_deg, yaw_deg), np.asarray(xyz, dtype=float))


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply one pose to ``(N, 3)`` points."""
    T = np.asarray(T, dtype=float)
    return np.asarray(pts, dtype=float) @ T[:3, :3].T + T[:3, 3]


# ---------------------------------------------------------------------------
# beams
# ---------------------------------------------------------------------------


def beam_dirs(az_rad: np.ndarray, el_rad: np.ndarray) -> np.ndarray:
    """Unit beam directions in the sensor frame, shaped ``(n_el, n_az, 3)``."""
    az = np.asarray(az_rad, dtype=float)[None, :]
    el = np.asarray(el_rad, dtype=float)[:, None]
    ce, se = np.cos(el), np.sin(el)
    x = ce * np.cos(az)
    y = ce * np.sin(az)
    z = np.broadcast_to(se, x.shape)
    return np.stack([x, y, z], axis=-1)


def spherical_in_frame(T_WS: np.ndarray, pts_W: np.ndarray):
    """Range, azimuth and elevation of world points as seen from a sensor pose.

    Returns ``(r, az, el)`` with angles in radians, azimuth in ``(-pi, pi]`` and
    elevation in ``[-pi/2, pi/2]``.  This is the predicate the field-of-view
    test is built on, kept separate from any ray casting so it can be checked on
    its own.
    """
    T_WS = np.asarray(T_WS, dtype=float)
    v_S = (np.asarray(pts_W, dtype=float) - T_WS[:3, 3]) @ T_WS[:3, :3]
    r = np.linalg.norm(v_S, axis=1)
    safe = np.where(r > 0.0, r, 1.0)
    az = np.arctan2(v_S[:, 1], v_S[:, 0])
    el = np.arcsin(np.clip(v_S[:, 2] / safe, -1.0, 1.0))
    return r, az, el
