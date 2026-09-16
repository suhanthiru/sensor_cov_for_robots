"""Tests for the chain, the geometry it carries, and the grid it sweeps.

These are deliberately about meaning rather than arithmetic.  A sign error in a
joint convention, or a link attached to the wrong parent, produces a machine
that renders perfectly well and reports coverage numbers that are quietly wrong
about which direction the boom was pointing.  The defence is to check the chain
against the manufacturer instead of against ourselves: if the link lengths, the
boom foot and the joint limits together reproduce the published working range
diagram, then very little can be wrong at once.
"""

from __future__ import annotations

import numpy as np
import pytest

from sensorcov import kinematics as K
from sensorcov.machine import MachineSpec


@pytest.fixture(scope="module")
def m():
    return MachineSpec.load()


# ---------------------------------------------------------------------------
# the chain against the spec sheet
# ---------------------------------------------------------------------------


def test_the_chain_reproduces_the_published_working_ranges(m):
    # 0.15 m on a machine that reaches 9.86 m is 1.5 percent.  The residual is
    # not zero because the boom foot and the joint limits are fitted against
    # three published numbers and cannot hit all three exactly.
    wr = K.working_ranges(m, step_deg=0.25)
    for key in ("max_reach_ground", "max_dig_depth", "max_cut_height"):
        got, want = wr[key], float(m.working_ranges[key])
        assert abs(got - want) < 0.15, f"{key}: computed {got:.3f}, published {want:.2f}"


def test_the_planar_closed_form_agrees_with_the_full_transform_stack(m):
    rng = np.random.default_rng(20260915)
    for _ in range(200):
        q = np.array([rng.uniform(-np.pi, np.pi)]
                     + [rng.uniform(*m.limits_rad(j)) for j in ("boom", "stick", "bucket")])
        tip = K.bucket_tip(m, q)
        x, z = K.tip_xz(m, q[1], q[2], q[3])
        # The planar radius is signed, so project onto the turret forward axis
        # rather than taking a magnitude; folded-back poses have x < 0.
        fwd = tip[0] * np.cos(q[0]) + tip[1] * np.sin(q[0])
        assert fwd == pytest.approx(x, abs=1e-9)
        assert tip[2] == pytest.approx(z, abs=1e-9)


def test_a_positive_boom_angle_raises_the_boom(m):
    # The opposite convention digs into the sky and still plots plausibly, so
    # the sign is pinned here rather than left to the reader of frames.py.
    low = K.bucket_tip(m, np.radians([0.0, -20.0, -90.0, -60.0]))
    high = K.bucket_tip(m, np.radians([0.0, 40.0, -90.0, -60.0]))
    assert high[2] > low[2]


def test_swing_rotates_the_tip_about_the_slew_axis_without_changing_its_radius(m):
    q0 = np.radians([0.0, 15.0, -80.0, -50.0])
    for deg in (37.0, 90.0, 213.0):
        q = q0.copy()
        q[0] = np.radians(deg)
        a, b = K.bucket_tip(m, q0), K.bucket_tip(m, q)
        assert np.hypot(*b[:2]) == pytest.approx(np.hypot(*a[:2]), abs=1e-9)
        assert b[2] == pytest.approx(a[2], abs=1e-9)
        turned = np.degrees(np.arctan2(b[1], b[0])) - np.degrees(np.arctan2(a[1], a[0]))
        assert (turned - deg + 180.0) % 360.0 - 180.0 == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# the solids the chain carries
# ---------------------------------------------------------------------------


def test_the_solids_land_on_the_published_heights_and_the_tail_swing_radius(m):
    by_name = {s.name: s for s in m.solids}
    deck = m.deck_height

    cab = by_name["cab"]
    assert deck + cab.T_link[2, 3] + 0.5 * cab.size[2] == pytest.approx(
        m.upperframe["cab_height"], abs=1e-9)

    fogs = by_name["fogs"]
    assert deck + fogs.T_link[2, 3] + 0.5 * fogs.size[2] == pytest.approx(
        m.upperframe["fogs_height"], abs=1e-9)

    cw = by_name["counterweight"]
    assert -(cw.T_link[0, 3] - 0.5 * cw.size[0]) == pytest.approx(
        m.upperframe["tail_swing_radius"], abs=1e-9)
    assert deck + cw.T_link[2, 3] - 0.5 * cw.size[2] == pytest.approx(
        m.upperframe["counterweight_clearance"], abs=1e-9)

    track = by_name["track_left"]
    assert 2 * track.T_link[1, 3] == pytest.approx(m.undercarriage["track_gauge"], abs=1e-9)


def test_the_boom_is_a_gooseneck_whose_pins_are_still_the_kinematic_length(m):
    # The bend has to be in the shape without lengthening the link, otherwise
    # the shadow moves and the reach changes with it.
    lower, upper = (s for s in m.solids if s.link == "boom")
    def end(s, sign):
        return (s.T_link @ np.array([sign * 0.5 * s.size[0], 0.0, 0.0, 1.0]))[:3]
    root, knee_a = end(lower, -1), end(lower, +1)
    knee_b, tip = end(upper, -1), end(upper, +1)

    assert np.allclose(root, [0.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(knee_a, knee_b, atol=1e-9)          # the segments meet
    assert tip[0] == pytest.approx(m.boom_len, abs=1e-9)   # pin to pin unchanged
    assert tip[2] == pytest.approx(0.0, abs=1e-9)
    assert knee_a[2] > 0.3                                  # and it actually bends


# ---------------------------------------------------------------------------
# the grid
# ---------------------------------------------------------------------------


def test_the_joint_grid_uses_the_steps_the_brief_asks_for(m):
    g = K.joint_grid(m)
    assert len(np.unique(g[:, 0])) == 24                    # swing every 15 deg
    for col, joint in ((1, "boom"), (2, "stick"), (3, "bucket")):
        vals = np.degrees(np.unique(g[:, col]))
        assert np.allclose(np.diff(vals), 10.0)             # every 10 deg
        lo, hi = m.joint_limits[joint]
        assert vals[0] == pytest.approx(lo, abs=1e-6)
        assert vals[-1] <= hi + 1e-6


def test_every_rejected_pose_is_accounted_for_by_a_named_reason(m):
    poses, counts = K.valid_poses(m, K.joint_grid(m, quick=True))
    assert counts["kept"] == len(poses)
    assert counts["kept"] + counts["below_grade"] + counts["self_collision"] == counts["total"]
    assert counts["kept"] > 0


def test_the_kept_poses_keep_the_bucket_at_or_above_grade(m):
    poses, _ = K.valid_poses(m, K.joint_grid(m, quick=True))
    rng = np.random.default_rng(7)
    for q in poses[rng.choice(len(poses), size=120, replace=False)]:
        ok, why = K.pose_is_valid(m, q)
        assert ok, why
        assert K.bucket_tip(m, q)[2] > -0.75
