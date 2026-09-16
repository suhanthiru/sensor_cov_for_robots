"""Tests for visibility: the field of view predicate, occlusion, and occupancy.

The failure this file exists to catch is a visibility routine that is confidently
wrong in a way no picture reveals.  An off-by-one in the segment query makes
every target occlude itself and coverage reads zero, which is at least obvious.
The dangerous version is subtler: a ``tfar`` that overshoots slightly, so a
sensor sees straight through thin geometry and every layout looks good.

So the checks here are against closed form answers wherever one exists.  An
empty scene must return exactly the field of view.  A box of known width at
known range must cast a shadow of exactly the width trigonometry predicts.  And
with the undercarriage removed, coverage must be invariant under a quarter turn
of swing, which is a property of the transform stack that no amount of
plausible-looking geometry can fake.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from sensorcov import kinematics as K
from sensorcov.envelope import build_envelope
from sensorcov.frames import pose_from_rpy, spherical_in_frame
from sensorcov.machine import MachineSpec, Solid
from sensorcov.raycast import MachineScene, occupied_mask
from sensorcov.sensors import Layout, SensorSpec

pytestmark = pytest.mark.raycast


@pytest.fixture(scope="module")
def m():
    return MachineSpec.load()


def scene_of(solids, ground=False):
    """A raycasting scene built from an explicit list of solids, all in world."""
    stand_in = SimpleNamespace(solids=solids)
    link_T = {"world": np.eye(4)}
    return MachineScene(stand_in, link_T, links=("world",), ground=ground)


# ---------------------------------------------------------------------------
# the field of view predicate, on its own
# ---------------------------------------------------------------------------


def test_the_fast_field_of_view_predicate_means_what_the_readable_one_means(m):
    rng = np.random.default_rng(20260915)
    pts = rng.uniform(-30.0, 30.0, size=(40000, 3)).astype(np.float32)
    T_WS = pose_from_rpy([1.0, 2.0, 3.0], roll_deg=5.0, pitch_deg=20.0, yaw_deg=-110.0)
    for name in ("spinning32", "spinning64", "solid_state", "camera"):
        spec = SensorSpec.load(name)
        r, az, el = spherical_in_frame(T_WS, pts)
        ref = spec.in_fov(r, az, el)
        fast = spec.in_fov_fast((pts - T_WS[:3, 3]) @ T_WS[:3, :3])
        assert np.array_equal(ref, fast), name
        assert ref.any() and not ref.all()


def test_an_empty_scene_returns_exactly_the_field_of_view(m):
    # Nothing to occlude means visibility and the field of view are the same
    # set.  This pins the predicate independently of any ray casting.
    env = build_envelope(radius=15.0, z_hi=6.0, cell=0.5)
    empty = scene_of([], ground=False)
    T_WS = pose_from_rpy([0.0, 0.0, 3.0], pitch_deg=10.0, yaw_deg=25.0)
    for name in ("spinning32", "solid_state", "camera"):
        spec = SensorSpec.load(name)
        visible = empty.line_of_sight(T_WS, spec, env.centers_W)
        expect = spec.in_fov_fast((env.centers_W - T_WS[:3, 3]) @ T_WS[:3, :3])
        assert np.array_equal(visible, expect), name


# ---------------------------------------------------------------------------
# occlusion, against trigonometry
# ---------------------------------------------------------------------------


def test_a_box_casts_the_shadow_that_trigonometry_predicts():
    # A half-metre-wide plate at 5 m shadows a half-angle of atan(0.5 / 5), so
    # at 10 m the shadow is 1.0 m to either side of the axis.  Anything wider
    # than that boundary means tfar is overshooting and sensors are seeing
    # through geometry; anything narrower means they are self-occluding.
    plate = Solid("plate", "world", "box", (0.05, 1.0, 1.0),
                  pose_from_rpy([5.0, 0.0, 0.0]))
    scene = scene_of([plate])
    spec = SensorSpec.load("solid_state")
    T_WS = np.eye(4)

    y = np.linspace(-2.0, 2.0, 801)
    pts = np.stack([np.full_like(y, 10.0), y, np.zeros_like(y)], axis=1).astype(np.float32)
    visible = scene.line_of_sight(T_WS, spec, pts)

    predicted = 10.0 * np.tan(np.arctan(0.5 / 5.0))      # 1.0 m
    in_shadow = np.abs(y) < predicted - 0.01
    in_light = np.abs(y) > predicted + 0.01
    assert not visible[in_shadow].any()
    assert visible[in_light].all()


def test_an_occluder_beyond_the_target_does_not_occlude_it():
    # The segment query has to stop at the target.  A wall behind the target is
    # the case that distinguishes a segment query from an ordinary ray cast,
    # and getting it wrong would make everything past the first surface blind.
    wall = Solid("wall", "world", "box", (0.2, 40.0, 40.0), pose_from_rpy([12.0, 0.0, 0.0]))
    scene = scene_of([wall])
    spec = SensorSpec.load("solid_state")
    pts = np.array([[8.0, 0.0, 0.0], [15.0, 0.0, 0.0]], dtype=np.float32)
    visible = scene.line_of_sight(np.eye(4), spec, pts)
    assert visible[0], "target in front of the wall should be visible"
    assert not visible[1], "target behind the wall should be occluded"


def test_the_ground_plane_does_not_occlude_a_sensor_looking_down_at_it():
    # A sensor on the cab looking at a cell just above grade must not be blocked
    # by the ground it is looking at.  If the plane is placed a hair too high
    # this fails, and every layout loses its near-field coverage silently.
    scene = scene_of([], ground=True)
    spec = SensorSpec.load("spinning32")
    T_WS = pose_from_rpy([0.0, 0.0, 3.0])
    r = np.arange(1.0, 15.0, 0.25)
    pts = np.stack([r, np.zeros_like(r), np.full_like(r, 0.125)], axis=1).astype(np.float32)

    # Compared against the field of view rather than against True, because the
    # near cells here are genuinely below this sensor's vertical window and
    # asserting they are visible would be asserting the wrong thing.  What is
    # being tested is that adding the ground removes nothing.
    visible = scene.line_of_sight(T_WS, spec, pts)
    expect = spec.in_fov_fast((pts - T_WS[:3, 3]) @ T_WS[:3, :3])
    assert np.array_equal(visible, expect)
    assert expect.any()


# ---------------------------------------------------------------------------
# the invariant that checks the whole transform stack at once
# ---------------------------------------------------------------------------


def test_coverage_is_invariant_under_a_quarter_turn_of_swing(m):
    # Everything above the slew bearing turns together, so with the tracks and
    # car body left out of the scene the coverage pattern can only rotate.  A
    # quarter turn maps the cubic grid exactly onto itself, so the counts must
    # match to the cell, not approximately.  Any error in the order of the
    # transforms, or a link parented to the wrong frame, breaks this.
    env = build_envelope(radius=12.0, z_hi=4.0, cell=0.5)
    upper = ("turret", "boom", "stick", "bucket")
    layout = Layout.load("d_cab_and_boom")
    base = None
    for swing_deg in (0.0, 90.0, 180.0, 270.0):
        q = np.radians([swing_deg, 12.0, -95.0, -60.0])
        link_T = K.link_poses(m, q)
        scene = MachineScene(m, link_T, links=upper, ground=True)
        seen = np.zeros(env.n, dtype=bool)
        for mo, T_WS in layout.poses_W(link_T):
            seen |= scene.line_of_sight(T_WS, mo.sensor, env.centers_W)
        if base is None:
            base = int(seen.sum())
            # The machine must actually be occluding something, or this test
            # would pass just as well on a scene that had lost its geometry.
            empty = scene_of([], ground=True)
            free = np.zeros(env.n, dtype=bool)
            for mo, T_WS in layout.poses_W(link_T):
                free |= empty.line_of_sight(T_WS, mo.sensor, env.centers_W)
            assert base < 0.9 * int(free.sum())
        else:
            assert int(seen.sum()) == base, f"swing {swing_deg} broke the symmetry"


def _coverage_by_swing(m, layout, boom_deg, links=None, cell=0.5):
    env = build_envelope(radius=12.0, z_hi=4.0, cell=cell)
    counts = []
    for swing_deg in (0.0, 90.0, 180.0, 270.0):
        q = np.radians([swing_deg, boom_deg, -95.0, -60.0])
        link_T = K.link_poses(m, q)
        kw = {} if links is None else {"links": links}
        scene = MachineScene(m, link_T, ground=True, **kw)
        seen = np.zeros(env.n, dtype=bool)
        for mo, T_WS in layout.poses_W(link_T):
            seen |= scene.line_of_sight(T_WS, mo.sensor, env.centers_W)
        counts.append(int(seen.sum()))
    return counts


def test_a_sensor_that_drops_below_the_tracks_is_what_breaks_that_symmetry(m):
    # The companion to the test above, and a real result rather than a formality.
    # The undercarriage is the only thing in the scene that does not turn with
    # the house, so it is the only thing that can make swing matter.  It only
    # occludes a sensor that is below the top of the tracks, and the single
    # sensor in this study that ever gets there is layout D's boom unit at full
    # boom-down, which drops to 0.71 m against a 0.98 m track.  That is exactly
    # where D earns its extra coverage, so it is worth knowing it is also the
    # only place the tracks cost it any.
    layout = Layout.load("d_cab_and_boom")
    boom_lo = m.joint_limits["boom"][0]

    assert len(set(_coverage_by_swing(m, layout, boom_lo))) > 1
    upper = ("turret", "boom", "stick", "bucket")
    assert len(set(_coverage_by_swing(m, layout, boom_lo, links=upper))) == 1


def test_the_undercarriage_never_occludes_a_cab_mounted_sensor(m):
    # Stated as a test because it is the reason the symmetry above holds so
    # cleanly, and because it would be easy to assume the tracks matter and
    # quietly carry them through every cost estimate.  A sensor on the cab roof
    # looks down over the tracks from 3 m; the sight line to any cell in the
    # envelope has already cleared them.
    upper = ("turret", "boom", "stick", "bucket")
    layout = Layout.load("a_cab_corners")
    for boom_deg in (m.joint_limits["boom"][0], -20.0, 40.0):
        full = _coverage_by_swing(m, layout, boom_deg)
        part = _coverage_by_swing(m, layout, boom_deg, links=upper)
        assert full == part, f"tracks changed coverage at boom {boom_deg}"


# ---------------------------------------------------------------------------
# occupancy
# ---------------------------------------------------------------------------


def test_points_inside_a_link_are_reported_occupied_not_blind(m):
    # The cell the boom is currently filling is solid, not unseen.  Counting it
    # as blind volume would make the headline transient number an artefact of
    # the boom being where the boom is.
    q = np.radians([0.0, 5.0, -100.0, -60.0])
    link_T = K.link_poses(m, q)
    by_name = {s.name: s for s in m.solids}

    centres = []
    for name in ("boom_lower", "boom_upper", "stick", "cab", "counterweight"):
        s = by_name[name]
        centres.append((link_T[s.link] @ s.T_link)[:3, 3])
    assert occupied_mask(m, link_T, np.array(centres)).all()

    # and a point well clear of the machine is not
    assert not occupied_mask(m, link_T, np.array([[14.0, 14.0, 5.0]]))[0]


def test_occupancy_follows_the_boom_as_it_moves(m):
    # The set of solid cells has to be recomputed per pose.  If it were computed
    # once and reused, this would come back identical for both configurations.
    env = build_envelope(radius=12.0, z_hi=6.0, cell=0.5)
    low = occupied_mask(m, K.link_poses(m, np.radians([0.0, -30.0, -60.0, -60.0])),
                        env.centers_W)
    high = occupied_mask(m, K.link_poses(m, np.radians([0.0, 55.0, -150.0, -60.0])),
                         env.centers_W)
    assert low.sum() > 0 and high.sum() > 0
    assert not np.array_equal(low, high)
