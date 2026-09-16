"""Tests for the detection metric, and for where the sensors are bolted.

Two classes of bug live here, and both of them produce output that looks fine.

The first is an asymmetry in the beam bookkeeping.  Beam azimuths for a spinning
unit are generated over [0, 2pi) while a target azimuth arrives from arctan2 in
(-pi, pi], and comparing the two without wrapping silently returns zero beams for
every target on one side of the machine.  Nothing crashes; the layout just
reports a large blind sector that is not there.  So the first test here walks a
target all the way round an unobstructed sensor and insists on returns from every
direction.

The second is a mount buried in the machine.  Sensor positions are written by
hand in the layout files against a geometry with non-obvious extents: the boom
box spans only 0.62 m across, so a mount that looks like it clears the cab can
sit directly over the boom, and a cab-front mount can be swept by a bucket that
is 1.15 m wide.  A buried sensor does not error, it just scores badly, and the
trade study then compares a real layout against a broken one.
"""

from __future__ import annotations

import numpy as np
import pytest

from sensorcov import kinematics as K
from sensorcov.detect import DetectionConfig, camera_pixels, detected, lidar_returns
from sensorcov.machine import MachineSpec
from sensorcov.raycast import MachineScene, occupied_mask
from sensorcov.sensors import Layout, SensorSpec
from tests.sensorcov.test_raycast import scene_of

pytestmark = pytest.mark.raycast

LAYOUTS = ("a_cab_corners", "b_roof_and_cameras", "c_corner_solid_state", "d_cab_and_boom")


@pytest.fixture(scope="module")
def m():
    return MachineSpec.load()


@pytest.fixture(scope="module")
def cfg():
    return DetectionConfig()


# ---------------------------------------------------------------------------
# the beam bookkeeping
# ---------------------------------------------------------------------------


def test_a_spinning_lidar_gets_returns_from_targets_on_every_side(cfg):
    # The regression test for the azimuth wrap.  With nothing but ground in the
    # scene and the sensor on the slew axis, every candidate position is
    # equivalent by symmetry, so any direction reporting zero returns is a
    # bookkeeping error rather than a fact about the machine.
    scene = scene_of([], ground=True)
    spec = SensorSpec.load("spinning32")
    T_WS = np.eye(4)
    T_WS[2, 3] = 3.0
    counts = lidar_returns(scene, T_WS, spec, cfg.targets_W(), cfg)
    assert (counts > 0).all(), "some azimuths returned no beams at all"
    assert (counts >= cfg.min_returns).all()


def test_the_beam_count_is_what_scales_the_returns(cfg):
    # Same optic, same window, twice the channels.  If the beam table were not
    # actually being resolved, these two would agree.
    scene = scene_of([], ground=True)
    T_WS = np.eye(4)
    T_WS[2, 3] = 3.0
    targets = cfg.targets_W()
    a = lidar_returns(scene, T_WS, SensorSpec.load("spinning32"), targets, cfg)
    b = lidar_returns(scene, T_WS, SensorSpec.load("spinning64"), targets, cfg)
    assert b.sum() > 1.7 * a.sum()


def test_returns_thin_out_with_range(cfg):
    scene = scene_of([], ground=True)
    spec = SensorSpec.load("spinning32")
    T_WS = np.eye(4)
    T_WS[2, 3] = 3.0
    targets = cfg.targets_W()
    counts = lidar_returns(scene, T_WS, spec, targets, cfg)
    r = np.hypot(targets[:, 0], targets[:, 1])
    near = counts[np.isclose(r, cfg.ring_radii[0])].mean()
    far = counts[np.isclose(r, cfg.ring_radii[-1])].mean()
    assert near > far


def test_a_target_outside_the_standoff_band_is_not_scored(cfg):
    # The brief asks about targets from 5 m out.  Anything nearer is a different
    # question, and quietly scoring it would flatter every layout.
    scene = scene_of([], ground=True)
    spec = SensorSpec.load("spinning32")
    T_WS = np.eye(4)
    T_WS[2, 3] = 3.0
    inside = np.array([[2.0, 0.0], [cfg.max_range_m + 4.0, 0.0]])
    assert (lidar_returns(scene, T_WS, spec, inside, cfg) == 0).all()


def test_the_machine_is_what_blocks_the_rest(m, cfg):
    # The counterweight is 2.6 m across and stands off the tail; a target
    # directly behind it must be shadowed, and the same target must come back
    # once the machine is taken out of the scene.
    q = np.radians([0.0, 20.0, -100.0, -60.0])
    link_T = K.link_poses(m, q)
    spec = SensorSpec.load("spinning32")
    T_WS = np.array(link_T["turret"]) @ np.eye(4)
    T_WS[:3, 3] = link_T["turret"][:3, 3] + np.array([1.8, 0.85, 0.95])

    behind = np.array([[-8.0, 0.0]])
    with_machine = lidar_returns(MachineScene(m, link_T), T_WS, spec, behind, cfg)
    without = lidar_returns(scene_of([], ground=True), T_WS, spec, behind, cfg)
    assert with_machine[0] == 0
    assert without[0] >= cfg.min_returns


def test_a_camera_is_scored_on_pixels_and_falls_off_with_range(cfg):
    scene = scene_of([], ground=True)
    spec = SensorSpec.load("camera")
    T_WS = np.eye(4)
    T_WS[2, 3] = 3.0
    targets = np.array([[6.0, 0.0], [12.0, 0.0], [25.0, 0.0]])
    px = camera_pixels(scene, T_WS, spec, targets, cfg)
    assert px[0] > px[1]
    assert px[2] == 0.0, "beyond the useful range the camera should not count"


def test_detection_is_any_sensor_not_all_of_them(m, cfg):
    q = np.radians([0.0, 15.0, -95.0, -60.0])
    link_T = K.link_poses(m, q)
    scene = MachineScene(m, link_T)
    layout = Layout.load("a_cab_corners")
    poses = layout.poses_W(link_T)
    any_ = detected(scene, poses, cfg, cfg.targets_W())
    per = [lidar_returns(scene, T, mo.sensor, cfg.targets_W(), cfg) >= cfg.min_returns
           for mo, T in poses]
    assert np.array_equal(any_, per[0] | per[1])


# ---------------------------------------------------------------------------
# where the sensors are bolted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout_name", LAYOUTS)
def test_no_sensor_is_buried_inside_the_machine_at_any_pose(m, layout_name):
    # Written after finding three of them.  The boom box is only 0.62 m across
    # and the bucket 1.15 m, so a mount can look clear in plan and still be
    # swept; a buried sensor scores badly rather than failing, which makes the
    # whole trade study quietly meaningless.
    layout = Layout.load(layout_name)
    poses, _ = K.valid_poses(m, K.joint_grid(m, quick=True))
    rng = np.random.default_rng(20260915)
    sample = poses[rng.choice(len(poses), size=90, replace=False)]

    for q in sample:
        link_T = K.link_poses(m, q)
        scene = MachineScene(m, link_T)
        for mo, T_WS in layout.poses_W(link_T):
            origin = T_WS[:3, 3][None, :]
            assert not occupied_mask(m, link_T, origin)[0], \
                f"{layout_name}/{mo.label} is inside the machine at {np.degrees(q).round(1)}"
            gap = float(scene.distance_to(origin.astype(np.float32))[0])
            assert gap >= 0.08, \
                f"{layout_name}/{mo.label} is {gap:.3f} m from a surface"
