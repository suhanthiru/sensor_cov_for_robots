"""Tests for the accumulators, the machine frame, and the persistent/transient split.

The bug this file is mostly about is not a crash but a choice of frame.  Scored
in the world, a turret-mounted sensor sweeps past every fixed cell as the machine
slews, so nearly every cell is seen in some configurations and missed in others
and the classification collapses to "almost everything is transient" whatever the
layout is.  The numbers look reasonable and mean nothing.  Scored in the machine
frame the swing drops out for anything bolted to the turret, and what is left is
the articulation effect the study exists to measure.  So the tests here pin the
frame down directly: a turret mount must land in the same place in M no matter
what the swing angle is.

The rest is bookkeeping that has to hold for the volumes to add up: a cell cannot
be seen more often than it was free, two sensors cannot see it more often than
one, and every free cell has to land in exactly one of the three classes.
"""

from __future__ import annotations

import numpy as np
import pytest

from sensorcov import kinematics as K
from sensorcov.detect import DetectionConfig
from sensorcov.envelope import build_envelope
from sensorcov.machine import MachineSpec
from sensorcov.metrics import classify, compute, pareto_front
from sensorcov.sensors import Layout
from sensorcov.sweep import run_sweep, score_pose

pytestmark = pytest.mark.raycast


@pytest.fixture(scope="module")
def m():
    return MachineSpec.load()


@pytest.fixture(scope="module")
def small_sweep():
    # Deliberately coarse: this is about the bookkeeping, not the physics.
    return run_sweep("d_cab_and_boom", quick=True, workers=1, progress=False,
                     det_cfg=DetectionConfig(n_azimuth=12, ring_radii=(7.0, 12.0)))


# ---------------------------------------------------------------------------
# the frame
# ---------------------------------------------------------------------------


def test_a_turret_mounted_sensor_does_not_move_in_the_machine_frame(m):
    # The definition of the machine frame, checked directly.  If this drifts,
    # every persistent blind spot smears into a transient one.
    from sensorcov.frames import rot_z

    layout = Layout.load("a_cab_corners")
    base = None
    for swing_deg in (0.0, 37.0, 150.0, 285.0):
        q = np.radians([swing_deg, 12.0, -95.0, -60.0])
        link_T = K.link_poses(m, q)
        link_T = {k: rot_z(-float(q[0])) @ v for k, v in link_T.items()}
        got = np.array([T[:3, 3] for _, T in layout.poses_W(link_T)])
        if base is None:
            base = got
        else:
            assert np.allclose(got, base, atol=1e-9)


def test_coverage_barely_moves_with_swing_for_a_cab_only_layout(m):
    # The companion fact, and the reason the frame choice works: once the swing
    # is out, the only thing left that turns is the undercarriage, and it does
    # not occlude a sensor three metres up.  A few tenths of a percent, not tens.
    env = build_envelope(radius=12.0, z_hi=4.0, cell=0.5)
    layout = Layout.load("a_cab_corners")
    cov = []
    for swing_deg in (0.0, 45.0, 90.0, 180.0, 270.0):
        q = np.radians([swing_deg, 12.0, -95.0, -60.0])
        free, seen, dual, _ = score_pose(m, layout, env, q)
        cov.append(seen.sum() / free.sum())
    assert max(cov) - min(cov) < 0.01


def test_articulation_is_what_moves_coverage_instead(m):
    # And the contrast: with swing held fixed, working the boom through its
    # range must move coverage by much more than swing ever did.  Without this
    # the test above would also pass on a layout that saw nothing at all.
    env = build_envelope(radius=12.0, z_hi=4.0, cell=0.5)
    layout = Layout.load("a_cab_corners")
    cov = []
    for boom_deg in np.linspace(*m.joint_limits["boom"], 5):
        q = np.radians([0.0, boom_deg, -95.0, -60.0])
        free, seen, dual, _ = score_pose(m, layout, env, q)
        cov.append(seen.sum() / free.sum())
    assert max(cov) - min(cov) > 0.03


# ---------------------------------------------------------------------------
# the counters
# ---------------------------------------------------------------------------


def test_the_counters_cannot_contradict_each_other(small_sweep):
    res = small_sweep
    n = len(res.poses)
    assert (res.free_count <= n).all()
    assert (res.seen_count <= res.free_count).all(), "seen more often than free"
    assert (res.dual_count <= res.seen_count).all(), "two sensors saw it more often than one"
    assert (res.detect_count <= n).all()
    assert res.ever_seen[res.seen_count > 0].all()
    assert not res.ever_seen[res.seen_count == 0].any()


def test_every_free_cell_lands_in_exactly_one_class(small_sweep):
    # persistent, transient and always-seen have to partition the free cells,
    # or the volumes in the report do not add up to the envelope.
    res = small_sweep
    persistent, transient, severity, ever_free = classify(res)
    always = ever_free & (res.seen_count == res.free_count)
    assert not (persistent & transient).any()
    assert not (persistent & always).any()
    assert not (transient & always).any()
    assert np.array_equal(persistent | transient | always, ever_free)

    env = res.envelope
    met = compute(res)
    total = met.persistent_blind_m3 + met.transient_blind_m3 + met.always_seen_m3
    assert total == pytest.approx(met.envelope_free_m3, rel=1e-9)
    assert met.envelope_free_m3 <= env.n * env.cell_volume


def test_severity_is_zero_where_a_cell_was_always_seen(small_sweep):
    persistent, transient, severity, ever_free = classify(small_sweep)
    always = ever_free & (small_sweep.seen_count == small_sweep.free_count)
    assert np.allclose(severity[always], 0.0)
    assert np.allclose(severity[persistent], 1.0)
    assert ((severity > 0.0) & (severity < 1.0))[transient].all()


def test_the_boom_fills_cells_that_are_solid_rather_than_blind(small_sweep):
    # Some cell in the envelope must have been inside the machine at some pose,
    # or the occupancy exclusion is not doing anything and the transient number
    # is inflated by the boom's own volume.
    assert (small_sweep.free_count < len(small_sweep.poses)).any()


def test_the_sweep_is_deterministic(m):
    a = run_sweep("a_cab_corners", quick=True, workers=1, progress=False,
                  det_cfg=DetectionConfig(n_azimuth=8, ring_radii=(10.0,)))
    b = run_sweep("a_cab_corners", quick=True, workers=1, progress=False,
                  det_cfg=DetectionConfig(n_azimuth=8, ring_radii=(10.0,)))
    assert np.array_equal(a.seen_count, b.seen_count)
    assert np.array_equal(a.detect_count, b.detect_count)
    assert np.allclose(a.per_pose_cov, b.per_pose_cov)


def test_workers_do_not_change_the_answer(m):
    # Chunks are absorbed out of order, so anything order-dependent in the
    # accumulation would show up here and nowhere else.
    kw = dict(quick=True, progress=False,
              det_cfg=DetectionConfig(n_azimuth=8, ring_radii=(10.0,)))
    one = run_sweep("a_cab_corners", workers=1, **kw)
    many = run_sweep("a_cab_corners", workers=4, **kw)
    assert np.array_equal(one.free_count, many.free_count)
    assert np.array_equal(one.seen_count, many.seen_count)
    assert np.array_equal(one.dual_count, many.dual_count)
    assert np.array_equal(one.detect_count, many.detect_count)
    assert np.allclose(one.per_pose_cov, many.per_pose_cov)


def test_a_sweep_survives_a_round_trip_to_disk(small_sweep, tmp_path):
    path = tmp_path / "s.npz"
    small_sweep.save(path)
    back = type(small_sweep).load(path)
    assert back.layout == small_sweep.layout
    assert back.grid_counts == small_sweep.grid_counts
    assert back.env_kwargs == small_sweep.env_kwargs
    assert np.array_equal(back.seen_count, small_sweep.seen_count)
    assert compute(back).row() == compute(small_sweep).row()


# ---------------------------------------------------------------------------
# the front
# ---------------------------------------------------------------------------


def test_the_pareto_front_keeps_what_nothing_beats_everywhere():
    rows = [
        {"layout": "cheap_bad", "coverage_mean": 50.0, "n_sensors": 2, "transient_blind_m3": 900.0},
        {"layout": "dominated", "coverage_mean": 48.0, "n_sensors": 4, "transient_blind_m3": 950.0},
        {"layout": "rich_good", "coverage_mean": 80.0, "n_sensors": 5, "transient_blind_m3": 300.0},
    ]
    front = {r["layout"] for r in pareto_front(rows)}
    assert "dominated" not in front
    assert front == {"cheap_bad", "rich_good"}
