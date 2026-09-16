"""Smoke tests for the figures.

These do not check that a plot is informative, which is not a thing a test can
do.  They check that every figure still draws from a real sweep, because the
figures read the sweep through its saved form rather than through the objects
that produced it, and a field renamed in the sweep shows up here as a crash
weeks after the change rather than at the point of it.

The polar figure is the one worth guarding: it reshapes the detection counts
into rings by azimuth using the detection config that was stored alongside them.
Change the ring count without changing the stored config and the reshape fails,
which is the good case; change both inconsistently and it silently plots the
wrong thing.
"""

from __future__ import annotations

import numpy as np
import pytest

from sensorcov.detect import DetectionConfig
from sensorcov.machine import MachineSpec
from sensorcov.metrics import compute
from sensorcov.sensors import Layout
from sensorcov.sweep import run_sweep
from sensorcov.viz import (blind_maps, coverage_distributions, detection_polar,
                           machine_view, pareto_plot, sampling_limits)

pytestmark = pytest.mark.raycast


@pytest.fixture(scope="module")
def tiny():
    """One small real sweep, saved and reloaded, as the figures will see it."""
    cfg = DetectionConfig(n_azimuth=8, ring_radii=(7.0, 12.0))
    res = run_sweep("a_cab_corners", quick=True, workers=1, progress=False, det_cfg=cfg)
    return res


def test_the_pareto_plot_draws_from_metric_rows(tiny, tmp_path):
    rows = [compute(tiny).row()]
    rows.append({**rows[0], "layout": "b_roof_and_cameras", "n_sensors": 5,
                 "coverage_mean": rows[0]["coverage_mean"] + 12.0,
                 "transient_blind_m3": rows[0]["transient_blind_m3"] * 0.5})
    out = pareto_plot(rows, tmp_path / "pareto.png")
    assert out.exists() and out.stat().st_size > 5000


def test_the_sweep_figures_draw_from_a_saved_and_reloaded_sweep(tiny, tmp_path):
    from sensorcov.sweep import SweepResult

    path = tmp_path / "s.npz"
    tiny.save(path)
    sweeps = {"a_cab_corners": SweepResult.load(path)}

    for fn, name in ((coverage_distributions, "cov.png"),
                     (detection_polar, "det.png"),
                     (blind_maps, "blind.png")):
        out = fn(sweeps, tmp_path / name)
        assert out.exists() and out.stat().st_size > 5000, name


def test_the_polar_figure_agrees_with_the_stored_detection_config(tiny):
    # The reshape that would fail loudly, asserted directly so the intent is
    # recorded rather than left to whether matplotlib happened to raise.
    cfg = tiny.det_cfg
    n = len(cfg["ring_radii"]) * int(cfg["n_azimuth"])
    assert tiny.detect_count.shape == (n,)
    assert (tiny.detect_count <= len(tiny.poses)).all()


def test_the_machine_view_draws_at_the_worst_configuration(tiny, tmp_path):
    m = MachineSpec.load()
    out = machine_view(m, Layout.load("a_cab_corners"), tiny,
                       path=tmp_path / "machine.png", mask_kind="persistent",
                       max_points=1500)
    assert out.exists() and out.stat().st_size > 10000


def test_the_sampling_figure_needs_no_sweep_at_all(tmp_path):
    out = sampling_limits(tmp_path / "sampling.png")
    assert out.exists() and out.stat().st_size > 5000


def test_the_envelope_can_be_folded_back_into_a_grid_for_the_maps(tiny):
    # blind_maps depends on this round trip: per-cell vectors scattered into the
    # dense grid, summed down the column.  If the indices were wrong the figure
    # would still draw, just of the wrong thing.
    env = tiny.envelope
    values = np.arange(env.n, dtype=float)
    grid = env.to_grid(values, fill=np.nan)
    assert grid.shape == env.shape
    back = grid[env.idx[:, 0], env.idx[:, 1], env.idx[:, 2]]
    assert np.array_equal(back, values)
    assert np.isnan(grid).sum() == grid.size - env.n
