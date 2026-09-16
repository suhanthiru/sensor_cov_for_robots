"""Figures.

Matplotlib throughout, including the three-dimensional views.  Open3D has an
offscreen renderer and it would look better, but it needs EGL headless, which
this platform does not provide, and a figure that only renders on some machines
is not much use in a repository.

The plots are chosen to show the thing the table cannot.  A coverage percentage
is a single number over a whole sweep; what matters is its shape, so the
distribution is drawn with the worst configuration marked on it.  A blind volume
in cubic metres says nothing about where it is, so persistent and transient
volume are drawn from above, in metres of blind column.  And the detection
metric is really a map, not a scalar: the polar figure shows, for every place a
person might stand, the fraction of the machine's configurations in which they
would be seen.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from . import kinematics as K  # noqa: E402
from .frames import rot_z  # noqa: E402
from .metrics import classify  # noqa: E402

FIGURES = Path(__file__).resolve().parent.parent / "figures"

LABEL = {"a_cab_corners": "A", "b_roof_and_cameras": "B",
         "c_corner_solid_state": "C", "d_cab_and_boom": "D"}

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


# ---------------------------------------------------------------------------
# the trade study
# ---------------------------------------------------------------------------


def pareto_plot(rows, path=None):
    """Coverage against sensor count, coloured by transient blind volume.

    The plot the brief asks for.  The colour axis is the point of it: two
    layouts can sit at the same coverage and the same cost and differ entirely
    in how much of their blind volume moves around, which is the part that a
    static analysis would have reported as fine.
    """
    path = path or FIGURES / "pareto.png"
    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    tr = np.array([r["transient_blind_m3"] for r in rows])
    cov = np.array([r["coverage_mean"] for r in rows])
    n = np.array([r["n_sensors"] for r in rows])
    worst = np.array([r["coverage_worst"] for r in rows])

    ax.vlines(n, worst, cov, color="0.75", lw=1.2, zorder=1)
    sc = ax.scatter(n, cov, c=tr, s=190, cmap="magma_r",
                    norm=Normalize(tr.min() * 0.9, tr.max() * 1.05),
                    edgecolor="black", linewidth=0.8, zorder=3)
    ax.scatter(n, worst, facecolor="white", edgecolor="0.5", s=42,
               marker="v", zorder=3)

    for r, x, y in zip(rows, n, cov):
        ax.annotate(LABEL.get(r["layout"], r["layout"]), (x, y),
                    textcoords="offset points", xytext=(11, 5),
                    fontsize=11, fontweight="bold")

    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("transient blind volume (m$^3$)")
    ax.set_xlabel("sensors")
    ax.set_ylabel("coverage of the free envelope (%)")
    ax.set_title("Coverage against sensor count\n"
                 "circle: mean over the sweep,  triangle: worst configuration",
                 fontsize=9.5, loc="left")
    ax.set_xticks(sorted(set(n.tolist())))
    ax.margins(x=0.18)
    return _save(fig, path)


def coverage_distributions(sweeps, path=None):
    """Per-pose coverage for every layout, with the worst configuration marked.

    A mean hides whether a layout is steady or whether it collapses in a few
    configurations, and for a safety case the second is what matters.
    """
    path = path or FIGURES / "coverage_distributions.png"
    names = list(sweeps)
    fig, axes = plt.subplots(len(names), 1, figsize=(6.4, 1.5 * len(names) + 1.1),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]

    for ax, name in zip(axes, names):
        res = sweeps[name]
        cov = 100.0 * res.per_pose_cov
        person = 100.0 * res.per_pose_person
        ax.hist(cov, bins=60, color="#3b6ea5", alpha=0.85, label="whole envelope")
        ax.hist(person, bins=60, color="#c4622d", alpha=0.55, label="0 to 2 m band")
        ax.axvline(cov.min(), color="black", lw=1.2, ls="--")
        ax.annotate(f"worst {cov.min():.1f}%", (cov.min(), 0.92),
                    xycoords=("data", "axes fraction"),
                    textcoords="offset points", xytext=(4, 0), fontsize=8)
        ax.set_ylabel(LABEL.get(name, name), rotation=0, labelpad=14,
                      fontweight="bold", fontsize=12, va="center")
        ax.set_yticks([])
    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.9)
    axes[-1].set_xlabel("coverage in one configuration (%)")
    axes[0].set_title("How coverage varies across the articulation sweep",
                      fontsize=9.5, loc="left")
    return _save(fig, path)


def detection_polar(sweeps, path=None):
    """Where a standing person is seen, and how often.

    Each wedge is one candidate standing position, shaded by the fraction of the
    machine's configurations in which at least three beams come back off them.
    Dark is never, in any configuration, which is the number that matters: it is
    a place a person can stand and not be seen whatever the machine is doing.
    The machine faces the top of each dial.
    """
    path = path or FIGURES / "detection.png"
    names = list(sweeps)
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 4.0),
                             subplot_kw={"projection": "polar"}, squeeze=False)
    axes = axes[0, :]
    fig.subplots_adjust(wspace=0.45)

    for i, (ax, name) in enumerate(zip(axes, names)):
        res = sweeps[name]
        cfg = res.det_cfg
        radii = np.asarray(cfg["ring_radii"], dtype=float)
        n_az = int(cfg["n_azimuth"])
        n_pose = max(len(res.poses), 1)
        rate = (res.detect_count.astype(float) / n_pose).reshape(len(radii), n_az)

        th = np.arange(n_az + 1) * (2 * np.pi / n_az)
        edges = np.concatenate([[radii[0] - 1.25],
                                0.5 * (radii[:-1] + radii[1:]), [radii[-1] + 1.25]])
        TH, R = np.meshgrid(th, edges)
        pm = ax.pcolormesh(TH, R, rate, cmap="viridis", vmin=0.0, vmax=1.0,
                           shading="flat")
        ax.set_theta_zero_location("N")
        never = int((res.detect_count == 0).sum())
        ax.set_title(f"{LABEL.get(name, name)}\n{never} of {rate.size} never seen",
                     fontweight="bold", fontsize=10, pad=12)

        # Radial labels off the vertical so they do not stack under the title,
        # and compass labels only on the first dial: repeating them four times
        # across a tight row is what made them collide with the neighbour.
        # Only the inner and outer rings are labelled.  Five labels on a dial
        # this size overlap each other and say nothing the caption does not.
        ax.set_rlabel_position(292.5)
        ax.set_yticks(radii)
        ax.set_yticklabels([f"{r:.0f} m" if r in (radii[0], radii[-1]) else ""
                            for r in radii], fontsize=7, color="0.92")
        ax.set_xticks(np.radians([0, 90, 180, 270]))
        if i == 0:
            ax.set_xticklabels(["front", "left", "rear", "right"], fontsize=7.5)
        else:
            ax.set_xticklabels(["", "", "", ""])
        ax.tick_params(axis="x", pad=1.5)
        ax.grid(alpha=0.25, color="white", lw=0.5)

    cb = fig.colorbar(pm, ax=list(axes), fraction=0.02, pad=0.03)
    cb.set_label("configurations in which a person standing here is detected",
                 fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle("A 1.7 m person on the ground, 5 to 15 m out, needing three returns\n"
                 "machine frame: the turret faces the top of each dial",
                 fontsize=9.5, y=1.04)
    return _save(fig, path)


# ---------------------------------------------------------------------------
# where the blind volume is
# ---------------------------------------------------------------------------


def blind_maps(sweeps, path=None):
    """Persistent and transient blind volume from above, in metres of blind column.

    Summed down each column rather than sliced at one height, so a thin blind
    layer and a blind column of the same footprint do not look alike.  The
    machine is drawn on top at its nominal pose for scale.  The frame is the
    machine frame, so the turret always faces +x.
    """
    path = path or FIGURES / "blind_maps.png"
    names = list(sweeps)
    # squeeze=False keeps the grid two-dimensional for a single layout too.
    # Without it matplotlib returns a flat pair of axes and atleast_2d makes it
    # one row of two rather than two rows of one, which transposes the figure.
    fig, axes = plt.subplots(2, len(names), figsize=(2.9 * len(names), 6.0),
                             sharex=True, sharey=True, squeeze=False)

    for col, name in enumerate(names):
        res = sweeps[name]
        env = res.envelope
        persistent, transient, severity, ever_free = classify(res)

        maps = [("persistent", env.to_grid(persistent, fill=0.0).sum(axis=2) * env.cell,
                 "Reds"),
                ("transient", env.to_grid(np.where(transient, severity, 0.0),
                                          fill=0.0).sum(axis=2) * env.cell, "Blues")]
        for row, (title, grid, cmap) in enumerate(maps):
            ax = axes[row, col]
            im = ax.imshow(grid.T, origin="lower", extent=env.extent, cmap=cmap,
                           vmin=0.0, vmax=max(grid.max(), 1e-6))
            ax.set_aspect("equal")
            ax.grid(False)
            th = np.linspace(0, 2 * np.pi, 200)
            ax.plot(env.radius * np.cos(th), env.radius * np.sin(th),
                    color="0.4", lw=0.8)
            _draw_plan(ax, res)
            if row == 0:
                ax.set_title(LABEL.get(name, name), fontweight="bold", fontsize=12)
            if col == 0:
                ax.set_ylabel(f"{title}\n\ny (m)", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("x (m)")
    fig.suptitle("Blind volume from above, metres of blind column\n"
                 "machine frame: the turret faces +x",
                 fontsize=9.5, y=0.97)
    return _save(fig, path)


def _draw_plan(ax, res, boom_deg=15.0):
    """The machine's footprint, for scale on a top-down map."""
    from .machine import MachineSpec

    m = MachineSpec.load()
    q = np.radians([0.0, boom_deg, -95.0, -60.0])
    link_T = K.link_poses(m, q)
    for s in m.solids:
        T = np.asarray(link_T[s.link], dtype=float) @ s.T_link
        h = 0.5 * np.asarray(s.size, dtype=float)
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * h[:2]
        pts = corners @ T[:2, :2].T + T[:2, 3]
        ax.fill(pts[:, 0], pts[:, 1], facecolor="0.25", edgecolor="none",
                alpha=0.55, zorder=4)


# ---------------------------------------------------------------------------
# the machine, in three dimensions
# ---------------------------------------------------------------------------


def _box_faces(size, T):
    h = 0.5 * np.asarray(size, dtype=float)
    c = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * h
    c = c @ np.asarray(T, dtype=float)[:3, :3].T + np.asarray(T, dtype=float)[:3, 3]
    idx = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
           (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    return [c[list(f)] for f in idx]


def machine_view(m, layout, res, q=None, path=None, mask_kind="persistent",
                 severity_min=None, max_points=14000, elev=22.0, azim=-62.0):
    """The machine at one configuration with its blind volume drawn around it.

    Drawn at the worst configuration in the sweep rather than a tidy one,
    because a picture of the good case is not the point.

    Transient volume is filtered by severity before drawing.  For a layout like
    D it covers two thirds of the envelope, and every cell of it plotted at once
    is an even grey fog that shows nothing; the cells blind in most
    configurations are a far smaller set with real shape to it, and that shape
    is the boom shadow.  Persistent cells are all severity one by definition, so
    no filter applies to them.
    """
    path = path or FIGURES / f"machine_{mask_kind}.png"
    if q is None:
        q = res.poses[int(np.argmin(res.per_pose_cov))]

    env = res.envelope
    persistent, transient, severity, ever_free = classify(res)
    if mask_kind == "persistent":
        sel = persistent
        note = "never covered in any configuration"
    else:
        thr = 0.5 if severity_min is None else severity_min
        sel = transient & (severity >= thr)
        note = f"blind in at least {100 * thr:.0f}% of configurations"

    pts = env.centers_W[sel]
    weight = severity[sel]
    rng = np.random.default_rng(0)
    if len(pts) > max_points:
        take = rng.choice(len(pts), max_points, replace=False)
        pts, weight = pts[take], weight[take]

    link_T = K.link_poses(m, q)
    R_MW = rot_z(-float(q[0]))
    link_T = {k: R_MW @ v for k, v in link_T.items()}

    fig = plt.figure(figsize=(7.6, 4.4))
    ax = fig.add_subplot(111, projection="3d")
    # A 3D axes reserves a bounding box far larger than the cube it draws, so
    # bbox_inches="tight" cannot recover the margin.  Placed by hand instead.
    ax.set_position([-0.04, -0.10, 1.06, 1.02])
    cmap = plt.get_cmap("Reds" if mask_kind == "persistent" else "Blues")
    if len(pts):
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5.5,
                   c=cmap(0.45 + 0.5 * weight), alpha=0.33, linewidths=0,
                   depthshade=False)

    for s_ in m.solids:
        T = np.asarray(link_T[s_.link], dtype=float) @ s_.T_link
        ax.add_collection3d(Poly3DCollection(
            _box_faces(s_.size, T), facecolor="#c8a233", edgecolor="0.2",
            linewidths=0.5, alpha=1.0, zorder=20))

    for mo, T_WS in layout.poses_W(link_T):
        p_ = T_WS[:3, 3]
        ax.scatter(*p_, s=95, color="#0b3d6b", marker="^", depthshade=False,
                   edgecolor="white", linewidths=0.7, zorder=40)

    lim = env.radius
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0.0, env.z_range[1])
    # The envelope is 30 m across and 6 m tall; drawn to scale it is a pancake,
    # so the vertical is stretched to roughly three times true scale to make the
    # layering legible.  Stated here because an unlabelled exaggeration is a lie.
    ax.set_box_aspect((1.0, 1.0, 0.62))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x (m)", fontsize=8, labelpad=-2)
    ax.set_ylabel("y (m)", fontsize=8, labelpad=-2)
    ax.set_zlabel("z (m)", fontsize=8, labelpad=-4)
    ax.tick_params(labelsize=7, pad=-1)
    ax.set_zticks([0, 2, 4, 6])
    ax.grid(False)
    deg = np.degrees(q).round(0)
    ax.set_title(
        f"{LABEL.get(res.layout, res.layout)}: {mask_kind} blind volume, "
        f"{note}\nworst configuration: swing {deg[0]:.0f}, boom {deg[1]:.0f}, "
        f"stick {deg[2]:.0f}, bucket {deg[3]:.0f} deg.  "
        f"Triangles are sensors; vertical scale exaggerated.",
        fontsize=8.5, loc="left")
    return _save(fig, path)


def sampling_limits(path=None, ranges=None):
    """What the beam model buys, drawn as the smallest target each sensor resolves.

    The brief's threshold, three returns off a 1.7 m person from 5 m out, is
    never the binding constraint inside this envelope: a person that far away is
    struck by tens of beams and the question is only whether anything is in the
    way.  This figure is where that claim is made visible, and where the
    constraint that does exist shows up, on short targets.
    """
    from .detect import DetectionConfig, sampling_limit
    from .sensors import SensorSpec

    path = path or FIGURES / "sampling_limits.png"
    cfg = DetectionConfig()
    ranges = np.linspace(2.0, 25.0, 200) if ranges is None else np.asarray(ranges)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    colours = {"spinning32": "#c4622d", "spinning64": "#3b6ea5", "solid_state": "#4f8a5b"}
    for name, colour in colours.items():
        spec = SensorSpec.load(name)
        lim = sampling_limit(spec, cfg, ranges)
        ax.plot(ranges, lim["guaranteed_m"], color=colour, lw=1.8, label=name)
        ax.plot(ranges, lim["expected_m"], color=colour, lw=1.0, ls=":")

    ax.axhline(cfg.height_m, color="black", lw=1.1, ls="--")
    ax.annotate(f"a standing person, {cfg.height_m:.1f} m", (2.4, cfg.height_m),
                textcoords="offset points", xytext=(0, 5), fontsize=8)
    ax.axvspan(cfg.min_range_m, cfg.max_range_m, color="0.88", zorder=0)
    ax.annotate("the band this study scores",
                (0.5 * (cfg.min_range_m + cfg.max_range_m), 0.04),
                xycoords=("data", "axes fraction"), ha="center", fontsize=8,
                color="0.35")

    ax.set_xlabel("range (m)")
    ax.set_ylabel("smallest target height resolved (m)")
    ax.set_ylim(0, 2.0)
    ax.set_xlim(ranges.min(), ranges.max())
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Where beam sampling binds, and where it does not\n"
                 "solid: a target spanning one whole beam row, so a hit does not "
                 "depend on alignment\ndotted: where the expected return count "
                 "first reaches three",
                 fontsize=9, loc="left")
    return _save(fig, path)
