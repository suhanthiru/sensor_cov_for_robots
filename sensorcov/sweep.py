"""The articulation sweep: walk the joint grid and accumulate what each cell sees.

This is the part of the study that a static analysis cannot do.  A sensor layout
evaluated at one configuration tells you where the blind spots are for that
configuration; it says nothing about whether they stay put.  On a machine whose
largest occluder is also its working member, they do not.

Storing visibility per pose per cell is not an option: thirty four thousand
poses against two hundred and seventy thousand cells is nine billion bits.  So
nothing is stored per pose.  Four counters per cell are enough to recover
everything the study asks for:

``free_count``   poses in which the cell was not inside the machine
``seen_count``   poses in which at least one sensor saw it
``dual_count``   poses in which at least two did
``ever_seen``    whether any pose ever saw it

From those, a cell that was free at some pose and never seen is **persistent**
blind volume, which is a placement flaw.  A cell that was seen at some poses and
not others is **transient**, and the fraction of poses it was missed in is a
severity rather than a flag.  The transient class is the interesting one: it is
invisible to a static study and it is the one that hurts, because a region that
is covered while the boom is up and blind while it is down is a region an
operator has been taught to trust.

All of this is measured in the machine frame, not the world: with the swing
angle left in, a turret-mounted sensor sweeps past every fixed point as the
machine slews and practically the whole envelope comes back transient, which
measures only that excavators slew.  Swing still matters in the sweep, because
the undercarriage does not turn with the house, but it no longer drowns the
articulation signal the study is about.

The denominator is free cells, not all cells.  The boom sweeps bodily through
the envelope, and counting the cells it currently fills as unseen would report
an enormous transient blind volume that is really just the boom being where the
boom is.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import kinematics as K
from .detect import DetectionConfig, detected
from .envelope import Envelope, build_envelope
from .frames import rot_z
from .machine import MachineSpec
from .raycast import MachineScene, occupied_mask
from .sensors import Layout

PERSON_BAND = (0.0, 2.0)     # m, where a person or a groundworker actually is


@dataclass
class SweepResult:
    """Everything the metrics are derived from, and nothing per pose per cell."""

    layout: str
    title: str
    n_sensors: int
    moves_with_the_boom: bool
    poses: np.ndarray            # (P, 4) joint angles in radians
    grid_counts: dict            # how the joint grid was filtered, and why
    free_count: np.ndarray       # (N,) uint32
    seen_count: np.ndarray       # (N,) uint32
    dual_count: np.ndarray       # (N,) uint32
    ever_seen: np.ndarray        # (N,) bool
    detect_count: np.ndarray     # (T,) uint32, poses each standing position was seen in
    per_pose_cov: np.ndarray     # (P,) fraction of free cells seen
    per_pose_person: np.ndarray  # (P,) the same over the 0 to 2 m band
    per_pose_dual: np.ndarray    # (P,) fraction seen by two or more
    per_pose_detect: np.ndarray  # (P,) fraction of standing positions detected
    env_kwargs: dict = field(default_factory=dict)
    det_cfg: dict = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def envelope(self) -> Envelope:
        return build_envelope(**self.env_kwargs)

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{k: v for k, v in asdict(self).items()
                                     if isinstance(v, np.ndarray)},
                            meta=np.array(repr({
                                "layout": self.layout, "title": self.title,
                                "n_sensors": self.n_sensors,
                                "moves_with_the_boom": self.moves_with_the_boom,
                                "grid_counts": self.grid_counts,
                                "env_kwargs": self.env_kwargs,
                                "det_cfg": self.det_cfg,
                                "seconds": self.seconds}), dtype=object))

    @classmethod
    def load(cls, path) -> SweepResult:
        import ast

        z = np.load(path, allow_pickle=True)
        meta = ast.literal_eval(str(z["meta"].item()))
        arrays = {k: z[k] for k in z.files if k != "meta"}
        return cls(**meta, **arrays)


# ---------------------------------------------------------------------------
# one pose
# ---------------------------------------------------------------------------


def score_pose(m: MachineSpec, layout: Layout, env: Envelope, q,
               cfg: DetectionConfig = None, targets=None):
    """Everything one configuration contributes, as plain arrays.

    Returned rather than accumulated so that a single pose can be scored and
    inspected on its own, which is what the figures and most of the tests need.
    """
    # Everything is scored in the machine frame M: the world with the swing
    # angle taken out, so +x is whichever way the turret currently faces.  See
    # frames.py for why.  Taking the yaw out is one 4x4 product per link rather
    # than a transform of a quarter of a million cell centres, and the ground
    # plane is unchanged by it because it is a horizontal plane about the same
    # axis.  The undercarriage is the one thing that does turn in M, which is
    # exactly right: the tracks do not swing with the house.
    link_T = K.link_poses(m, q)
    R_MW = rot_z(-float(q[0]))
    link_T = {k: R_MW @ v for k, v in link_T.items()}

    free = ~occupied_mask(m, link_T, env.centers_W)
    scene = MachineScene(m, link_T)

    n_seen = np.zeros(env.n, dtype=np.uint8)
    poses_W = layout.poses_W(link_T)
    for mo, T_WS in poses_W:
        # A bool array and a uint8 array have the same layout, so the view is a
        # reinterpretation rather than a conversion.
        n_seen += scene.line_of_sight(T_WS, mo.sensor, env.centers_W).view(np.uint8)

    seen = free & (n_seen >= 1)
    dual = free & (n_seen >= 2)
    det = (detected(scene, poses_W, cfg, targets)
           if cfg is not None else np.zeros(0, dtype=bool))
    return free, seen, dual, det


# ---------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------

_STATE = {}


def _init_worker(machine_path, layout_name, env_kwargs, det_kwargs):
    _STATE["m"] = MachineSpec.load(machine_path)
    _STATE["layout"] = Layout.load(layout_name)
    _STATE["env"] = build_envelope(**env_kwargs)
    _STATE["cfg"] = DetectionConfig(**det_kwargs)
    _STATE["targets"] = _STATE["cfg"].targets_W()


def _run_chunk(args):
    lo, poses = args
    m, layout, env = _STATE["m"], _STATE["layout"], _STATE["env"]
    cfg, targets = _STATE["cfg"], _STATE["targets"]

    free_count = np.zeros(env.n, dtype=np.uint32)
    seen_count = np.zeros(env.n, dtype=np.uint32)
    dual_count = np.zeros(env.n, dtype=np.uint32)
    ever = np.zeros(env.n, dtype=bool)
    detect_count = np.zeros(len(targets), dtype=np.uint32)

    band = env.band(*PERSON_BAND)
    n = len(poses)
    cov = np.zeros(n, dtype=np.float32)
    person = np.zeros(n, dtype=np.float32)
    dualf = np.zeros(n, dtype=np.float32)
    detf = np.zeros(n, dtype=np.float32)

    for i, q in enumerate(poses):
        free, seen, dual, det = score_pose(m, layout, env, q, cfg, targets)
        free_count += free
        seen_count += seen
        dual_count += dual
        ever |= seen
        detect_count += det

        n_free = int(free.sum())
        cov[i] = seen.sum() / max(n_free, 1)
        dualf[i] = dual.sum() / max(n_free, 1)
        n_band = int((free & band).sum())
        person[i] = (seen & band).sum() / max(n_band, 1)
        detf[i] = det.mean() if det.size else 0.0

    return lo, free_count, seen_count, dual_count, ever, detect_count, cov, person, dualf, detf


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------


def run_sweep(layout_name: str, machine_path=None, quick: bool = False,
              workers: int = 1, cell: float = 0.25, radius: float = 15.0,
              z_hi: float = 6.0, det_cfg: DetectionConfig = None,
              chunk: int = 96, progress: bool = True) -> SweepResult:
    """Walk the joint grid for one layout and accumulate the counters."""
    m = MachineSpec.load(machine_path)
    layout = Layout.load(layout_name)
    det_cfg = det_cfg or DetectionConfig()
    env_kwargs = {"radius": radius, "z_lo": 0.0, "z_hi": z_hi,
                  "cell": 1.0 if quick else cell}
    if quick:
        env_kwargs["cell"] = 0.5
    env = build_envelope(**env_kwargs)

    poses, counts = K.valid_poses(m, K.joint_grid(m, quick=quick))
    if progress:
        print(f"{layout_name}: {len(poses):,} poses of {counts['total']:,} "
              f"({counts['below_grade']:,} below grade, "
              f"{counts['self_collision']:,} interference), "
              f"{env.n:,} cells, {layout.n_sensors} sensors")

    free_count = np.zeros(env.n, dtype=np.uint32)
    seen_count = np.zeros(env.n, dtype=np.uint32)
    dual_count = np.zeros(env.n, dtype=np.uint32)
    ever = np.zeros(env.n, dtype=bool)
    detect_count = np.zeros(len(det_cfg.targets_W()), dtype=np.uint32)
    n = len(poses)
    cov = np.zeros(n, dtype=np.float32)
    person = np.zeros(n, dtype=np.float32)
    dualf = np.zeros(n, dtype=np.float32)
    detf = np.zeros(n, dtype=np.float32)

    jobs = [(lo, poses[lo:lo + chunk]) for lo in range(0, n, chunk)]
    init_args = (machine_path, layout_name, env_kwargs, asdict(det_cfg))
    t0 = time.perf_counter()
    done = 0

    def absorb(out):
        nonlocal done
        lo, fc, sc, dc, ev, dt, c, p, d, f = out
        free_count[:] += fc
        seen_count[:] += sc
        dual_count[:] += dc
        ever[:] |= ev
        detect_count[:] += dt
        hi = lo + len(c)
        cov[lo:hi], person[lo:hi], dualf[lo:hi], detf[lo:hi] = c, p, d, f
        done += len(c)
        if progress:
            el = time.perf_counter() - t0
            rate = done / max(el, 1e-9)
            print(f"\r  {done:,}/{n:,} poses  {el:6.1f}s  "
                  f"{rate:5.1f} pose/s  eta {(n - done) / max(rate, 1e-9):6.1f}s",
                  end="", flush=True)

    if workers <= 1:
        _init_worker(*init_args)
        for job in jobs:
            absorb(_run_chunk(job))
    else:
        # Torn down explicitly rather than with a `with` block.  Pool.__exit__
        # calls terminate(), which asks workers to die while they may be inside
        # a long Embree call and does not wait for them; run several layouts in
        # one process and the leaked pools accumulate.  Twelve workers from an
        # earlier layout stayed live for over an hour here and halved the
        # throughput of the one still running, which looks exactly like the
        # sweep being slow rather than like a leak.  close() lets each worker
        # finish and exit, join() waits until they actually have, and terminate()
        # is kept for the error path where waiting is not appropriate.
        pool = Pool(workers, initializer=_init_worker, initargs=init_args)
        try:
            for out in pool.imap_unordered(_run_chunk, jobs):
                absorb(out)
            pool.close()
        except BaseException:
            pool.terminate()
            raise
        finally:
            pool.join()
    if progress:
        print()

    return SweepResult(
        layout=layout_name, title=layout.title, n_sensors=layout.n_sensors,
        moves_with_the_boom=layout.moves_with_the_boom,
        poses=poses, grid_counts=counts,
        free_count=free_count, seen_count=seen_count, dual_count=dual_count,
        ever_seen=ever, detect_count=detect_count,
        per_pose_cov=cov, per_pose_person=person, per_pose_dual=dualf,
        per_pose_detect=detf,
        env_kwargs=env_kwargs, det_cfg=asdict(det_cfg),
        seconds=time.perf_counter() - t0)
