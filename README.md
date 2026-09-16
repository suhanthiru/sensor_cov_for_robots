# Sensor coverage for an articulated excavator

Where should the perception sensors go on a 20 tonne hydraulic excavator, and
what does each arrangement actually buy?

The usual way to answer this is to park the machine, pick a representative pose,
cast rays, and report a coverage percentage. That analysis is misleading on this
class of machine, and not by a little. The largest occluder on an excavator is
also its working member: the boom, stick and bucket sweep bodily through the
region the sensors are supposed to watch. A blind spot found that way is a blind
spot at one configuration. It says nothing about whether the region is still
covered when the boom comes down.

So this study sweeps the joint space instead, and splits the blind volume in two:

- **Persistent** blind volume is never covered in any configuration. It is a
  placement flaw, and the fix is to move or add a sensor.
- **Transient** blind volume is covered in some configurations and not others.

The transient class is the interesting one, and it is more dangerous than the
persistent one despite being smaller, for a reason that has nothing to do with
size. A persistent blind spot is discoverable. Anyone who runs a static coverage
analysis, or walks around the machine once with a test target, will find it, and
it can be trained and designed around. A transient one is covered when you look
and blind when the boom comes down. It is invisible to exactly the analysis most
people run, and it teaches operators to trust a region that is not always there.

<!-- RESULTS -->

## Running it

```bash
pip install -e ".[raycast,dev]"
```

```bash
python -m pytest -q
```

A single layout, at the coarse preset, to check the pipeline end to end in about
a minute:

```bash
python -m sensorcov.run_sweep --layout d_cab_and_boom --quick
```

The full study, all four layouts. About half an hour per layout across twelve
worker processes; sweeps are cached as `results/*.npz` and reused:

```bash
python -m sensorcov.study --workers 12
```

```bash
python -m sensorcov.run_figures
```

## Method

### The machine

A Cat 320 GC, built from primitives rather than a production URDF. Occlusion is
driven by gross volume rather than surface detail, and a dozen boxes build their
bounding hierarchy in under a millisecond, which is what makes a thirty thousand
pose sweep finish at all.

Link lengths, cab height, tail swing radius, track gauge and counterweight
clearance are read off the manufacturer's technical specification, and
`configs/machine/cat320.yaml` keeps measured and inferred numbers in separate
blocks so the distinction survives into the results. The solids are derived from
those numbers rather than typed in beside them, so correcting a dimension moves
the geometry with it.

The one shape detail that is not optional is the gooseneck. A real excavator
boom bends, and since boom shadow is the entire subject of the study, modelling
it as one straight box would put the shadow in the wrong place.

The boom foot position and the joint limits are not published, so they are
fitted against the three published working ranges instead. The chain reproduces
a 9.86 m maximum reach at ground line, a 6.72 m digging depth and a 9.45 m
cutting height to within 7 cm, and a test asserts it. That check is the main
reason to believe the kinematics: it compares against the manufacturer rather
than against ourselves, and a sign error anywhere in the chain breaks it.

### Two ray models, kept apart

This is the design decision the rest of the study rests on.

**Line of sight** asks whether a sensor could see into a region at all: in
range, inside the field of view, unoccluded. It ignores angular sampling, which
is the right abstraction for a coverage percentage.

**Beam accurate** generates the discrete rays the device really emits. A 32
channel lidar spread over 45 degrees of elevation leaves about 1.4 degrees
between channels, which at 15 m is a 38 cm vertical gap, wider than the 25 cm
cells the envelope is cut into. Targets genuinely fall between beams.

Conflating the two would quietly destroy the safety metric. Scored against a
solid field-of-view cone, every target in range is struck by infinitely many
rays, so "hit by at least three beams" reads 100 percent everywhere and means
nothing. Cameras are not pretended into the beam model at all; they are scored
on pixels on target, which is what actually limits a camera on a small obstacle
at range.

### The frame the numbers are in

Coverage is measured in a machine frame: the world with the swing angle taken
out, so `+x` is whichever way the upper structure is currently facing.

This is not a presentational choice. Measured in the world instead, a
turret-mounted sensor sweeps past every fixed cell as the machine slews, so
almost every cell is seen in some configurations and missed in others. The first
version of this study did exactly that and reported 94 percent of the envelope
as transient with nothing at all always seen, for every layout, which is a true
statement about excavators slewing and tells you nothing about where to put a
sensor. In the machine frame the swing drops out for anything bolted to the
turret and what is left is the articulation effect the study is about.

Swing still earns its place in the sweep, because the undercarriage does not
turn with the house. That turns out to matter exactly once: the tracks never
occlude a cab-mounted sensor, which looks down over them from 3 m, and only bite
on layout D's boom unit at full boom-down, where it drops to 0.71 m against a
0.98 m track.

### Occupied is not blind

The boom sweeps through the work envelope, so the cells it currently fills are
solid, not unseen. Counting them as blind would report an enormous transient
volume that is really just the boom being where the boom is, and it would be the
headline number. Occupancy is recomputed per pose and excluded from the
denominator.

It is computed analytically from the primitives rather than with Open3D's
`compute_occupancy`, which counts ray crossings: the machine is built from
deliberately overlapping boxes, and a point inside two of them crosses an even
number of surfaces and comes back empty.

### The sweep

Swing every 15 degrees, boom, stick and bucket every 10, which is 71,136
configurations before filtering. Poses that drive the bucket below grade are
dropped, because the ground here is an unbroken flat plane with no trench cut
into it, and poses that fold a link into the cab are dropped as unreachable. The
rejection counts are reported so the denominator is auditable.

Storing visibility per pose per cell is not possible, so nothing is. Four
counters per cell are enough to recover everything: how many poses the cell was
free in, how many it was seen in, how many it was seen by two or more sensors
in, and whether it was ever seen at all.

## Layout

```
sensorcov/
  frames.py       frame conventions and SE(3) helpers; the single source of truth
  machine.py      the Cat 320 as primitive solids, derived from the spec numbers
  kinematics.py   the four joint chain, working ranges, and the pose grid
  sensors.py      sensor families, beam tables, field of view, and mounts
  envelope.py     the voxelised work envelope and its bookkeeping
  raycast.py      Open3D scene assembly, line of sight, and occupancy
  detect.py       beam-accurate detection of a standing person
  sweep.py        the articulation sweep and its per-cell counters
  metrics.py      persistent and transient split, volumes, Pareto front
  study.py        the four layouts and the trade study
  viz.py          figures
configs/
  machine/        the spec sheet, split by provenance
  sensors/        the four sensor families, parameterised
  layouts/        the four candidate arrangements, with their rationale
tests/sensorcov/  mirrors the package
```

## Caveats

What this does not show.

The geometry is primitive. Boxes and a two-segment boom stand in for real hulls,
which is the right abstraction for gross occlusion and the wrong one for
anything at the scale of a handrail or a mirror. Cab glazing is modelled as
solid, so a sensor inside the cab would be scored as blind; none of these
layouts put one there.

The ground is a flat, unbroken plane. There is no trench, no spoil pile, no
slope and no bench. An excavator spends its life next to all four, and each of
them both occludes and creates the hazard that matters. Poses that put the
bucket below grade are excluded for this reason rather than because the machine
cannot reach them.

There is no detector in the loop. A target is "detected" when the geometry says
enough beams come back off it, at any range within the band and at any incidence.
Real perception fails on reflectivity, on rain and dust, on motion blur, and on
targets that are geometrically visible and statistically indistinguishable from
the spoil behind them. Every number here is an upper bound.

Sensors are treated as ideal and always working. No occlusion by their own
mounts beyond the machine geometry, no fouling, no calibration error. That last
one matters unevenly across the layouts, and the trade study says so in words
rather than pretending a scalar carries it.

The aiming within each layout was chosen to be reasonable, not optimised. Layout
C in particular is sensitive to it: four 70 degree windows cannot cover 360, and
where the missing 80 degrees is put changes its coverage by tens of points. The
study prices four named arrangements; it does not search the space of mounts.

Finally, the transient and persistent split is relative to the pose grid. A
configuration the grid does not visit cannot contribute a blind spot, and a
blind spot that exists only between grid points is invisible. The grid is the
one the brief specifies; a finer one would only ever find more.
