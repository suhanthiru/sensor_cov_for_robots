"""The work envelope, cut into voxels, and the bookkeeping that goes with it.

A cylinder rather than a box: the machine slews, so the region it can work in is
round, and cutting a square envelope would spend a third of the cells on corners
at 21 m that no sensor is being asked about.

The vertical extent runs from grade to 6 m, which covers the machine's own upper
structure and the height the boom sweeps through.  Coverage is also reported
separately over the 0 to 2 m band, because that is where people and vehicles are
and a single envelope-wide percentage hides it: a layout can look excellent
overall while being blind exactly where a person stands.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Envelope:
    """Voxel centres over the work envelope, with the grid indices kept.

    The indices are retained so a per-voxel result can be folded back into a
    dense array for a heatmap without re-deriving which cell went where.
    """

    centers_W: np.ndarray    # (N, 3) float32, cell centres in world
    idx: np.ndarray          # (N, 3) int32, grid indices of each cell
    shape: tuple             # (nx, ny, nz) of the enclosing grid
    origin: np.ndarray       # (3,) world position of grid index (0, 0, 0) centre
    cell: float              # m, edge length
    radius: float            # m, cylinder radius
    z_range: tuple           # m, (lo, hi)

    @property
    def n(self) -> int:
        return int(len(self.centers_W))

    @property
    def cell_volume(self) -> float:
        return float(self.cell ** 3)

    def volume_of(self, mask) -> float:
        """Volume in cubic metres of the cells a boolean mask selects."""
        return float(np.count_nonzero(mask)) * self.cell_volume

    def band(self, lo: float, hi: float) -> np.ndarray:
        """Cells whose centre height falls in ``[lo, hi)``."""
        z = self.centers_W[:, 2]
        return (z >= lo) & (z < hi)

    def ring(self, lo: float, hi: float) -> np.ndarray:
        """Cells whose horizontal distance from the slew axis is in ``[lo, hi)``."""
        r = np.hypot(self.centers_W[:, 0], self.centers_W[:, 1])
        return (r >= lo) & (r < hi)

    def to_grid(self, values, fill=np.nan) -> np.ndarray:
        """Scatter a per-cell vector back into the dense ``(nx, ny, nz)`` array."""
        out = np.full(self.shape, fill, dtype=float)
        out[self.idx[:, 0], self.idx[:, 1], self.idx[:, 2]] = values
        return out

    def column_max(self, values, fill=np.nan) -> np.ndarray:
        """Collapse a per-cell vector to a top-down ``(nx, ny)`` map by column maximum."""
        grid = self.to_grid(values, fill=fill)
        with np.errstate(invalid="ignore"):
            return np.nanmax(grid, axis=2)

    @property
    def extent(self):
        """Matplotlib-ready ``[x0, x1, y0, y1]`` for a top-down image."""
        nx, ny, _ = self.shape
        half = 0.5 * self.cell
        return [float(self.origin[0] - half), float(self.origin[0] + nx * self.cell - half),
                float(self.origin[1] - half), float(self.origin[1] + ny * self.cell - half)]


def build_envelope(radius: float = 15.0, z_lo: float = 0.0, z_hi: float = 6.0,
                   cell: float = 0.25) -> Envelope:
    """Cut the work envelope into cells.

    Cells are centred on the half-cell offsets so that no cell centre lands
    exactly on grade, where a ray grazing the ground plane would be a coin flip
    between hitting it and missing it.
    """
    n_side = int(np.ceil(2.0 * radius / cell))
    nz = int(np.ceil((z_hi - z_lo) / cell))
    ax = -radius + cell * (np.arange(n_side) + 0.5)
    az = z_lo + cell * (np.arange(nz) + 0.5)

    ix, iy, iz = np.meshgrid(np.arange(n_side), np.arange(n_side), np.arange(nz),
                             indexing="ij")
    X, Y, Z = np.meshgrid(ax, ax, az, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    idx = np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=1)

    inside = np.hypot(pts[:, 0], pts[:, 1]) <= radius
    return Envelope(centers_W=np.ascontiguousarray(pts[inside], dtype=np.float32),
                    idx=idx[inside].astype(np.int32),
                    shape=(n_side, n_side, nz),
                    origin=np.array([ax[0], ax[0], az[0]]),
                    cell=float(cell), radius=float(radius),
                    z_range=(float(z_lo), float(z_hi)))
