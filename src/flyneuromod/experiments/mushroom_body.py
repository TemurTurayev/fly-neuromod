"""Building the mushroom body subnetwork.

Whole-brain runs are useful but slow, and the dopamine layer acts on a circuit
that is anatomically well delimited: Kenyon cells, their output neurons, the
dopaminergic neurons that teach them, and the two large modulatory neurons (APL
and DPM) that keep Kenyon cell activity sparse. Extracting that subnetwork keeps
every connection between those cells and drops the rest of the brain, which
makes second-scale conditioning protocols practical on a laptop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..data.annotations import Annotations
from ..data.connectome import Connectome

logger = logging.getLogger(__name__)

MODULATORY_TYPES = ("APL", "DPM")


@dataclass(frozen=True, slots=True)
class MushroomBody:
    """The mushroom body subnetwork and the annotations restricted to it."""

    connectome: Connectome
    annotations: Annotations

    @property
    def n_neurons(self) -> int:
        return self.connectome.n_neurons

    def indices_of_type(self, *cell_types: str) -> np.ndarray:
        """Network indices of the given cell types inside the subnetwork."""
        root_ids = self.annotations.by_cell_type(*cell_types)
        return self.connectome.indices_of(root_ids)

    def kenyon_indices(self) -> np.ndarray:
        return self.connectome.indices_of(self.annotations.kenyon_cells())

    def mbon_indices(self, cell_type: str | None = None) -> np.ndarray:
        root_ids = (
            self.annotations.mbons() if cell_type is None else self.annotations.by_cell_type(cell_type)
        )
        return self.connectome.indices_of(root_ids)


def extract_mushroom_body(connectome: Connectome, annotations: Annotations) -> MushroomBody:
    """Select Kenyon cells, output neurons, dopaminergic neurons, APL and DPM.

    Parameters
    ----------
    connectome:
        Whole-brain connectome.
    annotations:
        Whole-brain cell-type annotations.

    Returns
    -------
    MushroomBody
        Subnetwork with all connections among the selected cells preserved.
    """
    available = set(connectome.root_ids.tolist())
    selected: list[int] = []
    for group in (
        annotations.kenyon_cells(),
        annotations.mbons(),
        annotations.dopaminergic(),
        annotations.by_cell_type(*MODULATORY_TYPES),
    ):
        selected.extend(int(r) for r in group if int(r) in available)

    unique = np.array(sorted(set(selected)), dtype=np.int64)
    sub = connectome.subnetwork(unique)
    logger.info(
        "mushroom body subnetwork: %d neurons, %d connections", sub.n_neurons, sub.n_connections
    )
    return MushroomBody(connectome=sub, annotations=annotations.restricted_to(unique))


def sparse_odour(
    kenyon_indices: np.ndarray,
    fraction: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Pick the Kenyon cells that represent one odour.

    Odours are encoded by a small, largely non-overlapping set of Kenyon cells;
    around 5-10% of them respond to a given odour. Until the antennal lobe input
    pathway is modelled, an odour here *is* such a random subset, which is what
    the electrophysiological pairing experiments effectively control for.

    Parameters
    ----------
    kenyon_indices:
        Indices of all Kenyon cells.
    fraction:
        Fraction of Kenyon cells the odour activates.
    rng:
        Random generator, for reproducible odours.
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    rng = rng or np.random.default_rng()
    n = max(1, int(round(fraction * len(kenyon_indices))))
    return rng.choice(kenyon_indices, size=n, replace=False)
