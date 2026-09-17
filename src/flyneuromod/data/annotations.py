"""Cell-type annotations of the FlyWire connectome.

Source: Schlegel et al. (2024), *Nature* 634:139-152, distributed as
``Supplemental_file1_neuron_annotations.tsv`` in
https://github.com/flyconnectome/flywire_annotations.

The important distinction this module enforces is between

``top_nt``
    the *predicted* transmitter of a neuron, from a classifier applied to
    electron-microscopy images, and

``cell_class``
    a curated identity such as ``DAN`` (dopaminergic neuron of the mushroom
    body) or ``Kenyon_Cell``.

For dopamine the two disagree dramatically: in v783 the predictor labels 5,909
neurons as dopaminergic, 5,172 of which are Kenyon cells — cells that are in
fact cholinergic. Anything that selects dopamine sources by ``top_nt`` therefore
turns the whole mushroom body into a dopamine source. Use :meth:`Annotations.dopaminergic`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("root_id", "cell_class", "cell_type", "top_nt", "side")

DAN_CELL_CLASS = "DAN"
KENYON_CELL_CLASS = "Kenyon_Cell"


@dataclass(frozen=True, slots=True)
class Annotations:
    """Curated annotations indexed by FlyWire root id."""

    table: pd.DataFrame

    def __post_init__(self) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in self.table.columns]
        if missing:
            raise ValueError(f"annotation table is missing columns: {missing}")

    # ------------------------------------------------------------------
    # selection helpers - all return arrays of FlyWire root ids
    # ------------------------------------------------------------------
    def root_ids_where(self, mask: pd.Series) -> np.ndarray:
        return self.table.loc[mask, "root_id"].to_numpy(dtype=np.int64)

    def by_cell_class(self, cell_class: str) -> np.ndarray:
        """Root ids of all neurons in a curated cell class."""
        return self.root_ids_where(self.table["cell_class"] == cell_class)

    def by_cell_type(self, *cell_types: str) -> np.ndarray:
        """Root ids of neurons whose exact ``cell_type`` is one of ``cell_types``."""
        if not cell_types:
            return np.zeros(0, dtype=np.int64)
        return self.root_ids_where(self.table["cell_type"].isin(cell_types))

    def by_type_prefix(self, prefix: str) -> np.ndarray:
        """Root ids of neurons whose ``cell_type`` starts with ``prefix``."""
        return self.root_ids_where(
            self.table["cell_type"].astype("string").str.startswith(prefix, na=False)
        )

    def dopaminergic(self) -> np.ndarray:
        """Root ids of curated dopaminergic neurons (``cell_class == "DAN"``).

        This is the identity used as a dopamine *source*. It deliberately
        ignores ``top_nt`` (see the module docstring).
        """
        return self.by_cell_class(DAN_CELL_CLASS)

    def kenyon_cells(self) -> np.ndarray:
        return self.by_cell_class(KENYON_CELL_CLASS)

    def mbons(self) -> np.ndarray:
        return self.by_type_prefix("MBON")

    def predicted_transmitter(self, transmitter: str) -> np.ndarray:
        """Root ids by *predicted* transmitter — provided for comparison only."""
        return self.root_ids_where(self.table["top_nt"] == transmitter)

    def cell_types_of(self, root_ids: np.ndarray | list[int]) -> pd.Series:
        """Cell type of each requested root id (``NaN`` where unannotated)."""
        indexed = self.table.set_index("root_id")["cell_type"]
        return indexed.reindex(np.asarray(root_ids, dtype=np.int64))

    def types_in(self, root_ids: np.ndarray | list[int]) -> pd.Series:
        """Count of each cell type among ``root_ids``."""
        return self.cell_types_of(root_ids).value_counts()

    def restricted_to(self, root_ids: np.ndarray | list[int]) -> Annotations:
        """Annotations restricted to neurons present in ``root_ids``."""
        keep = self.table["root_id"].isin(np.asarray(root_ids, dtype=np.int64))
        return Annotations(table=self.table.loc[keep].reset_index(drop=True))


def load_annotations(path: str | Path) -> Annotations:
    """Read the FlyWire annotation TSV.

    Raises
    ------
    FileNotFoundError
        If the annotation file is missing.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run `flyneuromod download` to fetch the annotations"
        )
    table = pd.read_csv(path, sep="\t", low_memory=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"annotation file {path} is missing columns: {missing}")
    table = table.astype({"root_id": np.int64})
    logger.info("loaded annotations for %d neurons", len(table))
    return Annotations(table=table)
