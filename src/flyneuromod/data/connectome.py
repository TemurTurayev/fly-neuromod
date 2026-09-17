"""Loading the FlyWire connectome into simulation-ready arrays."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

logger = logging.getLogger(__name__)

REQUIRED_CONNECTIVITY_COLUMNS = (
    "Presynaptic_ID",
    "Postsynaptic_ID",
    "Presynaptic_Index",
    "Postsynaptic_Index",
    "Excitatory x Connectivity",
)


@dataclass(frozen=True, slots=True)
class Connectome:
    """Neurons and signed connection counts of one connectome release.

    Attributes
    ----------
    root_ids:
        FlyWire root id of every neuron, ordered by simulation index.
    weights:
        Sparse ``[post, pre]`` matrix of signed synapse counts. The sign encodes
        the predicted transmitter (excitatory ``+``, inhibitory ``-``) as in the
        reference model; the magnitude is the number of anatomical synapses.
    release:
        Name of the connectome release, e.g. ``"flywire_783"``.
    """

    root_ids: np.ndarray
    weights: sparse.csr_matrix
    release: str

    def __post_init__(self) -> None:
        n = len(self.root_ids)
        if self.weights.shape != (n, n):
            raise ValueError(
                f"weights shape {self.weights.shape} does not match {n} neurons"
            )

    @property
    def n_neurons(self) -> int:
        return len(self.root_ids)

    @property
    def n_connections(self) -> int:
        """Number of connected neuron pairs (not the number of synapses)."""
        return int(self.weights.nnz)

    def index_map(self) -> dict[int, int]:
        """Mapping from FlyWire root id to simulation index."""
        return {int(root_id): i for i, root_id in enumerate(self.root_ids)}

    def indices_of(self, root_ids: np.ndarray | list[int]) -> np.ndarray:
        """Simulation indices of the given root ids.

        Raises
        ------
        KeyError
            If a root id is not part of this connectome.
        """
        lookup = self.index_map()
        missing = [int(r) for r in root_ids if int(r) not in lookup]
        if missing:
            raise KeyError(
                f"{len(missing)} root ids are not in {self.release} "
                f"(first few: {missing[:5]})"
            )
        return np.array([lookup[int(r)] for r in root_ids], dtype=np.int64)

    def subnetwork(self, root_ids: np.ndarray | list[int]) -> Connectome:
        """Return the induced subnetwork on ``root_ids``, preserving their order."""
        indices = self.indices_of(root_ids)
        if len(set(indices.tolist())) != len(indices):
            raise ValueError("duplicate neurons requested for the subnetwork")
        sub = self.weights[indices][:, indices]
        return Connectome(
            root_ids=np.asarray(root_ids, dtype=np.int64),
            weights=sparse.csr_matrix(sub),
            release=f"{self.release}:sub{len(indices)}",
        )


def load_connectome(
    completeness_csv: str | Path,
    connectivity_parquet: str | Path,
    release: str = "flywire_783",
) -> Connectome:
    """Build a :class:`Connectome` from the two reference-model tables.

    Parameters
    ----------
    completeness_csv:
        CSV whose index lists every modelled neuron by FlyWire root id; the row
        order defines the simulation indices used by the connectivity table.
    connectivity_parquet:
        Parquet table with one row per connected neuron pair.
    release:
        Label stored with the connectome.

    Raises
    ------
    FileNotFoundError
        If either file is missing.
    ValueError
        If the connectivity table lacks required columns or its indices do not
        match the completeness table.
    """
    completeness_csv, connectivity_parquet = Path(completeness_csv), Path(connectivity_parquet)
    for path in (completeness_csv, connectivity_parquet):
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found; run `flyneuromod download` to fetch the connectome"
            )

    neurons = pd.read_csv(completeness_csv, index_col=0)
    root_ids = neurons.index.to_numpy(dtype=np.int64)

    connections = pd.read_parquet(connectivity_parquet)
    missing = [c for c in REQUIRED_CONNECTIVITY_COLUMNS if c not in connections.columns]
    if missing:
        raise ValueError(f"connectivity table is missing columns: {missing}")

    pre = connections["Presynaptic_Index"].to_numpy(dtype=np.int64)
    post = connections["Postsynaptic_Index"].to_numpy(dtype=np.int64)
    signed_counts = connections["Excitatory x Connectivity"].to_numpy(dtype=np.float64)

    n = len(root_ids)
    if pre.size and (pre.max() >= n or post.max() >= n or pre.min() < 0 or post.min() < 0):
        raise ValueError(
            "connectivity indices fall outside the completeness table; the two files "
            "are probably from different connectome releases"
        )

    weights = sparse.csr_matrix((signed_counts, (post, pre)), shape=(n, n))
    weights.sum_duplicates()
    logger.info(
        "loaded %s: %d neurons, %d connections", release, n, weights.nnz
    )
    return Connectome(root_ids=root_ids, weights=weights, release=release)
