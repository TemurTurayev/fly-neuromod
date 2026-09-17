"""Correcting predicted transmitters where the literature knows better.

The sign of every connection in the reference model comes from a classifier run
on electron-microscopy images. It is right for most of the brain, and wrong for
a small number of cell types that matter for the circuit this library models.
Those corrections live in ``transmitter_overrides.yaml``, each with its source.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
import yaml
from scipy import sparse

from .annotations import Annotations
from .connectome import Connectome

logger = logging.getLogger(__name__)

SELECTORS = ("cell_class", "cell_type")


@dataclass(frozen=True, slots=True)
class TransmitterOverride:
    """One curated correction.

    Attributes
    ----------
    select:
        Annotation column to match on, ``cell_class`` or ``cell_type``.
    value:
        Value to match.
    transmitter:
        The transmitter the cells actually release, for documentation.
    sign:
        ``+1`` to make outgoing connections excitatory, ``-1`` inhibitory.
    source:
        Citation.
    """

    select: str
    value: str
    transmitter: str
    sign: int
    source: str

    def __post_init__(self) -> None:
        if self.select not in SELECTORS:
            raise ValueError(f"select must be one of {SELECTORS}, got {self.select!r}")
        if self.sign not in (-1, 1):
            raise ValueError("sign must be +1 or -1")
        if not self.source:
            raise ValueError(f"{self.value}: an override needs a source")

    def root_ids(self, annotations: Annotations) -> np.ndarray:
        """Root ids of the cells this override applies to."""
        if self.select == "cell_class":
            return annotations.by_cell_class(self.value)
        return annotations.by_cell_type(self.value)


def load_overrides(path: str | Path | None = None) -> tuple[TransmitterOverride, ...]:
    """Load the curated corrections, by default the packaged ones."""
    if path is None:
        text = (
            resources.files("flyneuromod.data")
            .joinpath("transmitter_overrides.yaml")
            .read_text()
        )
    else:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"override file {path} not found")
        text = path.read_text()

    raw = yaml.safe_load(text)
    if not isinstance(raw, dict) or "overrides" not in raw:
        raise ValueError("override file must be a mapping with an 'overrides' key")
    return tuple(
        TransmitterOverride(
            select=entry["select"],
            value=entry["value"],
            transmitter=entry["transmitter"],
            sign=int(entry["sign"]),
            source=entry.get("source", ""),
        )
        for entry in raw["overrides"]
    )


def apply_transmitter_overrides(
    connectome: Connectome,
    annotations: Annotations,
    overrides: tuple[TransmitterOverride, ...] | None = None,
) -> Connectome:
    """Return a connectome whose connection signs follow the curated identities.

    Only the sign changes: the number of synapses, which carries the anatomy, is
    untouched. The input connectome is not modified.
    """
    overrides = overrides if overrides is not None else load_overrides()
    weights = sparse.csc_matrix(connectome.weights).copy()  # column = presynaptic neuron
    index_of = connectome.index_map()

    for override in overrides:
        indices = [index_of[int(r)] for r in override.root_ids(annotations) if int(r) in index_of]
        if not indices:
            continue
        changed = 0
        for column in indices:
            start, stop = weights.indptr[column], weights.indptr[column + 1]
            block = weights.data[start:stop]
            changed += int(np.count_nonzero(np.sign(block) != override.sign))
            block[:] = np.abs(block) * override.sign
        logger.info(
            "%s=%s: %d cells set to %s (%d connections changed sign)",
            override.select,
            override.value,
            len(indices),
            override.transmitter,
            changed,
        )

    return Connectome(
        root_ids=connectome.root_ids,
        weights=sparse.csr_matrix(weights),
        release=f"{connectome.release}+curated_nt",
    )
