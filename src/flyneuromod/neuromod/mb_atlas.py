"""The mushroom body compartment map.

A compartment is the anatomical unit of dopaminergic teaching: one set of
dopaminergic neurons, one slab of Kenyon cell axons, one or two output neurons.
Plasticity is confined to it. This module turns the curated table in
``data/mb_compartments.yaml`` into objects the dopamine layer can use, and
checks it against the connectome annotations rather than trusting it blindly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

POTENTIATION_LABELS = ("measured_yes", "measured_no", "unknown")
VALENCE_LABELS = ("approach", "avoidance", None)


@dataclass(frozen=True, slots=True)
class Compartment:
    """One mushroom body compartment.

    Attributes
    ----------
    name:
        Machine-readable identifier, e.g. ``"gamma1pedc"``.
    label:
        Name as written in the literature, e.g. ``"γ1pedc"``.
    dan_types:
        Cell types of the dopaminergic neurons that innervate the compartment.
    mbon_types:
        Cell types of the output neurons that read it.
    mbon_transmitter:
        Transmitter of those output neurons.
    valence:
        Behaviour evoked by activating the output neurons.
    potentiation:
        Whether backward pairing potentiates here, and on what evidence.
    note:
        Free-text provenance.
    """

    name: str
    label: str
    dan_types: tuple[str, ...]
    mbon_types: tuple[str, ...]
    mbon_transmitter: str | None
    valence: str | None
    potentiation: str
    note: str = ""

    def __post_init__(self) -> None:
        if not self.dan_types:
            raise ValueError(f"{self.name}: a compartment needs at least one DAN type")
        if self.potentiation not in POTENTIATION_LABELS:
            raise ValueError(
                f"{self.name}: potentiation must be one of {POTENTIATION_LABELS}"
            )
        if self.valence not in VALENCE_LABELS:
            raise ValueError(f"{self.name}: valence must be one of {VALENCE_LABELS}")

    @property
    def potentiates(self) -> bool:
        """Whether the model should potentiate on backward pairing here.

        ``unknown`` compartments potentiate: the sign flip is a property of the
        receptors, which every compartment has, and only γ1pedc has been shown
        not to do it.
        """
        return self.potentiation != "measured_no"


@dataclass(frozen=True, slots=True)
class MushroomBodyAtlas:
    """The full compartment table."""

    compartments: tuple[Compartment, ...]
    unassigned_dan_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        names = [c.name for c in self.compartments]
        if len(set(names)) != len(names):
            raise ValueError("compartment names must be unique")

        seen_mbon: dict[str, str] = {}
        seen_dan: dict[str, str] = {}
        for compartment in self.compartments:
            for mbon in compartment.mbon_types:
                if mbon in seen_mbon:
                    raise ValueError(
                        f"{mbon} is assigned to both {seen_mbon[mbon]} and {compartment.name}; "
                        "each output neuron type must have one compartment"
                    )
                seen_mbon[mbon] = compartment.name
            for dan in compartment.dan_types:
                if dan in seen_dan:
                    raise ValueError(
                        f"{dan} is assigned to both {seen_dan[dan]} and {compartment.name}"
                    )
                seen_dan[dan] = compartment.name

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.compartments)

    def __len__(self) -> int:
        return len(self.compartments)

    def index_of(self, name: str) -> int:
        """Position of a compartment in the table, used to index concentrations."""
        for i, compartment in enumerate(self.compartments):
            if compartment.name == name:
                return i
        raise KeyError(f"unknown compartment {name!r}; available: {self.names}")

    def get(self, name: str) -> Compartment:
        return self.compartments[self.index_of(name)]

    def compartment_of_mbon(self, cell_type: str) -> str | None:
        """Compartment read out by an output neuron type, if any."""
        for compartment in self.compartments:
            if cell_type in compartment.mbon_types:
                return compartment.name
        return None

    def compartment_of_dan(self, cell_type: str) -> str | None:
        """Compartment innervated by a dopaminergic neuron type, if any."""
        for compartment in self.compartments:
            if cell_type in compartment.dan_types:
                return compartment.name
        return None

    def dan_types(self) -> tuple[str, ...]:
        return tuple(t for c in self.compartments for t in c.dan_types)

    def mbon_types(self) -> tuple[str, ...]:
        return tuple(t for c in self.compartments for t in c.mbon_types)

    def missing_from(self, known_types: set[str]) -> dict[str, list[str]]:
        """Types in the atlas that the connectome annotations do not contain.

        An empty result means the table and the connectome release agree.
        """
        return {
            "dan": sorted(t for t in self.dan_types() if t not in known_types),
            "mbon": sorted(t for t in self.mbon_types() if t not in known_types),
        }


def load_atlas(path: str | Path | None = None) -> MushroomBodyAtlas:
    """Load the compartment table, by default the one packaged with the library."""
    if path is None:
        text = (
            resources.files("flyneuromod.data").joinpath("mb_compartments.yaml").read_text()
        )
    else:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"compartment table {path} not found")
        text = path.read_text()

    raw = yaml.safe_load(text)
    if not isinstance(raw, dict) or "compartments" not in raw:
        raise ValueError("compartment table must be a mapping with a 'compartments' key")

    compartments = tuple(
        Compartment(
            name=entry["name"],
            label=entry["label"],
            dan_types=tuple(entry.get("dan_types", ())),
            mbon_types=tuple(entry.get("mbon_types", ())),
            mbon_transmitter=entry.get("mbon_transmitter"),
            valence=entry.get("valence"),
            potentiation=entry.get("potentiation", "unknown"),
            note=entry.get("note", ""),
        )
        for entry in raw["compartments"]
    )
    atlas = MushroomBodyAtlas(
        compartments=compartments,
        unassigned_dan_types=tuple(raw.get("unassigned_dan_types", ())),
    )
    logger.info("loaded %d mushroom body compartments", len(atlas))
    return atlas
