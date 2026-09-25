"""Drugs and mutants as changes to dopamine-layer parameters.

Each entry states what the manipulation does in the fly and which measured
effect it is calibrated against. The point of expressing pharmacology this way
is that an experiment in the model uses the same knobs as an experiment at the
bench: block the transporter, knock out a receptor, inhibit synthesis.

A warning that matters for anyone porting mammalian intuitions: **SCH-23390,
raclopride, eticlopride, haloperidol and spiperone do not work on the fly
receptors they target in mammals** (Sugamori et al. 1995; Hearn et al. 2002;
Srivastava et al. 2005). They are included as inert controls on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .field import ReleaseKinetics


@dataclass(frozen=True, slots=True)
class Manipulation:
    """A drug or mutant expressed as multipliers on dopamine-layer parameters.

    Attributes
    ----------
    name:
        Common name of the drug or allele.
    k_m_factor:
        Multiplier on the transporter Michaelis constant. Competitive uptake
        blockers raise it, which slows clearance.
    release_factor:
        Multiplier on the amount of dopamine released per spike.
    receptor_block:
        Pairs of receptor name and the ratio ``[antagonist] / K_i``, used by
        :meth:`ReceptorPopulation.set_competitive_antagonist`. Stored as a tuple
        so that the object is genuinely immutable and hashable; read it through
        :attr:`blocks`.
    receptor_knockout:
        Receptor names whose expression is set to zero, modelling a null mutant
        or cell-type-specific knockdown.
    note:
        What the manipulation does in the animal, and the calibration source.
    """

    name: str
    k_m_factor: float = 1.0
    release_factor: float = 1.0
    receptor_block: tuple[tuple[str, float], ...] = ()
    receptor_knockout: tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if self.k_m_factor <= 0:
            raise ValueError(f"{self.name}: k_m_factor must be positive")
        if self.release_factor < 0:
            raise ValueError(f"{self.name}: release_factor must not be negative")
        for receptor, ratio in self.receptor_block:
            if ratio < 0:
                raise ValueError(f"{self.name}: block ratio for {receptor} must not be negative")

    @property
    def blocks(self) -> dict[str, float]:
        """Receptor blocks as a fresh dictionary (changing it changes nothing)."""
        return dict(self.receptor_block)

    @property
    def receptors_named(self) -> set[str]:
        """Every receptor name this manipulation refers to."""
        return set(self.receptor_knockout) | {name for name, _ in self.receptor_block}

    def apply_to(self, kinetics: ReleaseKinetics) -> ReleaseKinetics:
        """Return the release kinetics as modified by this manipulation."""
        return kinetics.evolve(
            k_m=kinetics.k_m * self.k_m_factor,
            per_spike=kinetics.per_spike * self.release_factor,
        )

    def evolve(self, **changes: Any) -> Manipulation:
        return replace(self, **changes)


CONTROL = Manipulation(name="control", note="no manipulation")

COCAINE = Manipulation(
    name="cocaine",
    k_m_factor=3.0,
    note=(
        "Competitive block of the dopamine transporter; slows clearance in wild-type "
        "flies and has no effect in the transporter-null fumin mutant "
        "(Makos et al. 2010, ACS Chem Neurosci 1:74, doi:10.1021/cn900017w). The "
        "factor is calibrated against the ~3x rise in evoked dopamine seen with the "
        "uptake blocker nisoxetine (Shin & Venton 2022)."
    ),
)

METHYLPHENIDATE = COCAINE.evolve(
    name="methylphenidate",
    note=(
        "Transporter blocker with the same qualitative effect as cocaine in the fly "
        "(Makos et al. 2010); the factor is the same calibration and is approximate."
    ),
)

FUMIN = Manipulation(
    name="fumin (dDAT null)",
    k_m_factor=20.0,
    note=(
        "Transporter-null mutant: dopamine is cleared only by diffusion and "
        "metabolism, so clearance is much slower and uptake blockers stop working "
        "(Makos et al. 2010). The factor is assumed - no kinetic fit exists."
    ),
)

FLUPENTIXOL = Manipulation(
    name="flupentixol",
    receptor_block=(("Dop2R", 1000.0),),
    note=(
        "Blocks the Dop2R autoreceptor, which normally suppresses release: evoked "
        "dopamine in the mushroom body rises from 0.31 to 1.2 uM, about fourfold "
        "(Shin & Venton 2022). Calibrated: saturating competitive block of Dop2R "
        "([antagonist]/K_i = 1000) shifts apparent EC50 >1000-fold, preventing "
        "autoreceptor activation and restoring unsuppressed release."
    ),
)

IODOTYROSINE = Manipulation(
    name="3-iodotyrosine",
    release_factor=0.3,
    note=(
        "Inhibits tyrosine hydroxylase. It does not change a single evoked release, "
        "but depletes the pool under repeated stimulation (Xiao & Venton 2015, "
        "J Neurochem 134:445). The factor is assumed and represents the depleted state."
    ),
)

SCH23390 = Manipulation(
    name="SCH-23390",
    note=(
        "INERT CONTROL. A potent mammalian D1 antagonist that does not block the fly "
        "D1-like receptors (Sugamori et al. 1995, FEBS Lett 362:131). Included so that "
        "mammalian pharmacology is not applied to the fly by mistake."
    ),
)

DOP1R1_KNOCKOUT = Manipulation(
    name="Dop1R1 null (dumb)",
    receptor_knockout=("Dop1R1",),
    note=(
        "Removes the Gs/cAMP branch. Predicted effect in the model: pairing no longer "
        "depresses Kenyon cell output synapses, mirroring the learning deficit of "
        "dumb mutants (Kim et al. 2007, J Neurosci 27:7640)."
    ),
)

DOP1R2_KNOCKOUT = Manipulation(
    name="Dop1R2 null (damb)",
    receptor_knockout=("Dop1R2(Gq)",),
    note=(
        "Removes the Gq/calcium branch. Predicted effect: loss of the potentiating, "
        "backward-pairing arm and of dopamine-driven forgetting "
        "(Berry et al. 2012, Neuron 74:530; Handler et al. 2019)."
    ),
)

CATALOGUE: dict[str, Manipulation] = {
    "control": CONTROL,
    "cocaine": COCAINE,
    "methylphenidate": METHYLPHENIDATE,
    "fumin": FUMIN,
    "flupentixol": FLUPENTIXOL,
    "iodotyrosine": IODOTYROSINE,
    "sch23390": SCH23390,
    "dop1r1-null": DOP1R1_KNOCKOUT,
    "dop1r2-null": DOP1R2_KNOCKOUT,
}
"""Manipulations by command-line name."""


def get(name: str) -> Manipulation:
    """Look up a manipulation by its catalogue key (e.g. ``"dop1r1-null"``).

    Raises
    ------
    KeyError
        If the name is unknown, listing what is available.
    """
    try:
        return CATALOGUE[name]
    except KeyError:
        raise KeyError(f"unknown manipulation {name!r}; available: {sorted(CATALOGUE)}") from None
