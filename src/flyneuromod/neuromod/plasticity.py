"""Dopamine-gated plasticity at Kenyon cell output synapses.

The rule is built from the two molecular coincidence detectors that the fly
literature places in the Kenyon cell terminal, and it inherits their sensitivity
to *order*:

**Depression: calcium first, then Gs.** The rutabaga adenylyl cyclase is
activated by Ca²⁺/calmodulin and by Gαs together, and it responds far more
strongly when the calcium signal is already present when the transmitter
arrives (Yovell & Abrams 1992, PNAS 89:6526; Levin et al. 1992, Cell 68:479).
Here, the Kenyon cell's recent activity is a calcium trace, and depression is
driven by the *arrival* of Dop1R1 (Gs) activation onto that trace.

**Potentiation: Gq first, then calcium.** The IP₃ receptor opens when IP₃ is
bound before calcium arrives, and is inhibited by calcium that comes first
(Bezprozvanny et al. 1991, Nature 351:751). Here, Dop1R2 (Gq) activation leaves
an IP₃ trace, and potentiation is driven by the *arrival* of Kenyon cell calcium
onto it.

Together they give the result that a symmetric product of "dopamine × activity"
cannot give, whatever its parameters: odour then dopamine depresses, dopamine
then odour potentiates (Handler et al. 2019, Cell 178:60). A simpler rule was
tried first and never flipped sign — dopamine lingers for seconds, so any rule
that only multiplies concentrations sees both orders as overlap.

Both drives are increments per arriving event, not rates, so the amount learned
does not depend on the integration step.

The change is expressed presynaptically, per (Kenyon cell, compartment), as
measured by Hige et al. (2015), Neuron 88:985.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PlasticityParams:
    """Constants of the order-selective rule.

    Attributes
    ----------
    tau_eligibility:
        Decay time constant of the presynaptic calcium trace (seconds). It sets
        how long after an odour dopamine can still write a memory.
    rate_depression:
        Fraction of the weight lost when Gs activation rises by 1.0 onto a fully
        primed terminal.
    rate_potentiation:
        Fraction of the weight gained when a terminal goes from silent to fully
        active while the IP₃ signal is 1.0.
    min_fraction, max_fraction:
        Bounds on the synaptic weight as a fraction of its anatomical value.
    tau_recovery:
        Optional time constant of a slow drift back to the anatomical weight.
        ``None`` disables it.
    """

    tau_eligibility: float
    rate_depression: float
    rate_potentiation: float
    min_fraction: float = 0.0
    max_fraction: float = 1.5
    tau_recovery: float | None = None

    def __post_init__(self) -> None:
        if self.tau_eligibility <= 0:
            raise ValueError("tau_eligibility must be positive")
        if self.rate_depression < 0 or self.rate_potentiation < 0:
            raise ValueError("learning rates must not be negative")
        if not 0.0 <= self.min_fraction <= 1.0 <= self.max_fraction:
            raise ValueError("bounds must satisfy 0 <= min_fraction <= 1 <= max_fraction")
        if self.tau_recovery is not None and self.tau_recovery <= 0:
            raise ValueError("tau_recovery must be positive or None")

    def evolve(self, **changes: Any) -> PlasticityParams:
        return replace(self, **changes)


class SynapticPlasticity:
    """Dopamine-gated weights for a selected set of synapses.

    Parameters
    ----------
    weights:
        The simulator's weight array. The selected entries are written in place
        after every step, which is how plasticity reaches the running network.
    synapse_index:
        Positions of the plastic synapses inside ``weights``.
    presynaptic_index:
        For each plastic synapse, the index of its presynaptic cell in the
        spike-count vector passed to :meth:`step`.
    compartment_index:
        For each plastic synapse, the index of the dopamine field it sits in.
    n_compartments:
        Number of dopamine fields.
    params:
        Rule constants.
    dt:
        Time step of the slow layer in seconds.
    """

    def __init__(
        self,
        weights: np.ndarray,
        synapse_index: np.ndarray,
        presynaptic_index: np.ndarray,
        compartment_index: np.ndarray,
        n_compartments: int,
        params: PlasticityParams,
        dt: float,
    ) -> None:
        synapse_index = np.asarray(synapse_index, dtype=np.int64)
        presynaptic_index = np.asarray(presynaptic_index, dtype=np.int64)
        compartment_index = np.asarray(compartment_index, dtype=np.int64)
        if not (len(synapse_index) == len(presynaptic_index) == len(compartment_index)):
            raise ValueError(
                "synapse_index, presynaptic_index and compartment_index must have equal length"
            )
        if dt <= 0:
            raise ValueError("dt must be positive")
        if n_compartments <= 0:
            raise ValueError("n_compartments must be positive")
        if compartment_index.size and (
            compartment_index.min() < 0 or compartment_index.max() >= n_compartments
        ):
            raise ValueError("compartment_index outside the number of compartments")

        self.weights = weights
        self.synapse_index = synapse_index
        self.presynaptic_index = presynaptic_index
        self.compartment_index = compartment_index
        self.n_compartments = int(n_compartments)
        self.params = params
        self.dt = float(dt)

        self.baseline_weight = np.array(weights[synapse_index], dtype=np.float64)
        self.weight_factor = np.ones(len(synapse_index), dtype=np.float64)
        self.eligibility = np.zeros(len(synapse_index), dtype=np.float64)
        self._previous_gs = np.zeros(self.n_compartments, dtype=np.float64)
        self._eligibility_decay = float(np.exp(-dt / params.tau_eligibility))

    def reset(self) -> None:
        """Restore anatomical weights and clear every trace."""
        self.weight_factor[:] = 1.0
        self.eligibility[:] = 0.0
        self._previous_gs[:] = 0.0
        self.weights[self.synapse_index] = self.baseline_weight

    def step(
        self,
        presynaptic_spikes: np.ndarray,
        gs_activation: np.ndarray,
        ip3: np.ndarray,
    ) -> None:
        """Advance the plastic weights by one slow step.

        Parameters
        ----------
        presynaptic_spikes:
            Spikes emitted by each presynaptic cell during this step.
        gs_activation:
            Activation of Gs-coupled receptors per compartment, in [0, 1].
        ip3:
            IP₃ signal from Gq-coupled receptors per compartment, in [0, 1].
        """
        gs_activation = np.asarray(gs_activation, dtype=np.float64)
        ip3 = np.asarray(ip3, dtype=np.float64)
        if gs_activation.shape != (self.n_compartments,) or ip3.shape != (self.n_compartments,):
            raise ValueError(
                f"gs_activation and ip3 must have shape ({self.n_compartments},)"
            )

        spikes = np.asarray(presynaptic_spikes, dtype=np.float64)[self.presynaptic_index]
        active = np.minimum(spikes, 1.0)

        # depression: Gs activation *arriving* onto a terminal whose calcium
        # trace is already up (the trace from before this step's spikes)
        gs_arrival = np.maximum(gs_activation - self._previous_gs, 0.0)
        self._previous_gs[:] = gs_activation
        self.eligibility *= self._eligibility_decay
        depression = (
            self.params.rate_depression * gs_arrival[self.compartment_index] * self.eligibility
        )

        # potentiation: calcium *arriving* onto IP3 that is already there. The
        # IP3 receptor's calcium dependence is bell-shaped: calcium opens it, but
        # calcium that is already high inhibits it. So the drive is the arrival
        # times the fraction of receptors not yet inhibited, and a terminal that
        # has been firing all along - its trace refilled spike after spike - adds
        # almost nothing.
        headroom = 1.0 - self.eligibility
        calcium_arrival = headroom * active
        potentiation = (
            self.params.rate_potentiation
            * np.maximum(ip3, 0.0)[self.compartment_index]
            * calcium_arrival
            * headroom
        )
        self.eligibility += calcium_arrival

        self.weight_factor += potentiation - depression
        if self.params.tau_recovery is not None:
            self.weight_factor += self.dt * (1.0 - self.weight_factor) / self.params.tau_recovery
        np.clip(
            self.weight_factor,
            self.params.min_fraction,
            self.params.max_fraction,
            out=self.weight_factor,
        )
        self.weights[self.synapse_index] = self.baseline_weight * self.weight_factor

    def change_of(self, mask: np.ndarray | None = None) -> float:
        """Mean weight change of the selected synapses, as a fraction of baseline.

        Negative means depression. Returns ``nan`` when the selection is empty,
        so that "nothing measured" is never mistaken for "nothing changed".
        """
        factors = self.weight_factor if mask is None else self.weight_factor[mask]
        if factors.size == 0:
            logger.warning("weight change requested for an empty set of synapses")
            return float("nan")
        return float(factors.mean() - 1.0)
