"""Dopamine-gated plasticity at Kenyon cell output synapses.

The rule implemented here is the one supported by the *Drosophila* mushroom body
literature, not a generic reinforcement-learning update:

* **Three factors.** A synapse changes only when presynaptic activity and
  dopamine coincide *in the same compartment*. Dopamine alone does nothing;
  Kenyon cell activity alone does nothing. Odour specificity follows for free,
  because only the Kenyon cells that carried the odour hold an eligibility trace.
* **The sign depends on the order.** Kenyon cell activity followed by dopamine
  depresses the synapse through the Gs/cAMP branch (Dop1R1); dopamine followed
  by Kenyon cell activity potentiates it through the Gq/calcium branch
  (Dop1R2). Handler et al. (2019), *Cell* 178:60, doi:10.1016/j.cell.2019.05.040.
* **Presynaptic expression.** The change lives in the Kenyon cell terminal, so
  it applies per (Kenyon cell, compartment), and the same Kenyon cell can be
  depressed in one compartment while unchanged in another. Hige et al. (2015),
  *Neuron* 88:985, doi:10.1016/j.neuron.2015.11.003.

The eligibility trace is what turns the millisecond-scale spiking layer into
something a second-scale modulator can act on; its time constant sets the
coincidence window (optimal 0.1-1 s, gone by ~6 s in Handler et al. 2019).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PlasticityParams:
    """Constants of the three-factor rule.

    Attributes
    ----------
    tau_eligibility:
        Decay time constant of the presynaptic eligibility trace (seconds). It
        sets how long after an odour dopamine can still write a memory.
    rate_depression:
        Depression rate per unit of (cAMP x eligibility) per second.
    rate_potentiation:
        Potentiation rate per unit of (calcium x presynaptic spike).
    min_fraction, max_fraction:
        Bounds on the synaptic weight as a fraction of its anatomical value.
    tau_recovery:
        Optional time constant of the slow drift back to the anatomical weight
        (passive forgetting). ``None`` disables it.
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
        after every step, which is how the plasticity reaches the running
        network.
    synapse_index:
        Positions of the plastic synapses inside ``weights``.
    presynaptic_index:
        For each plastic synapse, the index of its presynaptic cell in the
        spike-count vector passed to :meth:`step`.
    compartment_index:
        For each plastic synapse, the index of the dopamine compartment it sits
        in, used to look up cAMP and calcium.
    params:
        Learning-rule constants.
    dt:
        Time step of the slow layer in seconds.
    """

    def __init__(
        self,
        weights: np.ndarray,
        synapse_index: np.ndarray,
        presynaptic_index: np.ndarray,
        compartment_index: np.ndarray,
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

        self.weights = weights
        self.synapse_index = synapse_index
        self.presynaptic_index = presynaptic_index
        self.compartment_index = compartment_index
        self.params = params
        self.dt = float(dt)

        self.baseline_weight = np.array(weights[synapse_index], dtype=np.float64)
        self.weight_factor = np.ones(len(synapse_index), dtype=np.float64)
        self.eligibility = np.zeros(len(synapse_index), dtype=np.float64)
        self._eligibility_decay = float(np.exp(-dt / params.tau_eligibility))

    def reset(self) -> None:
        """Restore anatomical weights and clear the eligibility trace."""
        self.weight_factor[:] = 1.0
        self.eligibility[:] = 0.0
        self.weights[self.synapse_index] = self.baseline_weight

    def step(self, presynaptic_spikes: np.ndarray, camp: np.ndarray, calcium: np.ndarray) -> None:
        """Advance the plastic weights by one slow step.

        Parameters
        ----------
        presynaptic_spikes:
            Spikes emitted by each presynaptic cell during this step.
        camp:
            cAMP level per compartment (Gs branch, drives depression).
        calcium:
            Calcium level per compartment (Gq branch, drives potentiation).
        """
        spikes = np.asarray(presynaptic_spikes, dtype=np.float64)[self.presynaptic_index]
        camp_here = np.asarray(camp, dtype=np.float64)[self.compartment_index]
        calcium_here = np.asarray(calcium, dtype=np.float64)[self.compartment_index]

        # presynaptic activity leaves a decaying trace; dopamine arriving while
        # the trace is up depresses the synapse (forward pairing)
        self.eligibility *= self._eligibility_decay
        depression = self.params.rate_depression * np.maximum(camp_here, 0.0) * self.eligibility
        self.eligibility += spikes

        # dopamine that arrived first leaves calcium elevated; Kenyon cell
        # spikes landing in that window potentiate instead (backward pairing)
        potentiation = self.params.rate_potentiation * np.maximum(calcium_here, 0.0) * spikes

        self.weight_factor += potentiation - self.dt * depression
        if self.params.tau_recovery is not None:
            self.weight_factor += self.dt * (1.0 - self.weight_factor) / self.params.tau_recovery
        np.clip(
            self.weight_factor,
            self.params.min_fraction,
            self.params.max_fraction,
            out=self.weight_factor,
        )

        self.weights[self.synapse_index] = self.baseline_weight * self.weight_factor

    def depression_of(self, mask: np.ndarray | None = None) -> float:
        """Mean weight change of the selected synapses, as a fraction of baseline.

        Returns a negative number for depression. This is the quantity reported
        by electrophysiological pairing experiments.
        """
        factors = self.weight_factor if mask is None else self.weight_factor[mask]
        if factors.size == 0:
            return 0.0
        return float(factors.mean() - 1.0)
