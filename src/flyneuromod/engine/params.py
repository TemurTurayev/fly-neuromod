"""Parameters of the fast (spiking) network layer.

All values default to the published whole-brain leaky integrate-and-fire model of
Shiu et al. (2024), *Nature* 634:210-219, doi:10.1038/s41586-024-07763-9, so that
simulations here can be compared directly with the reference implementation
(https://github.com/philshiu/Drosophila_brain_model, MIT licence).

Provenance of each constant is given in ``docs/parameters.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

MILLIVOLT = 1e-3  # volt
MILLISECOND = 1e-3  # second


@dataclass(frozen=True, slots=True)
class LIFParams:
    """Immutable parameter set of the leaky integrate-and-fire network.

    Units are SI (volt, second, hertz). Use :meth:`evolve` to derive a modified
    copy; instances are never mutated in place.

    Attributes
    ----------
    v_rest:
        Resting potential the membrane relaxes towards.
    v_reset:
        Potential the membrane is set to after a spike.
    v_threshold:
        Spike threshold.
    tau_membrane:
        Membrane time constant (resistance * capacitance).
    tau_synapse:
        Decay time constant of the synaptic input variable ``g``.
    t_refractory:
        Absolute refractory period; ``v`` and ``g`` are frozen while it lasts.
    t_delay:
        Axonal/synaptic transmission delay applied to every connection.
    w_synapse:
        Voltage increment per single anatomical synapse (free parameter fitted in
        the reference model).
    dt:
        Integration step of the fast layer.
    """

    v_rest: float = -52.0 * MILLIVOLT
    v_reset: float = -52.0 * MILLIVOLT
    v_threshold: float = -45.0 * MILLIVOLT
    tau_membrane: float = 20.0 * MILLISECOND
    tau_synapse: float = 5.0 * MILLISECOND
    t_refractory: float = 2.2 * MILLISECOND
    t_delay: float = 1.8 * MILLISECOND
    w_synapse: float = 0.275 * MILLIVOLT
    dt: float = 0.1 * MILLISECOND

    def __post_init__(self) -> None:
        if self.tau_membrane <= 0 or self.tau_synapse <= 0:
            raise ValueError("time constants must be positive")
        if self.tau_membrane == self.tau_synapse:
            raise ValueError(
                "tau_membrane must differ from tau_synapse; the exact propagator is "
                "singular when they are equal"
            )
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.dt > self.tau_synapse:
            raise ValueError(
                f"dt ({self.dt} s) must not exceed tau_synapse ({self.tau_synapse} s); "
                "the synaptic decay would be undersampled"
            )
        if self.t_refractory < 0 or self.t_delay < 0:
            raise ValueError("refractory period and delay must not be negative")
        if self.v_threshold <= self.v_rest:
            raise ValueError("threshold must be above the resting potential")

    @property
    def delay_steps(self) -> int:
        """Transmission delay expressed in integration steps (at least one)."""
        return max(1, round(self.t_delay / self.dt))

    @property
    def refractory_steps(self) -> int:
        """Refractory period expressed in integration steps."""
        return round(self.t_refractory / self.dt)

    def evolve(self, **changes: Any) -> LIFParams:
        """Return a new parameter set with ``changes`` applied."""
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class PoissonDrive:
    """External Poisson drive emulating optogenetic activation.

    In the reference model each driven neuron receives an independent Poisson
    train that is added directly to the membrane potential, and its refractory
    period is set to zero.

    Attributes
    ----------
    rate:
        Mean event rate in hertz.
    weight_factor:
        Scaling factor applied to ``w_synapse`` (250 in the reference model,
        which reliably drives spiking).
    """

    rate: float = 150.0
    weight_factor: float = 250.0

    def __post_init__(self) -> None:
        if self.rate < 0:
            raise ValueError("Poisson rate must not be negative")
        if self.weight_factor <= 0:
            raise ValueError("weight factor must be positive")

    def weight(self, params: LIFParams) -> float:
        """Voltage increment per Poisson event."""
        return self.weight_factor * params.w_synapse
