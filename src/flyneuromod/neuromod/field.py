"""Extracellular dopamine concentration: release, pooling, DAT uptake.

Dopamine does not act like a fast synapse. It is released from varicosities of
dopaminergic neurons, spreads over a small volume (a mushroom body compartment
is the classic example), and is cleared by the dopamine transporter. Clearance
is an enzymatic process and therefore saturates, which is why it is modelled
with Michaelis-Menten kinetics rather than an exponential decay: that is also
what makes transporter blockers (cocaine, methylphenidate, nomifensine) and the
``fumin`` transporter mutant representable as a change of the apparent ``K_m``.

Concentrations are in micromolar, times in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ReleaseKinetics:
    """Release and clearance constants of one neuromodulator.

    Attributes
    ----------
    per_spike:
        Concentration increment in the field per presynaptic spike (micromolar).
    v_max:
        Maximum uptake rate of the transporter (micromolar per second).
    k_m:
        Michaelis constant of the transporter (micromolar). A competitive
        blocker is applied by raising it: ``k_m * (1 + [drug] / K_i)``.
    k_diffusion:
        First-order rate at which dopamine escapes the compartment by diffusion
        (per second). Transporter uptake alone saturates at ``v_max``, so a
        compartment driven harder than that would accumulate dopamine without
        bound - which the fly brain does not do: transporter-null flies still
        clear dopamine, and the brain is small enough for diffusion to matter
        (Makos et al. 2010; Vickrey et al. 2013).
    baseline:
        Tonic concentration maintained in the absence of spiking (micromolar).
        A matching tonic release term is derived from it, so an unstimulated
        field sits exactly at ``baseline``.
    """

    per_spike: float
    v_max: float
    k_m: float
    baseline: float = 0.0
    k_diffusion: float = 0.0

    def __post_init__(self) -> None:
        if self.per_spike < 0:
            raise ValueError("per_spike must not be negative")
        if self.v_max < 0:
            raise ValueError("v_max must not be negative")
        if self.k_m <= 0:
            raise ValueError("k_m must be positive")
        if self.baseline < 0:
            raise ValueError("baseline must not be negative")
        if self.k_diffusion < 0:
            raise ValueError("k_diffusion must not be negative")

    def clearance_at(self, concentration: float | np.ndarray) -> float | np.ndarray:
        """Total clearance rate at a given concentration (micromolar/second)."""
        transporter = self.v_max * concentration / (self.k_m + concentration)
        return transporter + self.k_diffusion * concentration

    @property
    def tonic_release(self) -> float:
        """Release rate that balances clearance at ``baseline`` (micromolar/second)."""
        if self.baseline == 0.0:
            return 0.0
        return float(self.clearance_at(self.baseline))

    def evolve(self, **changes: Any) -> ReleaseKinetics:
        """Return a new kinetics object with ``changes`` applied."""
        return replace(self, **changes)


class DopamineField:
    """Concentration of dopamine in a set of release fields.

    A *field* is a volume over which released dopamine is treated as well mixed
    — a mushroom body compartment, or a single postsynaptic neuron when the
    model is run at synaptic resolution.

    Parameters
    ----------
    n_fields:
        Number of independent fields.
    kinetics:
        Release and uptake constants.
    dt:
        Time step of the slow layer in seconds.
    """

    def __init__(self, n_fields: int, kinetics: ReleaseKinetics, dt: float) -> None:
        if n_fields <= 0:
            raise ValueError("n_fields must be positive")
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.n_fields = int(n_fields)
        self.kinetics = kinetics
        self.dt = float(dt)
        self.concentration = np.full(self.n_fields, kinetics.baseline, dtype=np.float64)

    def reset(self) -> None:
        """Return every field to its baseline concentration."""
        self.concentration[:] = self.kinetics.baseline

    def step(self, spikes: np.ndarray) -> np.ndarray:
        """Advance the concentrations by one ``dt``.

        Parameters
        ----------
        spikes:
            Number of dopaminergic spikes delivered to each field during this
            step; fractional values are allowed (a field can pool the spikes of
            several neurons weighted by their number of release sites).

        Returns
        -------
        numpy.ndarray
            The updated concentration of every field (micromolar).
        """
        spikes = np.asarray(spikes, dtype=np.float64)
        if spikes.shape != (self.n_fields,):
            raise ValueError(
                f"expected {self.n_fields} release values, got shape {spikes.shape}"
            )

        c = self.concentration
        c += self.kinetics.per_spike * spikes
        c += self.dt * (self.kinetics.tonic_release - self.kinetics.clearance_at(c))
        np.clip(c, 0.0, None, out=c)
        return c
