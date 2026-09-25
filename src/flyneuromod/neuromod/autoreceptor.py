"""Presynaptic Dop2R autoreceptor feedback loop."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .receptors import ReceptorPopulation, ReceptorSpec


@dataclass(frozen=True, slots=True)
class AutoreceptorParams:
    """Parameters of the presynaptic autoreceptor feedback loop.

    Attributes
    ----------
    spec:
        Receptor properties (typically Dop2R).
    max_suppression:
        Maximum fraction of release suppressed at full receptor occupancy [0, 1].
    min_gain:
        Lower bound on the release gain array (0, 1].
    """

    spec: ReceptorSpec
    max_suppression: float
    min_gain: float = 0.1

    def __post_init__(self) -> None:
        if not (0.0 <= self.max_suppression <= 1.0):
            raise ValueError(
                f"max_suppression must be between 0 and 1, got {self.max_suppression}"
            )
        if not (0.0 < self.min_gain <= 1.0):
            raise ValueError(
                f"min_gain must be in (0, 1], got {self.min_gain}"
            )

    def evolve(self, **changes: Any) -> AutoreceptorParams:
        """Return a new AutoreceptorParams object with changes applied."""
        return replace(self, **changes)


class AutoreceptorFeedback:
    """Presynaptic autoreceptor feedback loop on dopaminergic terminals.

    Parameters
    ----------
    params:
        Autoreceptor configuration.
    n_fields:
        Number of independent release fields.
    dt:
        Time step of the slow layer in seconds.
    occupancy_scale:
        Per-field expression level in [0, 1].
    """

    def __init__(
        self,
        params: AutoreceptorParams,
        n_fields: int,
        dt: float,
        occupancy_scale: np.ndarray | float = 1.0,
    ) -> None:
        self.params = params
        self.population = ReceptorPopulation(
            spec=params.spec,
            n_cells=n_fields,
            dt=dt,
            occupancy_scale=occupancy_scale,
        )

    def reset(self) -> None:
        """Return receptor occupancy to zero."""
        self.population.reset()

    def step(self, concentration: np.ndarray) -> np.ndarray:
        """Advance receptor state by one ``dt`` and return the release gain array.

        Parameters
        ----------
        concentration:
            Extracellular dopamine concentration seen by each field (micromolar).

        Returns
        -------
        numpy.ndarray
            Release gain per field clipped to [min_gain, 1.0].
        """
        occupancy = self.population.step(concentration)
        gain = 1.0 - self.params.max_suppression * occupancy
        return np.clip(gain, self.params.min_gain, 1.0)
