"""Intrinsic excitability effector modeling potassium conductance modulation."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Provenance: SI unit definitions
MILLIVOLT: float = 1e-3  # Provenance: SI unit definition (1 mV = 1e-3 V)
MILLISECOND: float = 1e-3  # Provenance: SI unit definition (1 ms = 1e-3 s)

# Provenance: assumed typical insect neuron K+ reversal (-80 mV); no dFB measurement is used.
DEFAULT_E_K: float = -80.0 * MILLIVOLT

# Provenance: assumed placeholder to be calibrated in step 2 against Pimentel et al. 2016.
DEFAULT_MAX_CONDUCTANCE: float = 1.0

# Provenance: assumed placeholder to be calibrated in step 2 against Pimentel et al. 2016.
DEFAULT_TAU_EFFECTOR: float = 1.0


@dataclass(frozen=True, slots=True)
class ExcitabilityParams:
    """Parameters for intrinsic excitability modulation via K+ conductance.

    Attributes
    ----------
    max_conductance:
        Maximum K+ leak conductance addition at 100% activation (k >= 0).
        Provenance: assumed placeholder to be calibrated in step 2 against Pimentel et al. 2016.
    e_k:
        Potassium reversal potential in volts.
        Provenance: assumed typical insect neuron K+ reversal (-80 mV); no dFB measurement is used.
    tau_effector:
        Relaxation time constant of effector activation in seconds.
        Provenance: assumed placeholder to be calibrated in step 2 against Pimentel et al. 2016.
    """

    max_conductance: float = DEFAULT_MAX_CONDUCTANCE
    e_k: float = DEFAULT_E_K
    tau_effector: float = DEFAULT_TAU_EFFECTOR

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_conductance) or self.max_conductance < 0:
            raise ValueError("max_conductance must be non-negative and finite")
        if not math.isfinite(self.e_k):
            raise ValueError("e_k must be finite")
        if not math.isfinite(self.tau_effector) or self.tau_effector <= 0:
            raise ValueError("tau_effector must be positive and finite")

    def evolve(self, **changes: Any) -> ExcitabilityParams:
        """Return a new parameter set with changes applied."""
        return replace(self, **changes)


class ExcitabilityEffector:
    """Modulates intrinsic excitability by relaxing K+ conductance toward receptor occupancy."""

    def __init__(
        self,
        params: ExcitabilityParams,
        target_indices: Sequence[int] | np.ndarray,
        dt: float,
    ) -> None:
        if dt <= 0 or not math.isfinite(dt):
            raise ValueError("dt must be positive and finite")

        idx = np.asarray(target_indices, dtype=np.int64)
        if idx.ndim != 1:
            raise ValueError("target_indices must be 1D")
        if idx.size > 0 and np.any(idx < 0):
            raise ValueError("target_indices must be non-negative")

        self.params = params
        self.target_indices = idx
        self.dt = float(dt)
        self.activation = np.zeros(len(idx), dtype=np.float64)
        self._decay = float(math.exp(-self.dt / params.tau_effector))

    def reset(self) -> None:
        """Reset activation state to zero."""
        self.activation[:] = 0.0

    def step(self, gi_occupancy: float | Sequence[float] | np.ndarray) -> np.ndarray:
        """Advance activation toward receptor occupancy and return potassium conductance k.

        Parameters
        ----------
        gi_occupancy:
            Receptor occupancy O (scalar or array matching target_indices length).

        Returns
        -------
        np.ndarray
            Added K+ conductance k per target neuron (a new array).
        """
        occ = np.asarray(gi_occupancy, dtype=np.float64)
        if not np.all(np.isfinite(occ)):
            raise ValueError("gi_occupancy must be finite")

        if occ.ndim == 1 and occ.shape[0] != self.target_indices.shape[0]:
            raise ValueError("gi_occupancy shape does not match target_indices")
        if occ.ndim > 1:
            raise ValueError("gi_occupancy must be a scalar or 1D array")

        self.activation = occ + (self.activation - occ) * self._decay
        k = self.params.max_conductance * self.activation
        return np.array(k, dtype=np.float64)
