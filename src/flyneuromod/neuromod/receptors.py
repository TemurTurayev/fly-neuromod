"""Dopamine receptors and the second messengers they drive.

*Drosophila* has four dopamine receptors, all of them G-protein-coupled:

===========  ==========  ====================================================
Receptor     Coupling    Role modelled here
===========  ==========  ====================================================
Dop1R1       Gs          raises cAMP; required for learning in Kenyon cells
Dop1R2       Gq          raises intracellular calcium; forgetting, arousal
Dop2R        Gi          lowers cAMP; autoreceptor and postsynaptic brake
DopEcR       Gs/other    not modelled in v0.1
===========  ==========  ====================================================

Occupancy follows a Hill curve with first-order kinetics, which is enough to
reproduce the two features that matter for behaviour: receptors with different
affinities respond to different dopamine regimes (tonic versus phasic), and the
response outlasts the dopamine transient.

Concentrations are micromolar, times seconds. Second messengers are in
normalised units: 1.0 is the response to full occupancy of a Gs receptor.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

COUPLINGS = ("Gs", "Gi", "Gq")


@dataclass(frozen=True, slots=True)
class ReceptorSpec:
    """Pharmacological description of one receptor type.

    Attributes
    ----------
    name:
        Gene name, e.g. ``"Dop1R1"``.
    ec50:
        Dopamine concentration producing half-maximal occupancy (micromolar).
    hill:
        Hill coefficient of the dose-response curve.
    tau_on, tau_off:
        Time constants of the rise and the decay of receptor activation
        (seconds). ``tau_off`` includes receptor deactivation, not transmitter
        clearance, which the field model handles separately.
    coupling:
        One of ``"Gs"``, ``"Gi"``, ``"Gq"``.
    """

    name: str
    ec50: float
    hill: float
    tau_on: float
    tau_off: float
    coupling: str

    def __post_init__(self) -> None:
        if self.ec50 <= 0:
            raise ValueError(f"{self.name}: ec50 must be positive")
        if self.hill <= 0:
            raise ValueError(f"{self.name}: hill coefficient must be positive")
        if self.tau_on <= 0 or self.tau_off <= 0:
            raise ValueError(f"{self.name}: time constants must be positive")
        if self.coupling not in COUPLINGS:
            raise ValueError(f"{self.name}: coupling must be one of {COUPLINGS}")

    def evolve(self, **changes: Any) -> ReceptorSpec:
        return replace(self, **changes)


class ReceptorPopulation:
    """Activation state of one receptor type across a set of cells.

    Parameters
    ----------
    spec:
        Receptor properties.
    n_cells:
        Number of cells (or cell compartments) carrying the receptor.
    dt:
        Time step of the slow layer in seconds.
    occupancy_scale:
        Per-cell expression level in [0, 1]. Zero models a cell-type-specific
        knockout or a cell that does not express the receptor.
    """

    def __init__(
        self,
        spec: ReceptorSpec,
        n_cells: int,
        dt: float,
        occupancy_scale: np.ndarray | float = 1.0,
    ) -> None:
        if n_cells <= 0:
            raise ValueError("n_cells must be positive")
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.spec = spec
        self.n_cells = int(n_cells)
        self.dt = float(dt)
        self.occupancy_scale = np.broadcast_to(
            np.asarray(occupancy_scale, dtype=np.float64), (self.n_cells,)
        ).copy()
        if np.any(self.occupancy_scale < 0):
            raise ValueError("occupancy_scale must not be negative")
        self.occupancy = np.zeros(self.n_cells, dtype=np.float64)
        self._antagonist_factor = 1.0
        self._decay_on = float(np.exp(-self.dt / spec.tau_on))
        self._decay_off = float(np.exp(-self.dt / spec.tau_off))

    def set_competitive_antagonist(self, concentration: float, k_i: float) -> None:
        """Apply a competitive antagonist, shifting the apparent EC50.

        The apparent EC50 becomes ``ec50 * (1 + [antagonist] / K_i)``, the
        standard Schild relation. Pass ``concentration=0`` to wash it out.
        """
        if concentration < 0:
            raise ValueError("antagonist concentration must not be negative")
        if k_i <= 0:
            raise ValueError("K_i must be positive")
        self._antagonist_factor = 1.0 + concentration / k_i

    def reset(self) -> None:
        self.occupancy[:] = 0.0

    def step(self, concentration: np.ndarray) -> np.ndarray:
        """Advance receptor activation by one ``dt``.

        Parameters
        ----------
        concentration:
            Dopamine concentration seen by each cell (micromolar).

        Returns
        -------
        numpy.ndarray
            Fractional activation of the receptor in every cell.
        """
        concentration = np.asarray(concentration, dtype=np.float64)
        if concentration.shape != (self.n_cells,):
            raise ValueError(
                f"expected {self.n_cells} concentrations, got shape {concentration.shape}"
            )

        ec50 = self.spec.ec50 * self._antagonist_factor
        ligand = np.power(np.maximum(concentration, 0.0), self.spec.hill)
        target = self.occupancy_scale * ligand / (ligand + ec50**self.spec.hill)

        # exact relaxation towards the target over one step: stable for any dt,
        # and it can never overshoot past the target
        decay = np.where(target > self.occupancy, self._decay_on, self._decay_off)
        self.occupancy[:] = target + (self.occupancy - target) * decay
        return self.occupancy.copy()


@dataclass(frozen=True, slots=True)
class SignalingParams:
    """Kinetics of the intracellular signals the receptors drive.

    Attributes
    ----------
    tau_camp, tau_calcium:
        Relaxation time constants of cAMP and of the Gq-driven calcium signal.
    gain_gs, gain_gi, gain_gq:
        Contribution of full receptor occupancy to the respective signal.
    basal_camp:
        cAMP level with no dopamine present.
    """

    tau_camp: float = 1.0
    tau_calcium: float = 0.3
    gain_gs: float = 1.0
    gain_gi: float = 1.0
    gain_gq: float = 1.0
    basal_camp: float = 0.0

    def __post_init__(self) -> None:
        if self.tau_camp <= 0 or self.tau_calcium <= 0:
            raise ValueError("signalling time constants must be positive")


class SecondMessenger:
    """cAMP and calcium driven by receptor occupancy.

    ``calcium`` is the Gq branch: IP₃ production and the store calcium it
    releases, which the plasticity rule reads as its IP₃ signal. ``camp`` is kept
    as a readout comparable with cAMP imaging; the rule itself reads receptor
    activation, because the coincidence detection happens at the cyclase, before
    cAMP accumulates.
    """

    def __init__(self, n_cells: int, params: SignalingParams | None = None, dt: float = 1e-3):
        if n_cells <= 0:
            raise ValueError("n_cells must be positive")
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.n_cells = int(n_cells)
        self.params = params or SignalingParams()
        self.dt = float(dt)
        self.camp = np.full(self.n_cells, self.params.basal_camp, dtype=np.float64)
        self.calcium = np.zeros(self.n_cells, dtype=np.float64)
        self._decay_camp = float(np.exp(-self.dt / self.params.tau_camp))
        self._decay_calcium = float(np.exp(-self.dt / self.params.tau_calcium))

    def reset(self) -> None:
        self.camp[:] = self.params.basal_camp
        self.calcium[:] = 0.0

    def step(
        self,
        gs_occupancy: np.ndarray,
        gi_occupancy: np.ndarray,
        gq_occupancy: np.ndarray | None = None,
    ) -> None:
        """Advance cAMP and calcium by one ``dt``."""
        p = self.params
        drive = p.basal_camp + p.gain_gs * gs_occupancy - p.gain_gi * gi_occupancy
        self.camp[:] = drive + (self.camp - drive) * self._decay_camp

        gq = np.zeros(self.n_cells) if gq_occupancy is None else gq_occupancy
        target = p.gain_gq * gq
        self.calcium[:] = target + (self.calcium - target) * self._decay_calcium
