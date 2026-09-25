"""The dopamine layer: from dopaminergic spikes to changed synapses.

The layer runs alongside the spiking network as a step callback and closes this
loop once per slow step:

1. **Sources.** Spikes of curated dopaminergic neurons are pooled per mushroom
   body compartment *and hemisphere*, weighted by each neuron's share of the
   compartment's synapses onto Kenyon cells.
2. **Volume.** The pooled release raises the extracellular dopamine
   concentration of that field; transporter uptake and diffusion clear it.
3. **Receptors.** Dop1R1 (Gs) and Dop1R2 (Gq) bind dopamine with their own
   affinities and kinetics.
4. **Signalling.** Gq activation builds an IP₃ signal; cAMP is tracked as a
   readout comparable with imaging.
5. **Synapses.** Gs activation arriving onto a recently active Kenyon cell
   terminal depresses it; calcium arriving onto IP₃ potentiates it.

The fast network never sees dopamine as current. Dopaminergic connections are
removed from the spiking layer by default, because no ionotropic dopamine
receptor is known in *Drosophila*.

The left and right mushroom bodies are separate volumes, so each compartment
exists twice. Dopamine released on one side does not train the other.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..data.annotations import Annotations
from ..data.connectome import Connectome
from .autoreceptor import AutoreceptorFeedback, AutoreceptorParams
from .constants import (
    DAN_AUTORECEPTOR_FEEDBACK,
    DOPAMINE_RECEPTORS,
    KC_SIGNALING,
    KC_TO_MBON_PLASTICITY,
    MB_COMPARTMENT_RELEASE,
)
from .field import DopamineField, ReleaseKinetics
from .mb_atlas import MushroomBodyAtlas, load_atlas
from .pharmacology import CONTROL, Manipulation
from .plasticity import PlasticityParams, SynapticPlasticity
from .receptors import ReceptorPopulation, ReceptorSpec, SecondMessenger, SignalingParams

if TYPE_CHECKING:  # pragma: no cover
    from ..engine.lif import LIFNetwork

logger = logging.getLogger(__name__)

SIDES = ("left", "right")
MODULATOR_KIND = "dopamine"
STABILITY_FACTOR = 0.1
"""The slow step must be at most this fraction of the fastest time constant."""


@dataclass(frozen=True, slots=True)
class DopamineConfig:
    """Configuration of the dopamine layer.

    Attributes
    ----------
    slow_dt:
        Time step of the modulator in seconds. It must be a multiple of the
        network step and small compared with every time constant of the layer.
    release:
        Release and clearance kinetics of a compartment.
    receptors:
        Receptor types on Kenyon cell terminals. Names must be unique.
    autoreceptor:
        Presynaptic autoreceptor feedback loop on dopaminergic terminals (None disables).
    signaling:
        IP₃/calcium and cAMP kinetics.
    plasticity:
        Constants of the learning rule.
    remove_fast_dopamine_synapses:
        Zero the fast synaptic weights of dopaminergic neurons. Some of them
        genuinely co-release GABA or glutamate (Yamazaki et al. 2023), which this
        also removes; set it to ``False`` to keep them.
    manipulation:
        A drug or mutant from :mod:`flyneuromod.neuromod.pharmacology`. Every
        receptor it names must exist in ``receptors`` or ``autoreceptor``.
    """

    slow_dt: float = 1e-3
    release: ReleaseKinetics = MB_COMPARTMENT_RELEASE
    receptors: tuple[ReceptorSpec, ...] = DOPAMINE_RECEPTORS
    autoreceptor: AutoreceptorParams | None = DAN_AUTORECEPTOR_FEEDBACK
    signaling: SignalingParams = KC_SIGNALING
    plasticity: PlasticityParams = KC_TO_MBON_PLASTICITY
    remove_fast_dopamine_synapses: bool = True
    manipulation: Manipulation = CONTROL

    def __post_init__(self) -> None:
        if self.slow_dt <= 0:
            raise ValueError("slow_dt must be positive")

        names = [r.name for r in self.receptors]
        if len(set(names)) != len(names):
            raise ValueError(f"receptor names must be unique, got {names}")
        if "Gs" not in {r.coupling for r in self.receptors}:
            raise ValueError("a Gs-coupled receptor is required to drive depression")

        all_names = set(names)
        if self.autoreceptor is not None:
            all_names.add(self.autoreceptor.spec.name)
        unknown = self.manipulation.receptors_named - all_names
        if unknown:
            raise ValueError(
                f"{self.manipulation.name} refers to receptors {sorted(unknown)} "
                f"that are not configured ({sorted(all_names)}); it would silently do nothing"
            )

        autoreceptor_taus = (
            [self.autoreceptor.spec.tau_on, self.autoreceptor.spec.tau_off]
            if self.autoreceptor is not None
            else []
        )
        fastest = min(
            [r.tau_on for r in self.receptors]
            + [r.tau_off for r in self.receptors]
            + autoreceptor_taus
            + [self.signaling.tau_camp, self.signaling.tau_calcium]
            + [self.plasticity.tau_eligibility, self._clearance_time()]
        )
        if self.slow_dt > STABILITY_FACTOR * fastest:
            raise ValueError(
                f"slow_dt ({self.slow_dt} s) is too coarse for the fastest time constant "
                f"of the layer ({fastest:.3g} s); use at most {STABILITY_FACTOR * fastest:.3g} s"
            )

    def _clearance_time(self) -> float:
        kinetics = self.manipulation.apply_to(self.release)
        rate = kinetics.v_max / kinetics.k_m + kinetics.k_diffusion
        return 1.0 / rate if rate > 0 else float("inf")


@dataclass(frozen=True, slots=True)
class DopamineTargets:
    """Which neurons and synapses the layer acts on.

    Fields are indexed as ``compartment * 2 + side``, with side 0 = left.

    Attributes
    ----------
    dan_index:
        Network indices of the dopaminergic neurons that innervate a compartment.
    dan_field:
        Field of each of them.
    dan_release_weight:
        Share of its field's release contributed by each neuron; the shares of a
        field sum to one.
    kenyon_index:
        Network indices of Kenyon cells.
    synapse_position:
        Positions in the network weight array of the plastic synapses.
    synapse_field:
        Field of each plastic synapse (the Kenyon cell's hemisphere).
    synapse_presynaptic_slot:
        Position of each synapse's Kenyon cell in ``kenyon_index``.
    """

    dan_index: np.ndarray
    dan_field: np.ndarray
    dan_release_weight: np.ndarray
    kenyon_index: np.ndarray
    synapse_position: np.ndarray
    synapse_field: np.ndarray
    synapse_presynaptic_slot: np.ndarray


class DopamineLayer:
    """Dopaminergic neuromodulation of a running spiking network.

    Parameters
    ----------
    network:
        The spiking network the layer modulates. Only one dopamine layer may be
        attached to a network at a time.
    connectome:
        Connectome the network was built from. Anatomy is read from here, never
        from the network's live weights, which the layer itself changes.
    annotations:
        Cell-type annotations.
    atlas:
        Compartment table; the packaged one by default.
    config:
        Layer configuration.
    """

    def __init__(
        self,
        network: LIFNetwork,
        connectome: Connectome,
        annotations: Annotations,
        atlas: MushroomBodyAtlas | None = None,
        config: DopamineConfig | None = None,
    ) -> None:
        if network.n_neurons != connectome.n_neurons:
            raise ValueError(
                f"network has {network.n_neurons} neurons but the connectome has "
                f"{connectome.n_neurons}; they must be the same circuit"
            )
        self.network = network
        self.atlas = atlas or load_atlas()
        self.config = config or DopamineConfig()

        steps = self.config.slow_dt / network.params.dt
        if abs(steps - round(steps)) > 1e-9:
            raise ValueError(
                f"slow_dt ({self.config.slow_dt} s) must be a multiple of the network dt "
                f"({network.params.dt} s)"
            )
        self._steps_per_slow_step = max(1, int(round(steps)))
        self.n_fields = 2 * len(self.atlas)

        self.targets = self._resolve_targets(connectome, annotations)
        self._build_state()

        network.attach(self, MODULATOR_KIND)
        self._removed_synapses: tuple[np.ndarray, np.ndarray] | None = None
        if self.config.remove_fast_dopamine_synapses:
            self._removed_synapses = network.zero_outgoing(
                connectome.indices_of(
                    [r for r in annotations.dopaminergic() if connectome.contains(r)]
                )
            )
            logger.info(
                "removed %d fast synapses of dopaminergic neurons",
                int(np.count_nonzero(self._removed_synapses[1])),
            )

        self._dan_spike_count = np.zeros(len(self.targets.dan_index), dtype=np.float64)
        self._kenyon_spike_count = np.zeros(len(self.targets.kenyon_index), dtype=np.float64)
        self._dan_slot = _slot_lookup(self.targets.dan_index, network.n_neurons)
        self._kenyon_slot = _slot_lookup(self.targets.kenyon_index, network.n_neurons)
        self._step_in_window = 0

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _build_state(self) -> None:
        manipulation = self.config.manipulation
        dt = self.config.slow_dt
        self.field = DopamineField(
            n_fields=self.n_fields, kinetics=manipulation.apply_to(self.config.release), dt=dt
        )
        self.receptors = {
            spec.name: ReceptorPopulation(
                spec,
                n_cells=self.n_fields,
                dt=dt,
                occupancy_scale=0.0 if spec.name in manipulation.receptor_knockout else 1.0,
            )
            for spec in self.config.receptors
        }
        for name, ratio in manipulation.blocks.items():
            if name in self.receptors:
                self.receptors[name].set_competitive_antagonist(concentration=ratio, k_i=1.0)

        if self.config.autoreceptor is not None:
            spec = self.config.autoreceptor.spec
            occ_scale = 0.0 if spec.name in manipulation.receptor_knockout else 1.0
            self.autoreceptor: AutoreceptorFeedback | None = AutoreceptorFeedback(
                self.config.autoreceptor,
                n_fields=self.n_fields,
                dt=dt,
                occupancy_scale=occ_scale,
            )
            if spec.name in manipulation.blocks:
                ratio = manipulation.blocks[spec.name]
                self.autoreceptor.population.set_competitive_antagonist(
                    concentration=ratio, k_i=1.0
                )
        else:
            self.autoreceptor = None

        self._autoreceptor_gain = np.ones(self.n_fields, dtype=np.float64)

        self.messenger = SecondMessenger(
            n_cells=self.n_fields, params=self.config.signaling, dt=dt
        )
        self.plasticity = SynapticPlasticity(
            weights=self.network.synapse_weight,
            synapse_index=self.targets.synapse_position,
            presynaptic_index=self.targets.synapse_presynaptic_slot,
            compartment_index=self.targets.synapse_field,
            n_compartments=self.n_fields,
            params=self.config.plasticity,
            dt=dt,
        )
        # γ1pedc does not potentiate on backward pairing (Hige et al. 2015)
        self._potentiation_mask = np.repeat(
            np.array([c.potentiates for c in self.atlas.compartments], dtype=np.float64), 2
        )

    def _field_of(self, compartment: str, side: str) -> int:
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        return 2 * self.atlas.index_of(compartment) + SIDES.index(side)

    def _resolve_targets(
        self, connectome: Connectome, annotations: Annotations
    ) -> DopamineTargets:
        index_of = connectome._index  # noqa: SLF001 - read-only use of the cached lookup
        sides = annotations.table.set_index("root_id")["side"]
        cell_types = annotations.table.set_index("root_id")["cell_type"]

        def side_index(root_id: int) -> int | None:
            side = sides.get(root_id)
            return SIDES.index(side) if side in SIDES else None

        dan_index, dan_field = [], []
        for root_id in annotations.dopaminergic():
            root_id = int(root_id)
            compartment = self.atlas.compartment_of_dan(str(cell_types.get(root_id)))
            side = side_index(root_id)
            if compartment is None or side is None or root_id not in index_of:
                continue  # dopaminergic but outside the compartment map
            dan_index.append(index_of[root_id])
            dan_field.append(2 * self.atlas.index_of(compartment) + side)

        kenyon_ids = [int(r) for r in annotations.kenyon_cells() if int(r) in index_of]
        kenyon_index = np.array([index_of[r] for r in kenyon_ids], dtype=np.int64)
        kenyon_side = np.array(
            [-1 if side_index(r) is None else side_index(r) for r in kenyon_ids], dtype=np.int64
        )

        mbon_compartment = np.full(self.network.n_neurons, -1, dtype=np.int64)
        for root_id in annotations.mbons():
            root_id = int(root_id)
            compartment = self.atlas.compartment_of_mbon(str(cell_types.get(root_id)))
            if compartment is not None and root_id in index_of:
                mbon_compartment[index_of[root_id]] = self.atlas.index_of(compartment)

        positions, presynaptic, postsynaptic = self.network.outgoing_synapses(kenyon_index)
        kenyon_slot = _slot_lookup(kenyon_index, self.network.n_neurons)
        slots = kenyon_slot[presynaptic]
        synapse_side = kenyon_side[slots]
        plastic = (mbon_compartment[postsynaptic] >= 0) & (synapse_side >= 0)

        dan_index_array = np.array(dan_index, dtype=np.int64)
        dan_field_array = np.array(dan_field, dtype=np.int64)
        targets = DopamineTargets(
            dan_index=dan_index_array,
            dan_field=dan_field_array,
            dan_release_weight=_release_shares(
                connectome, dan_index_array, dan_field_array, kenyon_index
            ),
            kenyon_index=kenyon_index,
            synapse_position=positions[plastic],
            synapse_field=2 * mbon_compartment[postsynaptic[plastic]] + synapse_side[plastic],
            synapse_presynaptic_slot=slots[plastic],
        )
        logger.info(
            "dopamine layer: %d dopaminergic neurons, %d Kenyon cells, %d plastic synapses",
            len(dan_index_array),
            len(kenyon_index),
            len(targets.synapse_position),
        )
        return targets

    # ------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear concentrations, receptor states, traces and learned weights."""
        self.field.reset()
        for population in self.receptors.values():
            population.reset()
        if self.autoreceptor is not None:
            self.autoreceptor.reset()
        self._autoreceptor_gain[:] = 1.0
        self.messenger.reset()
        self.plasticity.reset()
        self._dan_spike_count[:] = 0.0
        self._kenyon_spike_count[:] = 0.0
        self._step_in_window = 0

    def detach(self) -> None:
        """Restore the fast synapses this layer removed and release the network.

        Learned weights are also returned to their anatomical values.
        """
        self.plasticity.reset()
        if self._removed_synapses is not None:
            self.network.restore_synapses(*self._removed_synapses)
            self._removed_synapses = None
        self.network.detach(MODULATOR_KIND)

    def __call__(self, network: LIFNetwork, step: int, spiking: np.ndarray) -> None:
        """Step callback: accumulate spikes and advance the slow layer."""
        if network is not self.network:
            raise RuntimeError("this dopamine layer is attached to a different network")
        if spiking.size:
            dan_slots = self._dan_slot[spiking]
            np.add.at(self._dan_spike_count, dan_slots[dan_slots >= 0], 1.0)
            kenyon_slots = self._kenyon_slot[spiking]
            np.add.at(self._kenyon_spike_count, kenyon_slots[kenyon_slots >= 0], 1.0)

        self._step_in_window += 1
        if self._step_in_window < self._steps_per_slow_step:
            return
        self._step_in_window = 0
        self._advance_slow_layer()

    def _advance_slow_layer(self) -> None:
        release = np.bincount(
            self.targets.dan_field,
            weights=self._dan_spike_count * self.targets.dan_release_weight,
            minlength=self.n_fields,
        )
        gain = self._autoreceptor_gain if self.autoreceptor is not None else None
        concentration = self.field.step(release, gain=gain)

        if self.autoreceptor is not None:
            self._autoreceptor_gain = self.autoreceptor.step(concentration)

        occupancy = {
            name: population.step(concentration) for name, population in self.receptors.items()
        }
        gs = self._sum_coupling(occupancy, "Gs")
        self.messenger.step(
            gs_occupancy=gs,
            gi_occupancy=self._sum_coupling(occupancy, "Gi"),
            gq_occupancy=self._sum_coupling(occupancy, "Gq"),
        )
        self.plasticity.step(
            presynaptic_spikes=self._kenyon_spike_count,
            gs_activation=np.minimum(gs, 1.0),
            ip3=self.messenger.calcium * self._potentiation_mask,
        )
        self._dan_spike_count[:] = 0.0
        self._kenyon_spike_count[:] = 0.0

    def _sum_coupling(self, occupancy: dict[str, np.ndarray], coupling: str) -> np.ndarray:
        total = np.zeros(self.n_fields, dtype=np.float64)
        for spec in self.config.receptors:
            if spec.coupling == coupling:
                total += occupancy[spec.name]
        return total

    # ------------------------------------------------------------------
    # readout
    # ------------------------------------------------------------------
    def concentration(self, compartment: str, side: str | None = None) -> float:
        """Dopamine concentration in micromolar; the mean of both sides if ``side`` is None."""
        if side is None:
            return float(
                np.mean([self.field.concentration[self._field_of(compartment, s)] for s in SIDES])
            )
        return float(self.field.concentration[self._field_of(compartment, side)])

    def concentrations(self) -> dict[str, float]:
        """Every field's concentration, keyed ``"compartment/side"``."""
        return {
            f"{c.name}/{side}": float(self.field.concentration[self._field_of(c.name, side)])
            for c in self.atlas.compartments
            for side in SIDES
        }

    def weight_change(self, compartment: str, side: str | None = None) -> float:
        """Mean change of the plastic synapses of a compartment, as a fraction.

        Negative means depression. ``nan`` if the compartment has no synapses.
        """
        return self.plasticity.change_of(self._synapse_mask(compartment, side))

    def weight_change_of_neurons(
        self, neurons: np.ndarray, compartment: str, side: str | None = None
    ) -> float:
        """Mean weight change of the synapses made by specific Kenyon cells.

        Parameters
        ----------
        neurons:
            Network indices of Kenyon cells, e.g. the cells of one odour.
        """
        slots = self._kenyon_slot[np.asarray(neurons, dtype=np.int64)]
        slots = slots[slots >= 0]
        from_cells = np.isin(self.targets.synapse_presynaptic_slot, slots)
        return self.plasticity.change_of(self._synapse_mask(compartment, side) & from_cells)

    def _synapse_mask(self, compartment: str, side: str | None) -> np.ndarray:
        sides = SIDES if side is None else (side,)
        fields = [self._field_of(compartment, s) for s in sides]
        return np.isin(self.targets.synapse_field, fields)


def _slot_lookup(indices: np.ndarray, n_neurons: int) -> np.ndarray:
    """Map network index -> position in a compact vector (-1 if absent)."""
    lookup = np.full(n_neurons, -1, dtype=np.int64)
    lookup[indices] = np.arange(len(indices), dtype=np.int64)
    return lookup


def _release_shares(
    connectome: Connectome,
    dan_index: np.ndarray,
    dan_field: np.ndarray,
    kenyon_index: np.ndarray,
) -> np.ndarray:
    """Each neuron's share of its field's synapses onto Kenyon cells.

    Read from the connectome rather than the running network, whose weights the
    layer changes. Within a field the shares sum to one, so the calibration of
    release per spike applies to a field rather than to a single neuron: two
    neurons innervate γ1pedc and forty innervate γ5, and without this one large
    neuron would flood its compartment simply for being large.
    """
    if dan_index.size == 0:
        return np.zeros(0, dtype=np.float64)
    onto_kenyon = abs(connectome.weights[kenyon_index][:, dan_index])
    counts = np.asarray(onto_kenyon.sum(axis=0)).ravel()

    shares = np.zeros(len(dan_index), dtype=np.float64)
    for field_id in np.unique(dan_field):
        members = dan_field == field_id
        total = counts[members].sum()
        shares[members] = counts[members] / total if total > 0 else 1.0 / members.sum()
    return shares
