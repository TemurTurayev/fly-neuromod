"""The dopamine layer: from dopaminergic spikes to changed synapses.

The layer runs alongside the spiking network as a step callback and closes this
loop, once per slow step:

1. **Sources.** Spikes of curated dopaminergic neurons are pooled per mushroom
   body compartment, weighted by how many release sites each neuron actually has
   there in the connectome.
2. **Volume.** The pooled release raises the extracellular dopamine
   concentration of that compartment; the transporter clears it.
3. **Receptors.** Dop1R1 (Gs), Dop1R2 (Gq) and Dop2R (Gi) bind dopamine with
   their own affinities and kinetics.
4. **Signalling.** Receptor occupancy drives cAMP and calcium in the Kenyon cell
   terminals of the compartment.
5. **Synapses.** cAMP with a recent Kenyon cell spike depresses that cell's
   output synapse; calcium with a spike arriving afterwards potentiates it.

The fast network never sees dopamine as current. Dopaminergic connections are
removed from the spiking layer by default, because no ionotropic dopamine
receptor is known in *Drosophila* — every documented effect of dopamine in the
fly runs through G-protein-coupled receptors on a timescale of seconds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..data.annotations import Annotations
from ..data.connectome import Connectome
from .constants import (
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


@dataclass(frozen=True, slots=True)
class DopamineConfig:
    """Configuration of the dopamine layer.

    Attributes
    ----------
    slow_dt:
        Time step of the modulator in seconds. Dopamine acts far more slowly
        than spikes, so the layer is integrated at a coarser step than the
        network (1 ms by default against 0.1 ms).
    release:
        Release and clearance kinetics of a compartment.
    receptors:
        Receptor types present on Kenyon cell terminals.
    signaling:
        cAMP and calcium kinetics.
    plasticity:
        Learning-rule constants.
    remove_fast_dopamine_synapses:
        Zero the fast synaptic weights of dopaminergic neurons, since dopamine
        acts through GPCRs rather than ligand-gated channels. Note that some
        dopaminergic neurons genuinely co-release fast transmitters (GABA in PAM
        neurons, glutamate in PPL1 neurons; Yamazaki et al. 2023), which this
        switch also removes - set it to ``False`` to keep them.
    manipulation:
        A drug or mutant from :mod:`flyneuromod.neuromod.pharmacology`.
    """

    slow_dt: float = 1e-3
    release: ReleaseKinetics = MB_COMPARTMENT_RELEASE
    receptors: tuple[ReceptorSpec, ...] = DOPAMINE_RECEPTORS
    signaling: SignalingParams = KC_SIGNALING
    plasticity: PlasticityParams = KC_TO_MBON_PLASTICITY
    remove_fast_dopamine_synapses: bool = True
    manipulation: Manipulation = CONTROL

    def __post_init__(self) -> None:
        if self.slow_dt <= 0:
            raise ValueError("slow_dt must be positive")
        couplings = {r.coupling for r in self.receptors}
        if "Gs" not in couplings:
            raise ValueError("a Gs-coupled receptor is required to drive cAMP")


@dataclass(frozen=True, slots=True)
class DopamineTargets:
    """Which neurons and synapses the layer acts on.

    Attributes
    ----------
    dan_index:
        Network indices of the dopaminergic neurons that innervate a compartment.
    dan_compartment:
        Compartment index for each of them.
    dan_release_weight:
        Share of its compartment's release contributed by each neuron,
        proportional to its synapses onto Kenyon cells there; the shares of a
        compartment sum to one.
    kenyon_index:
        Network indices of Kenyon cells.
    synapse_position:
        Positions in the network weight array of the plastic Kenyon cell to
        output neuron synapses.
    synapse_compartment:
        Compartment index of each plastic synapse.
    """

    dan_index: np.ndarray
    dan_compartment: np.ndarray
    dan_release_weight: np.ndarray
    kenyon_index: np.ndarray
    synapse_position: np.ndarray
    synapse_compartment: np.ndarray
    synapse_presynaptic_slot: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))


class DopamineLayer:
    """Dopaminergic neuromodulation of a running spiking network.

    Parameters
    ----------
    network:
        The spiking network the layer modulates.
    connectome:
        Connectome the network was built from, used to map root ids to indices.
    annotations:
        Cell-type annotations, used to find dopaminergic neurons, Kenyon cells
        and output neurons.
    atlas:
        Compartment table; the packaged one by default.
    config:
        Layer configuration.

    Examples
    --------
    >>> layer = DopamineLayer(network, connectome, annotations)  # doctest: +SKIP
    >>> network.run(5.0, callbacks=[layer])  # doctest: +SKIP
    >>> layer.concentrations()["gamma1pedc"]  # doctest: +SKIP
    """

    def __init__(
        self,
        network: LIFNetwork,
        connectome: Connectome,
        annotations: Annotations,
        atlas: MushroomBodyAtlas | None = None,
        config: DopamineConfig | None = None,
    ) -> None:
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

        self.targets = self._resolve_targets(connectome, annotations)
        n_compartments = len(self.atlas)

        manipulation = self.config.manipulation
        self.field = DopamineField(
            n_fields=n_compartments,
            kinetics=manipulation.apply_to(self.config.release),
            dt=self.config.slow_dt,
        )
        self.receptors = {
            spec.name: ReceptorPopulation(
                spec,
                n_cells=n_compartments,
                dt=self.config.slow_dt,
                occupancy_scale=0.0 if spec.name in manipulation.receptor_knockout else 1.0,
            )
            for spec in self.config.receptors
        }
        for name, ratio in manipulation.receptor_block.items():
            if name in self.receptors:
                self.receptors[name].set_competitive_antagonist(concentration=ratio, k_i=1.0)

        self.messenger = SecondMessenger(
            n_cells=n_compartments, params=self.config.signaling, dt=self.config.slow_dt
        )
        self.plasticity = SynapticPlasticity(
            weights=network.synapse_weight,
            synapse_index=self.targets.synapse_position,
            presynaptic_index=self.targets.synapse_presynaptic_slot,
            compartment_index=self.targets.synapse_compartment,
            params=self.config.plasticity,
            dt=self.config.slow_dt,
        )
        # compartments where backward pairing was shown not to potentiate
        self._potentiation_mask = np.array(
            [c.potentiates for c in self.atlas.compartments], dtype=np.float64
        )

        if self.config.remove_fast_dopamine_synapses:
            self._remove_fast_dopamine_synapses(connectome, annotations)

        self._dan_spike_count = np.zeros(len(self.targets.dan_index), dtype=np.float64)
        self._kenyon_spike_count = np.zeros(len(self.targets.kenyon_index), dtype=np.float64)
        self._dan_slot = self._slot_lookup(self.targets.dan_index, network.n_neurons)
        self._kenyon_slot = self._slot_lookup(self.targets.kenyon_index, network.n_neurons)
        self._step_in_window = 0

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _slot_lookup(indices: np.ndarray, n_neurons: int) -> np.ndarray:
        """Map network index -> position in a compact vector (-1 if absent)."""
        lookup = np.full(n_neurons, -1, dtype=np.int64)
        lookup[indices] = np.arange(len(indices), dtype=np.int64)
        return lookup

    def _resolve_targets(
        self, connectome: Connectome, annotations: Annotations
    ) -> DopamineTargets:
        """Find dopaminergic neurons, Kenyon cells and the plastic synapses."""
        index_of = connectome.index_map()

        dan_root_ids = annotations.dopaminergic()
        dan_types = annotations.cell_types_of(dan_root_ids)
        dan_index, dan_compartment = [], []
        for root_id, cell_type in zip(dan_root_ids, dan_types, strict=True):
            compartment = self.atlas.compartment_of_dan(str(cell_type))
            if compartment is None or int(root_id) not in index_of:
                continue  # dopaminergic but outside the compartment map
            dan_index.append(index_of[int(root_id)])
            dan_compartment.append(self.atlas.index_of(compartment))

        kenyon_root_ids = [r for r in annotations.kenyon_cells() if int(r) in index_of]
        kenyon_index = np.array([index_of[int(r)] for r in kenyon_root_ids], dtype=np.int64)

        mbon_root_ids = annotations.mbons()
        mbon_types = annotations.cell_types_of(mbon_root_ids)
        mbon_compartment = np.full(self.network.n_neurons, -1, dtype=np.int64)
        for root_id, cell_type in zip(mbon_root_ids, mbon_types, strict=True):
            compartment = self.atlas.compartment_of_mbon(str(cell_type))
            if compartment is None or int(root_id) not in index_of:
                continue
            mbon_compartment[index_of[int(root_id)]] = self.atlas.index_of(compartment)

        positions, presynaptic, postsynaptic = self.network.outgoing_synapses(kenyon_index)
        is_plastic = mbon_compartment[postsynaptic] >= 0
        synapse_position = positions[is_plastic]
        synapse_compartment = mbon_compartment[postsynaptic[is_plastic]]

        kenyon_slot = self._slot_lookup(kenyon_index, self.network.n_neurons)
        synapse_presynaptic_slot = kenyon_slot[presynaptic[is_plastic]]

        dan_index_array = np.array(dan_index, dtype=np.int64)
        dan_compartment_array = np.array(dan_compartment, dtype=np.int64)
        release_weight = self._release_weights(
            dan_index_array, dan_compartment_array, kenyon_index
        )

        logger.info(
            "dopamine layer: %d dopaminergic neurons, %d Kenyon cells, %d plastic synapses",
            len(dan_index_array),
            len(kenyon_index),
            len(synapse_position),
        )
        return DopamineTargets(
            dan_index=dan_index_array,
            dan_compartment=dan_compartment_array,
            dan_release_weight=release_weight,
            kenyon_index=kenyon_index,
            synapse_position=synapse_position,
            synapse_compartment=synapse_compartment,
            synapse_presynaptic_slot=synapse_presynaptic_slot,
        )

    def _release_weights(
        self, dan_index: np.ndarray, dan_compartment: np.ndarray, kenyon_index: np.ndarray
    ) -> np.ndarray:
        """Share of a compartment's dopamine release contributed by each neuron.

        Within a compartment the weights are the neurons' shares of the synapses
        onto Kenyon cells, so a neuron with twice as many release sites releases
        twice as much, and they sum to one. Normalising *per compartment* rather
        than across the brain matters: the concentration that was measured is a
        compartment-level quantity, and compartments differ enormously in how
        many dopaminergic neurons they have - two for γ1pedc against forty for
        γ5. Without this, one PPL1 neuron would flood its compartment simply for
        being large, and the calibration against the measured 0.3-0.5 µM would
        apply to no compartment at all.
        """
        if dan_index.size == 0:
            return np.zeros(0, dtype=np.float64)
        is_kenyon = np.zeros(self.network.n_neurons, dtype=bool)
        is_kenyon[kenyon_index] = True
        positions, presynaptic, postsynaptic = self.network.outgoing_synapses(dan_index)
        onto_kenyon = is_kenyon[postsynaptic]
        slot = self._slot_lookup(dan_index, self.network.n_neurons)
        counts = np.bincount(
            slot[presynaptic[onto_kenyon]],
            weights=np.abs(self.network.synapse_weight[positions[onto_kenyon]]),
            minlength=len(dan_index),
        )

        weights = np.zeros(len(dan_index), dtype=np.float64)
        for compartment in np.unique(dan_compartment):
            members = dan_compartment == compartment
            total = counts[members].sum()
            if total > 0:
                weights[members] = counts[members] / total
            else:  # no annotated synapses onto Kenyon cells: share equally
                weights[members] = 1.0 / members.sum()
        return weights

    def _remove_fast_dopamine_synapses(
        self, connectome: Connectome, annotations: Annotations
    ) -> None:
        """Zero the fast weights of curated dopaminergic neurons."""
        index_of = connectome.index_map()
        indices = np.array(
            [index_of[int(r)] for r in annotations.dopaminergic() if int(r) in index_of],
            dtype=np.int64,
        )
        if indices.size == 0:
            return
        positions = self.network._flat_positions(indices)  # noqa: SLF001 - same package
        removed = int(np.count_nonzero(self.network.synapse_weight[positions]))
        self.network.synapse_weight[positions] = 0.0
        logger.info("removed %d fast synapses of dopaminergic neurons", removed)

    # ------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear concentrations, receptor states and learned weights."""
        self.field.reset()
        for population in self.receptors.values():
            population.reset()
        self.messenger.reset()
        self.plasticity.reset()
        self._dan_spike_count[:] = 0.0
        self._kenyon_spike_count[:] = 0.0
        self._step_in_window = 0

    def __call__(self, network: LIFNetwork, step: int, spiking: np.ndarray) -> None:
        """Step callback: accumulate spikes and advance the slow layer."""
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
            self.targets.dan_compartment,
            weights=self._dan_spike_count * self.targets.dan_release_weight,
            minlength=len(self.atlas),
        )
        concentration = self.field.step(release)

        occupancy = {
            name: population.step(concentration) for name, population in self.receptors.items()
        }
        self.messenger.step(
            gs_occupancy=self._sum_coupling(occupancy, "Gs"),
            gi_occupancy=self._sum_coupling(occupancy, "Gi"),
            gq_occupancy=self._sum_coupling(occupancy, "Gq") * self._potentiation_mask,
        )
        self.plasticity.step(
            presynaptic_spikes=self._kenyon_spike_count,
            camp=self.messenger.camp,
            calcium=self.messenger.calcium,
        )

        self._dan_spike_count[:] = 0.0
        self._kenyon_spike_count[:] = 0.0

    def _sum_coupling(self, occupancy: dict[str, np.ndarray], coupling: str) -> np.ndarray:
        """Total occupancy of receptors with a given G-protein coupling."""
        total = np.zeros(len(self.atlas), dtype=np.float64)
        for spec in self.config.receptors:
            if spec.coupling == coupling:
                total += occupancy[spec.name]
        return total

    # ------------------------------------------------------------------
    # readout
    # ------------------------------------------------------------------
    def concentrations(self) -> dict[str, float]:
        """Dopamine concentration per compartment in micromolar."""
        return dict(zip(self.atlas.names, self.field.concentration.tolist(), strict=True))

    def weight_change(self, compartment: str) -> float:
        """Mean change of the plastic synapses of one compartment, as a fraction.

        Negative means depression. This is the quantity that pairing experiments
        report for Kenyon cell to output neuron synapses.
        """
        mask = self.targets.synapse_compartment == self.atlas.index_of(compartment)
        return self.plasticity.depression_of(mask)

    def weight_change_of_cells(self, kenyon_slots: np.ndarray, compartment: str) -> float:
        """Mean weight change restricted to synapses of specific Kenyon cells.

        This is how odour specificity is measured: only the cells that carried
        the trained odour should have changed.
        """
        in_compartment = self.targets.synapse_compartment == self.atlas.index_of(compartment)
        from_cells = np.isin(self.targets.synapse_presynaptic_slot, kenyon_slots)
        return self.plasticity.depression_of(in_compartment & from_cells)
