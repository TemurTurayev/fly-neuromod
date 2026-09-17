"""Associative conditioning in the mushroom body.

The protocol follows the electrophysiological experiment that defines the rule
we implement: present an odour, pair it with activation of a compartment's
dopaminergic neuron, and measure what happened to the odour response of that
compartment's output neuron. A second, unpaired odour is the internal control.

Reference experiment: Hige et al. (2015), *Neuron* 88:985, doi:10.1016/j.neuron.2015.11.003.
Pairing odour with PPL1-γ1pedc activation reduced the odour-evoked response of
MBON-γ1pedc>α/β by 90 ± 4%, while the unpaired odour lost about 25% — the latter
because the two odours share Kenyon cells, not because the rule is unspecific.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..engine.lif import LIFNetwork
from ..engine.params import LIFParams, PoissonDrive
from ..neuromod.dopamine import DopamineConfig, DopamineLayer
from .mushroom_body import MushroomBody, sparse_odour

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ConditioningProtocol:
    """Timing and strength of a pairing experiment.

    Attributes
    ----------
    odour_duration:
        Length of each odour presentation in seconds.
    odour_rate:
        Drive rate applied to the Kenyon cells of the odour, in hertz.
    dopamine_rate:
        Drive rate applied to the dopaminergic neurons during pairing.
    dopamine_onset:
        Delay from odour onset to dopamine onset, in seconds. Positive values
        are forward pairing (odour first), negative values backward pairing.
    dopamine_duration:
        Length of the dopaminergic activation in seconds.
    n_pairings:
        Number of pairing trials.
    rest:
        Pause after pairing before testing, in seconds; lets dopamine and cAMP
        return to baseline so the test itself writes nothing. cAMP relaxes with
        a time constant of several seconds, so this has to be long: with a short
        pause the first test odour is still bathed in cAMP and gets trained too,
        which shows up as a spurious loss for the control odour.
    odour_fraction:
        Fraction of Kenyon cells each odour activates.
    odour_overlap:
        Fraction of the trained odour's Kenyon cells that the control odour
        shares. Real odour pairs overlap; this is what produces the partial
        depression of the control odour.
    """

    odour_duration: float = 1.0
    odour_rate: float = 150.0
    dopamine_rate: float = 20.0
    dopamine_onset: float = 0.2
    dopamine_duration: float = 1.0
    n_pairings: int = 1
    rest: float = 25.0
    odour_fraction: float = 0.1
    odour_overlap: float = 0.2

    def __post_init__(self) -> None:
        if self.odour_duration <= 0 or self.dopamine_duration <= 0:
            raise ValueError("durations must be positive")
        if self.n_pairings < 1:
            raise ValueError("at least one pairing is required")
        if not 0 <= self.odour_overlap <= 1:
            raise ValueError("odour_overlap must be a fraction")


@dataclass(frozen=True, slots=True)
class ConditioningResult:
    """Output neuron responses before and after pairing, in hertz."""

    trained_before: float
    trained_after: float
    control_before: float
    control_after: float
    weight_change: float
    compartment: str

    @property
    def trained_depression(self) -> float:
        """Fractional loss of the trained odour response (1.0 = abolished)."""
        return _fractional_loss(self.trained_before, self.trained_after)

    @property
    def control_depression(self) -> float:
        return _fractional_loss(self.control_before, self.control_after)

    def summary(self) -> str:
        return (
            f"{self.compartment}: trained {self.trained_before:.1f} -> {self.trained_after:.1f} Hz "
            f"({100 * self.trained_depression:.0f}% loss), "
            f"control {self.control_before:.1f} -> {self.control_after:.1f} Hz "
            f"({100 * self.control_depression:.0f}% loss), "
            f"synaptic weight {100 * self.weight_change:+.0f}%"
        )


def _fractional_loss(before: float, after: float) -> float:
    if before <= 0:
        return 0.0
    return float((before - after) / before)


def run_conditioning(
    mushroom_body: MushroomBody,
    compartment: str = "gamma1pedc",
    readout_type: str = "MBON11",
    dan_type: str = "PPL101",
    protocol: ConditioningProtocol | None = None,
    config: DopamineConfig | None = None,
    params: LIFParams | None = None,
    seed: int = 0,
) -> ConditioningResult:
    """Run one pairing experiment and measure the output neuron response.

    Parameters
    ----------
    mushroom_body:
        Subnetwork produced by :func:`~.mushroom_body.extract_mushroom_body`.
    compartment:
        Compartment being trained, used for the weight readout.
    readout_type:
        Cell type of the output neuron whose odour response is measured.
    dan_type:
        Cell type of the dopaminergic neurons used as the teaching signal.
    protocol:
        Timing of the experiment.
    config:
        Dopamine layer configuration, e.g. carrying a drug or a knockout.
    params:
        Fast-layer parameters.
    seed:
        Seed for the odour identity and the Poisson drives.
    """
    protocol = protocol or ConditioningProtocol()
    rng = np.random.default_rng(seed)
    network = LIFNetwork(
        mushroom_body.connectome.weights, params or LIFParams(), rng=np.random.default_rng(seed)
    )
    layer = DopamineLayer(
        network,
        mushroom_body.connectome,
        mushroom_body.annotations,
        config=config or DopamineConfig(),
    )

    kenyon = mushroom_body.kenyon_indices()
    trained_odour = sparse_odour(kenyon, protocol.odour_fraction, rng)
    control_odour = _overlapping_odour(kenyon, trained_odour, protocol, rng)
    readout = mushroom_body.indices_of_type(readout_type)
    dans = mushroom_body.indices_of_type(dan_type)
    if readout.size == 0:
        raise ValueError(f"no {readout_type} in this subnetwork")
    if dans.size == 0:
        raise ValueError(f"no {dan_type} in this subnetwork")

    trained_before = _present_odour(network, layer, trained_odour, readout, protocol)
    control_before = _present_odour(network, layer, control_odour, readout, protocol)

    for _ in range(protocol.n_pairings):
        _pair(network, layer, trained_odour, dans, protocol)
        _rest(network, layer, protocol.rest)

    trained_after = _present_odour(network, layer, trained_odour, readout, protocol)
    _rest(network, layer, protocol.rest)
    control_after = _present_odour(network, layer, control_odour, readout, protocol)

    result = ConditioningResult(
        trained_before=trained_before,
        trained_after=trained_after,
        control_before=control_before,
        control_after=control_after,
        weight_change=layer.weight_change(compartment),
        compartment=compartment,
    )
    logger.info(result.summary())
    return result


def _overlapping_odour(
    kenyon: np.ndarray,
    trained: np.ndarray,
    protocol: ConditioningProtocol,
    rng: np.random.Generator,
) -> np.ndarray:
    """A second odour sharing a fraction of the trained odour's Kenyon cells."""
    n_shared = int(round(protocol.odour_overlap * len(trained)))
    shared = rng.choice(trained, size=n_shared, replace=False) if n_shared else np.zeros(0, int)
    remaining = np.setdiff1d(kenyon, trained)
    fresh = rng.choice(remaining, size=len(trained) - n_shared, replace=False)
    return np.concatenate([shared, fresh])


def _present_odour(
    network: LIFNetwork,
    layer: DopamineLayer,
    odour: np.ndarray,
    readout: np.ndarray,
    protocol: ConditioningProtocol,
) -> float:
    """Present an odour and return the mean firing rate of the readout neurons."""
    network.reset_state()
    network.set_poisson_drives({int(i): PoissonDrive(rate=protocol.odour_rate) for i in odour})
    trains = network.run(protocol.odour_duration, callbacks=[layer])
    rates = trains.rates()
    return float(rates[readout].mean())


def _pair(
    network: LIFNetwork,
    layer: DopamineLayer,
    odour: np.ndarray,
    dans: np.ndarray,
    protocol: ConditioningProtocol,
) -> None:
    """One pairing trial: odour, then dopamine after ``dopamine_onset``."""
    network.reset_state()
    odour_drives = {int(i): PoissonDrive(rate=protocol.odour_rate) for i in odour}
    dan_drives = {int(i): PoissonDrive(rate=protocol.dopamine_rate) for i in dans}

    if protocol.dopamine_onset >= 0:
        network.set_poisson_drives(odour_drives)
        network.run(min(protocol.dopamine_onset, protocol.odour_duration), callbacks=[layer])
        network.set_poisson_drives({**odour_drives, **dan_drives})
        overlap = max(protocol.odour_duration - protocol.dopamine_onset, 0.0)
        if overlap > 0:
            network.run(min(overlap, protocol.dopamine_duration), callbacks=[layer])
        remaining = protocol.dopamine_duration - overlap
    else:
        network.set_poisson_drives(dan_drives)
        network.run(-protocol.dopamine_onset, callbacks=[layer])
        network.set_poisson_drives({**odour_drives, **dan_drives})
        network.run(min(protocol.odour_duration, protocol.dopamine_duration), callbacks=[layer])
        remaining = protocol.dopamine_duration - min(
            protocol.odour_duration, protocol.dopamine_duration
        )

    if remaining > 0:
        network.set_poisson_drives(dan_drives)
        network.run(remaining, callbacks=[layer])


def _rest(network: LIFNetwork, layer: DopamineLayer, duration: float) -> None:
    """Let dopamine and the second messengers decay with no input."""
    if duration <= 0:
        return
    network.set_poisson_drives({})
    network.run(duration, callbacks=[layer], record=False)
