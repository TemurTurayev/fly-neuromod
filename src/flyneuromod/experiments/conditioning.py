"""Associative conditioning in the mushroom body.

Present an odour, pair it with activation of a compartment's dopaminergic
neurons, and measure what happened to that compartment's output neuron and to
the trained synapses. A second, partly overlapping odour is the control.

Reference experiment: Hige et al. (2015), *Neuron* 88:985,
doi:10.1016/j.neuron.2015.11.003. One pairing of an odour with PPL1-γ1pedc
activation removed about 90% of the Kenyon cell input to MBON-γ1pedc>α/β, while
the control odour lost about 25% — because odours share Kenyon cells, not
because the rule is unspecific.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..engine.lif import LIFNetwork
from ..engine.params import LIFParams, PoissonDrive
from ..neuromod.dopamine import DopamineConfig, DopamineLayer
from ..neuromod.mb_atlas import load_atlas
from .mushroom_body import MushroomBody, sparse_odour
from .protocol import ConditioningProtocol, pairing_timeline

logger = logging.getLogger(__name__)

CALIBRATION_WINDOW = 0.5
"""Seconds of spontaneous activity per step of the operating-point fit."""
CALIBRATION_ITERATIONS = 12
"""Bisection steps; 12 halvings resolve the drive to 0.02% of its bracket."""
MAX_TONIC_DRIVE = 64.0
"""Upper bound on the tonic drive searched, in volts per second."""


@dataclass(frozen=True, slots=True)
class ConditioningResult:
    """Responses before and after pairing (hertz) and the synaptic change.

    Response losses are ``nan`` when the odour evoked nothing to lose, so that a
    failed measurement cannot pass for an absence of learning.
    """

    baseline: float
    trained_before: float
    trained_after: float
    control_before: float
    control_after: float
    weight_change: float
    trained_weight_change: float
    compartment: str

    @property
    def trained_depression(self) -> float:
        """Fractional loss of the odour-evoked (above-baseline) response."""
        return _fractional_loss(
            self.trained_before - self.baseline, self.trained_after - self.baseline
        )

    @property
    def control_depression(self) -> float:
        return _fractional_loss(
            self.control_before - self.baseline, self.control_after - self.baseline
        )

    def summary(self) -> str:
        return (
            f"{self.compartment} (baseline {self.baseline:.0f} Hz): "
            f"trained {self.trained_before:.1f} -> {self.trained_after:.1f} Hz "
            f"({_percent(self.trained_depression)} loss), "
            f"control {self.control_before:.1f} -> {self.control_after:.1f} Hz "
            f"({_percent(self.control_depression)} loss), "
            f"trained synapses {_percent(self.trained_weight_change, signed=True)}, "
            f"compartment {_percent(self.weight_change, signed=True)}"
        )


def _percent(value: float, signed: bool = False) -> str:
    if np.isnan(value):
        return "n/a"
    return f"{100 * value:+.0f}%" if signed else f"{100 * value:.0f}%"


def _fractional_loss(before: float, after: float) -> float:
    if before <= 0:
        logger.warning("the odour evoked no response above baseline; loss is undefined")
        return float("nan")
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
    """Run one conditioning experiment.

    Parameters
    ----------
    mushroom_body:
        Subnetwork from :func:`~.mushroom_body.extract_mushroom_body`.
    compartment:
        Compartment being trained.
    readout_type:
        Output neuron type measured; must read out ``compartment``.
    dan_type:
        Dopaminergic type used as the teaching signal; must innervate
        ``compartment``.
    protocol:
        Timing and strength of the experiment.
    config:
        Dopamine layer configuration, e.g. carrying a drug or a mutant.
    params:
        Fast-layer parameters.
    seed:
        Seed for the odours and the Poisson drives.

    Raises
    ------
    ValueError
        Before any simulation, if the cell types do not belong to the compartment
        or are missing from the subnetwork.
    """
    protocol = protocol or ConditioningProtocol()
    _check_circuit(mushroom_body, compartment, readout_type, dan_type)

    rng = np.random.default_rng(seed)
    network = LIFNetwork(
        mushroom_body.connectome.weights, params or LIFParams(), rng=np.random.default_rng(seed)
    )
    layer = DopamineLayer(
        network, mushroom_body.connectome, mushroom_body.annotations, config=config
    )
    kenyon = mushroom_body.kenyon_indices()
    trained_odour = sparse_odour(kenyon, protocol.odour_fraction, rng)
    control_odour = _overlapping_odour(kenyon, trained_odour, protocol, rng)
    readout = mushroom_body.indices_of_type(readout_type)
    dans = mushroom_body.indices_of_type(dan_type)

    baseline = _restore_operating_point(network, layer, readout, protocol)

    trained_before = _present_odour(network, layer, trained_odour, readout, protocol)
    _rest(network, layer, protocol.rest)
    control_before = _present_odour(network, layer, control_odour, readout, protocol)
    _rest(network, layer, protocol.rest)

    for _ in range(protocol.n_pairings):
        _pair(network, layer, trained_odour, dans, protocol)
        _rest(network, layer, protocol.rest)

    trained_after = _present_odour(network, layer, trained_odour, readout, protocol)
    _rest(network, layer, protocol.rest)
    control_after = _present_odour(network, layer, control_odour, readout, protocol)

    result = ConditioningResult(
        baseline=baseline,
        trained_before=trained_before,
        trained_after=trained_after,
        control_before=control_before,
        control_after=control_after,
        weight_change=layer.weight_change(compartment),
        trained_weight_change=layer.weight_change_of_neurons(trained_odour, compartment),
        compartment=compartment,
    )
    logger.info(result.summary())
    return result


def _check_circuit(
    mushroom_body: MushroomBody, compartment: str, readout_type: str, dan_type: str
) -> None:
    """Refuse inconsistent experiments before spending minutes simulating them."""
    atlas = load_atlas()
    atlas.index_of(compartment)  # raises KeyError listing the valid names
    if atlas.compartment_of_dan(dan_type) != compartment:
        raise ValueError(
            f"{dan_type} does not innervate {compartment} "
            f"(it innervates {atlas.compartment_of_dan(dan_type)})"
        )
    if atlas.compartment_of_mbon(readout_type) != compartment:
        raise ValueError(
            f"{readout_type} does not read out {compartment} "
            f"(it reads out {atlas.compartment_of_mbon(readout_type)})"
        )
    for cell_type in (readout_type, dan_type):
        if mushroom_body.indices_of_type(cell_type).size == 0:
            raise ValueError(f"no {cell_type} in this subnetwork")


def _restore_operating_point(
    network: LIFNetwork,
    layer: DopamineLayer,
    readout: np.ndarray,
    protocol: ConditioningProtocol,
) -> float:
    """Fit a tonic drive so the readout fires at the target spontaneous rate.

    Returns the rate reached, which responses are measured relative to.
    """
    if protocol.readout_baseline_rate <= 0:
        return 0.0
    target = protocol.readout_baseline_rate

    def rate_at(drive: float) -> float:
        network.set_tonic_drive({int(i): drive for i in readout})
        network.reset_state()
        network.set_poisson_drives({})
        trains = network.run(CALIBRATION_WINDOW, callbacks=[layer])
        return float(trains.rates()[readout].mean())

    low, high = 0.0, 1.0
    while rate_at(high) < target and high < MAX_TONIC_DRIVE:
        high *= 2
    for _ in range(CALIBRATION_ITERATIONS):
        middle = 0.5 * (low + high)
        if rate_at(middle) < target:
            low = middle
        else:
            high = middle

    network.set_tonic_drive({int(i): high for i in readout})
    reached = rate_at(high)
    layer.reset()  # the fit must leave no trace in the dopamine layer
    if abs(reached - target) > 0.2 * target:
        logger.warning(
            "operating point not reached: %.1f Hz instead of %.1f Hz", reached, target
        )
    logger.info("readout tonic drive %.3f V/s gives %.1f Hz spontaneous", high, reached)
    return reached


def _overlapping_odour(
    kenyon: np.ndarray,
    trained: np.ndarray,
    protocol: ConditioningProtocol,
    rng: np.random.Generator,
) -> np.ndarray:
    """A second odour sharing a fraction of the trained odour's Kenyon cells."""
    n_shared = int(round(protocol.odour_overlap * len(trained)))
    shared = rng.choice(trained, size=n_shared, replace=False)
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
    trains = network.run(protocol.test_duration, callbacks=[layer])
    return float(trains.rates()[readout].mean())


def _pair(
    network: LIFNetwork,
    layer: DopamineLayer,
    odour: np.ndarray,
    dans: np.ndarray,
    protocol: ConditioningProtocol,
) -> None:
    """One pairing trial, segment by segment, exactly as the timeline says."""
    network.reset_state()
    odour_drives = {int(i): PoissonDrive(rate=protocol.odour_rate) for i in odour}
    dan_drives = {int(i): PoissonDrive(rate=protocol.dopamine_rate) for i in dans}
    for segment in pairing_timeline(protocol):
        drives: dict[int, PoissonDrive] = {}
        if segment.odour:
            drives |= odour_drives
        if segment.dopamine:
            drives |= dan_drives
        network.set_poisson_drives(drives)
        network.run(segment.duration, callbacks=[layer], record=False)


def _rest(network: LIFNetwork, layer: DopamineLayer, duration: float) -> None:
    """Let dopamine and the second messengers decay with no input."""
    if duration <= 0:
        return
    network.set_poisson_drives({})
    network.run(duration, callbacks=[layer], record=False)
