"""The timing curve: does the sign of learning follow the order of events?

This is the experiment that separates a dopamine layer from a reward signal.
Handler et al. (2019), *Cell* 178:60, varied the interval between odour and
dopaminergic activation in the γ4 and γ5 compartments and found that the sign of
the change flips: with the odour first the synapse is depressed, with dopamine
first it is potentiated, and beyond a few seconds nothing happens at all.

A model that only depresses cannot produce this curve, however well it is tuned.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from ..neuromod.dopamine import DopamineConfig
from .conditioning import run_conditioning
from .mushroom_body import MushroomBody
from .protocol import HANDLER_2019, ConditioningProtocol

logger = logging.getLogger(__name__)

DEFAULT_INTERVALS = (-1.2, -0.5, 0.0, 0.5, 1.0, 6.0)
"""Intervals in seconds; negative means dopamine before odour (backward pairing)."""


@dataclass(frozen=True, slots=True)
class TimingPoint:
    """One interval and the weight change it produced."""

    interval: float
    weight_change: float
    response_change: float


def timing_curve(
    mushroom_body: MushroomBody,
    compartment: str = "gamma5",
    readout_type: str = "MBON01",
    dan_type: str = "PAM01",
    intervals: Sequence[float] = DEFAULT_INTERVALS,
    protocol: ConditioningProtocol | None = None,
    config: DopamineConfig | None = None,
    seed: int = 0,
) -> list[TimingPoint]:
    """Measure the change in synaptic weight as a function of pairing interval.

    Parameters
    ----------
    mushroom_body:
        The subnetwork to run in.
    compartment, readout_type, dan_type:
        Which compartment to train, whose output neuron to read, and which
        dopaminergic neurons to drive. The defaults are γ5, where the sign flip
        was measured.
    intervals:
        Delays from odour onset to dopamine onset, in seconds.
    protocol:
        Base protocol; its ``dopamine_onset`` is replaced by each interval.
        Defaults to the 2 s odour and 1 s dopamine of Handler et al. (2019).
    config:
        Dopamine layer configuration.
    seed:
        Seed, shared by all points so that the odour is the same throughout.

    Returns
    -------
    list of TimingPoint
        One entry per interval, in the order given.
    """
    base = protocol or HANDLER_2019
    points: list[TimingPoint] = []
    for interval in intervals:
        result = run_conditioning(
            mushroom_body,
            compartment=compartment,
            readout_type=readout_type,
            dan_type=dan_type,
            protocol=base.evolve(dopamine_onset=interval),
            config=config,
            seed=seed,
        )
        points.append(
            TimingPoint(
                interval=interval,
                weight_change=result.trained_weight_change,
                response_change=-result.trained_depression,
            )
        )
        logger.info(
            "interval %+.2f s: weight %+.1f%%, response %+.1f%%",
            interval,
            100 * points[-1].weight_change,
            100 * points[-1].response_change,
        )
    return points


def describe(points: Sequence[TimingPoint]) -> str:
    """One line per interval, for printing a curve in the terminal."""
    lines = ["interval (s)   weight change   response change"]
    for point in points:
        response = (
            "    n/a"
            if point.response_change != point.response_change  # nan
            else f"{100 * point.response_change:+7.1f}%"
        )
        lines.append(
            f"{point.interval:+8.2f}       {100 * point.weight_change:+7.1f}%        {response}"
        )
    return "\n".join(lines)
