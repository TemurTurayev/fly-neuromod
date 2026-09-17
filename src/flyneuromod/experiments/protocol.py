"""Stimulus protocols as explicit timelines.

A pairing is two time windows - odour and dopamine - and everything an
experiment does follows from where they sit relative to each other. Building
the schedule as data, before any simulation, makes it testable on its own: the
first implementation computed it with branching logic and silently shifted the
long intervals, stretched the dopamine pulse in backward pairing, and crashed on
simultaneous onset.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True, slots=True)
class ConditioningProtocol:
    """Timing and strength of a pairing experiment.

    Attributes
    ----------
    odour_duration:
        Odour presentation during pairing, in seconds. Dopamine signalling is
        slow, so a pairing of a few hundred milliseconds barely moves the
        receptors; behavioural experiments use a 60 s odour with thirty 1 s
        dopaminergic pulses (Aso & Rubin 2016).
    test_duration:
        Odour presentation used to measure a response, in seconds.
    odour_rate:
        Drive applied to the Kenyon cells of an odour, in hertz.
    dopamine_rate:
        Drive applied to the dopaminergic neurons during pairing, in hertz.
    dopamine_onset:
        Delay from odour onset to dopamine onset, in seconds. Positive is forward
        pairing (odour first), negative backward pairing.
    dopamine_duration:
        Length of the dopaminergic activation, in seconds.
    n_pairings:
        Number of pairing trials.
    rest:
        Pause between phases, in seconds. It must be long compared with the
        eligibility trace and with cAMP: a test odour presented while either is
        still up is trained by accident.
    readout_baseline_rate:
        Spontaneous rate the output neuron is held at by a fitted tonic drive, in
        hertz; 0 disables the fit. Extracting the mushroom body removes the
        output neuron's inputs from the rest of the brain and leaves it at
        threshold, where any loss of drive looks catastrophic. This is a
        calibration knob, not a measurement.
    odour_fraction:
        Fraction of Kenyon cells an odour activates.
    odour_overlap:
        Fraction of the trained odour's Kenyon cells shared by the control odour.
    """

    odour_duration: float = 5.0
    test_duration: float = 1.0
    odour_rate: float = 60.0
    dopamine_rate: float = 20.0
    dopamine_onset: float = 0.2
    dopamine_duration: float = 5.0
    n_pairings: int = 1
    rest: float = 15.0
    readout_baseline_rate: float = 30.0
    odour_fraction: float = 0.1
    odour_overlap: float = 0.2

    def __post_init__(self) -> None:
        for name in ("odour_duration", "test_duration", "dopamine_duration"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("odour_rate", "dopamine_rate", "rest", "readout_baseline_rate"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")
        if self.n_pairings < 1:
            raise ValueError("at least one pairing is required")
        if not 0 < self.odour_fraction <= 1:
            raise ValueError("odour_fraction must be in (0, 1]")
        if not 0 <= self.odour_overlap <= 1:
            raise ValueError("odour_overlap must be a fraction")
        if self.odour_fraction * (2 - self.odour_overlap) > 1:
            raise ValueError(
                "two odours of this size and overlap do not fit in the Kenyon cell "
                f"population (fraction {self.odour_fraction}, overlap {self.odour_overlap})"
            )

    def evolve(self, **changes: Any) -> ConditioningProtocol:
        """Return a copy with ``changes`` applied."""
        return replace(self, **changes)


HANDLER_2019 = ConditioningProtocol(
    odour_duration=2.0,
    dopamine_duration=1.0,
)
"""Timing protocol of Handler et al. (2019): 2 s odour, 1 s dopaminergic activation.

Intervals are measured onset to onset, as in the paper.
"""


@dataclass(frozen=True, slots=True)
class Segment:
    """A stretch of time with a fixed set of active stimuli."""

    start: float
    stop: float
    odour: bool
    dopamine: bool

    @property
    def duration(self) -> float:
        return self.stop - self.start


def pairing_timeline(protocol: ConditioningProtocol) -> tuple[Segment, ...]:
    """The sequence of stimulus segments of one pairing trial.

    Time zero is odour onset. Gaps between the two windows appear as segments
    with neither stimulus, so the interval is delivered exactly as requested.
    """
    odour = (0.0, protocol.odour_duration)
    dopamine = (
        protocol.dopamine_onset,
        protocol.dopamine_onset + protocol.dopamine_duration,
    )
    edges = sorted({*odour, *dopamine})
    segments = []
    for start, stop in zip(edges, edges[1:], strict=False):
        if stop - start <= 1e-12:
            continue
        segments.append(
            Segment(
                start=start,
                stop=stop,
                odour=odour[0] <= start < odour[1],
                dopamine=dopamine[0] <= start < dopamine[1],
            )
        )
    return tuple(segments)
