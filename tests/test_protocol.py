"""Pairing protocols are explicit timelines, checked without simulating anything."""

import pytest

from flyneuromod.experiments.protocol import (
    HANDLER_2019,
    ConditioningProtocol,
    pairing_timeline,
)


def total(segments, stimulus):
    return sum(s.duration for s in segments if getattr(s, stimulus))


def onset(segments, stimulus):
    return min(s.start for s in segments if getattr(s, stimulus))


@pytest.mark.parametrize("interval", [-1.2, -0.5, 0.0, 0.5, 1.0, 6.0])
def test_timeline_delivers_exactly_the_requested_interval(interval):
    """Review findings C1 and H1: the old schedule shifted, stretched and crashed."""
    segments = pairing_timeline(HANDLER_2019.evolve(dopamine_onset=interval))
    assert total(segments, "odour") == pytest.approx(HANDLER_2019.odour_duration)
    assert total(segments, "dopamine") == pytest.approx(HANDLER_2019.dopamine_duration)
    assert onset(segments, "dopamine") - onset(segments, "odour") == pytest.approx(interval)


def test_gaps_between_windows_are_simulated_as_silence():
    segments = pairing_timeline(HANDLER_2019.evolve(dopamine_onset=6.0))
    gaps = [s for s in segments if not s.odour and not s.dopamine]
    assert len(gaps) == 1
    assert gaps[0].duration == pytest.approx(4.0)  # odour ends at 2 s, dopamine starts at 6 s


def test_short_dopamine_inside_a_long_odour():
    protocol = ConditioningProtocol(odour_duration=5.0, dopamine_onset=0.2, dopamine_duration=1.0)
    segments = pairing_timeline(protocol)
    assert total(segments, "odour") == pytest.approx(5.0)
    assert total(segments, "dopamine") == pytest.approx(1.0)
    assert [(s.odour, s.dopamine) for s in segments] == [
        (True, False),
        (True, True),
        (True, False),
    ]


def test_segments_are_contiguous():
    segments = pairing_timeline(HANDLER_2019.evolve(dopamine_onset=-1.2))
    for before, after in zip(segments, segments[1:], strict=False):
        assert before.stop == pytest.approx(after.start)


@pytest.mark.parametrize(
    "changes",
    [
        {"odour_duration": 0},
        {"test_duration": 0},
        {"n_pairings": 0},
        {"odour_overlap": 1.5},
        {"odour_fraction": 0},
        {"odour_fraction": 0.6, "odour_overlap": 0.2},  # two odours cannot fit
        {"rest": -1},
        {"dopamine_rate": -5},
    ],
)
def test_invalid_protocols_are_refused(changes):
    with pytest.raises(ValueError):
        ConditioningProtocol(**changes)


def test_evolve_keeps_the_original():
    base = ConditioningProtocol()
    changed = base.evolve(dopamine_onset=-0.5)
    assert changed.dopamine_onset == -0.5
    assert base.dopamine_onset == 0.2
