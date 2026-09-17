"""Receptor occupancy and the second messengers it drives."""

import numpy as np
import pytest

from flyneuromod.neuromod.receptors import (
    ReceptorPopulation,
    ReceptorSpec,
    SecondMessenger,
    SignalingParams,
)


def spec(**kwargs) -> ReceptorSpec:
    defaults = dict(name="Dop1R1", ec50=1.0, hill=1.0, tau_on=0.2, tau_off=1.0, coupling="Gs")
    return ReceptorSpec(**{**defaults, **kwargs})


def test_steady_state_occupancy_is_half_at_ec50():
    population = ReceptorPopulation(spec(ec50=1.0), n_cells=1, dt=1e-3)
    for _ in range(20_000):
        population.step(np.array([1.0]))
    assert population.occupancy[0] == pytest.approx(0.5, abs=1e-3)


def test_occupancy_follows_hill_curve():
    population = ReceptorPopulation(spec(ec50=1.0, hill=2.0), n_cells=3, dt=1e-3)
    for _ in range(20_000):
        population.step(np.array([0.5, 1.0, 2.0]))
    expected = np.array([0.5**2, 1.0, 2.0**2]) / (1.0 + np.array([0.5**2, 1.0, 2.0**2]))
    assert population.occupancy == pytest.approx(expected, abs=1e-3)


def test_occupancy_rises_with_on_time_constant():
    population = ReceptorPopulation(spec(ec50=1.0, tau_on=0.2), n_cells=1, dt=1e-3)
    for _ in range(200):  # one tau_on at saturating concentration
        population.step(np.array([1000.0]))
    assert population.occupancy[0] == pytest.approx(1 - np.exp(-1), abs=0.02)


def test_occupancy_decays_with_off_time_constant():
    population = ReceptorPopulation(spec(tau_off=1.0), n_cells=1, dt=1e-3)
    population.occupancy[0] = 1.0
    for _ in range(1000):  # one tau_off with no dopamine
        population.step(np.array([0.0]))
    assert population.occupancy[0] == pytest.approx(np.exp(-1), abs=0.02)


def test_competitive_antagonist_shifts_the_dose_response():
    """An antagonist at its K_i doubles the apparent EC50."""
    free = ReceptorPopulation(spec(ec50=1.0), n_cells=1, dt=1e-3)
    blocked = ReceptorPopulation(spec(ec50=1.0), n_cells=1, dt=1e-3, occupancy_scale=1.0)
    blocked.set_competitive_antagonist(concentration=1.0, k_i=1.0)
    for _ in range(20_000):
        free.step(np.array([1.0]))
        blocked.step(np.array([1.0]))
    assert free.occupancy[0] == pytest.approx(0.5, abs=1e-3)
    assert blocked.occupancy[0] == pytest.approx(1 / 3, abs=1e-3)


def test_receptor_knockout_removes_the_response():
    population = ReceptorPopulation(spec(), n_cells=2, dt=1e-3, occupancy_scale=np.array([1.0, 0.0]))
    for _ in range(5_000):
        population.step(np.array([10.0, 10.0]))
    assert population.occupancy[0] > 0.8
    assert population.occupancy[1] == pytest.approx(0.0)


def test_camp_rises_with_gs_and_falls_with_gi():
    params = SignalingParams(tau_camp=1.0, gain_gs=1.0, gain_gi=1.0, basal_camp=0.0)
    messenger = SecondMessenger(n_cells=1, params=params, dt=1e-3)
    for _ in range(3_000):
        messenger.step(gs_occupancy=np.array([1.0]), gi_occupancy=np.array([0.0]))
    excited = messenger.camp[0]
    assert excited > 0.9

    for _ in range(3_000):
        messenger.step(gs_occupancy=np.array([0.0]), gi_occupancy=np.array([1.0]))
    assert messenger.camp[0] < 0.0


def test_camp_returns_to_basal():
    params = SignalingParams(tau_camp=0.5, basal_camp=0.1)
    messenger = SecondMessenger(n_cells=1, params=params, dt=1e-3)
    messenger.camp[0] = 5.0
    for _ in range(5_000):
        messenger.step(gs_occupancy=np.zeros(1), gi_occupancy=np.zeros(1))
    assert messenger.camp[0] == pytest.approx(0.1, abs=1e-3)


def test_gq_calcium_tracks_its_own_occupancy():
    params = SignalingParams(tau_calcium=0.3, gain_gq=2.0)
    messenger = SecondMessenger(n_cells=1, params=params, dt=1e-3)
    for _ in range(3_000):
        messenger.step(
            gs_occupancy=np.zeros(1), gi_occupancy=np.zeros(1), gq_occupancy=np.array([0.5])
        )
    assert messenger.calcium[0] == pytest.approx(1.0, abs=0.05)


def test_invalid_receptor_spec_is_rejected():
    with pytest.raises(ValueError):
        spec(ec50=0.0)
    with pytest.raises(ValueError):
        spec(hill=0.0)
    with pytest.raises(ValueError):
        spec(tau_on=0.0)
    with pytest.raises(ValueError):
        spec(coupling="Gx")
