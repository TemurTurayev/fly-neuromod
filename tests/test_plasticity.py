"""The order-selective rule at Kenyon cell output synapses."""

import math

import numpy as np
import pytest

from flyneuromod.neuromod.plasticity import PlasticityParams, SynapticPlasticity


def make_plasticity(dt=1e-3, **kwargs):
    """Two synapses from two Kenyon cells, each in its own compartment."""
    weights = np.array([10.0, 10.0])
    params = PlasticityParams(
        tau_eligibility=kwargs.pop("tau_eligibility", 2.0),
        rate_depression=kwargs.pop("rate_depression", 0.5),
        rate_potentiation=kwargs.pop("rate_potentiation", 0.5),
        **kwargs,
    )
    plasticity = SynapticPlasticity(
        weights=weights,
        synapse_index=np.arange(2),
        presynaptic_index=np.array([0, 1]),
        compartment_index=np.array([0, 1]),
        n_compartments=2,
        params=params,
        dt=dt,
    )
    return plasticity, weights


def run(plasticity, seconds, spikes=(0.0, 0.0), gs=(0.0, 0.0), ip3=(0.0, 0.0), dt=1e-3):
    for _ in range(int(round(seconds / dt))):
        plasticity.step(np.array(spikes), gs_activation=np.array(gs), ip3=np.array(ip3))


def test_nothing_changes_without_dopamine():
    plasticity, weights = make_plasticity()
    run(plasticity, 1.0, spikes=(1.0, 1.0))
    assert weights == pytest.approx([10.0, 10.0])


def test_dopamine_alone_changes_nothing():
    plasticity, weights = make_plasticity()
    run(plasticity, 0.5, gs=(0.0, 0.0))
    run(plasticity, 0.5, gs=(0.8, 0.8), ip3=(0.8, 0.8))
    assert weights == pytest.approx([10.0, 10.0])


def test_calcium_then_gs_depresses_only_the_primed_synapse():
    """Forward pairing: the cell fired, then dopamine arrived."""
    plasticity, weights = make_plasticity()
    run(plasticity, 0.2, spikes=(1.0, 0.0))  # odour drives Kenyon cell 0 only
    run(plasticity, 0.5, gs=(0.6, 0.6))  # Gs activation arrives in both compartments
    assert weights[0] < 9.0
    assert weights[1] == pytest.approx(10.0)


def test_ip3_then_calcium_potentiates():
    """Backward pairing: dopamine was there first, then the cell fired."""
    plasticity, weights = make_plasticity()
    run(plasticity, 0.2, ip3=(0.8, 0.8))
    run(plasticity, 0.2, spikes=(1.0, 0.0), ip3=(0.8, 0.8))
    assert weights[0] > 10.0
    assert weights[1] == pytest.approx(10.0)


def test_gs_already_present_does_not_depress_a_cell_that_starts_firing_later():
    """Only the arrival of Gs activation counts, not a plateau it finds on arrival."""
    plasticity, weights = make_plasticity(rate_potentiation=0.0)
    run(plasticity, 0.5, gs=(0.6, 0.6))  # dopamine first
    run(plasticity, 0.5, spikes=(1.0, 0.0), gs=(0.6, 0.6))  # then the odour
    assert weights[0] == pytest.approx(10.0)


def test_ip3_rising_on_an_active_cell_does_not_potentiate():
    """Calcium that came first gets no credit from IP3 that arrives afterwards."""
    plasticity, weights = make_plasticity(rate_depression=0.0)
    run(plasticity, 0.5, spikes=(1.0, 0.0))  # the cell is already firing
    before = weights[0]
    run(plasticity, 0.5, spikes=(1.0, 0.0), ip3=(0.8, 0.0))
    assert weights[0] - before < 0.01 * before


def test_plasticity_stays_in_its_compartment():
    plasticity, weights = make_plasticity()
    run(plasticity, 0.2, spikes=(1.0, 1.0))
    run(plasticity, 0.5, gs=(0.6, 0.0))  # dopamine only in compartment 0
    assert weights[0] < 10.0
    assert weights[1] == pytest.approx(10.0)


def test_late_dopamine_writes_less():
    early, weights_early = make_plasticity(tau_eligibility=1.0)
    late, weights_late = make_plasticity(tau_eligibility=1.0)
    for plasticity in (early, late):
        run(plasticity, 0.2, spikes=(1.0, 0.0))
    run(late, 3.0)  # three trace time constants of silence
    for plasticity in (early, late):
        run(plasticity, 0.5, gs=(0.6, 0.6))
    assert weights_early[0] < weights_late[0] < 10.0


@pytest.mark.parametrize("coarse_dt", [2e-3, 5e-3])
def test_amount_learned_does_not_depend_on_the_integration_step(coarse_dt):
    """Review finding H2: a rate multiplied by dt made potentiation step-dependent."""

    def learn(dt):
        plasticity, weights = make_plasticity(dt=dt)
        spike_every = int(round(0.02 / dt))  # a 50 Hz spike train
        # backward pairing, then forward pairing, on the same synapse
        for step in range(int(round(0.4 / dt))):
            fires = step % spike_every == 0 and step * dt >= 0.2
            plasticity.step(np.array([1.0 if fires else 0.0, 0.0]), np.zeros(2), np.full(2, 0.7))
        for step in range(int(round(0.5 / dt))):
            gs = min(1.0, step * dt / 0.3) * 0.6
            plasticity.step(np.zeros(2), np.array([gs, 0.0]), np.zeros(2))
        return weights[0] / 10.0 - 1.0

    fine = learn(1e-3)
    coarse = learn(coarse_dt)
    assert coarse == pytest.approx(fine, abs=0.1 * abs(fine) + 1e-3)


def test_weights_stay_within_bounds():
    plasticity, weights = make_plasticity(
        rate_depression=50.0, rate_potentiation=50.0, min_fraction=0.2, max_fraction=1.5
    )
    run(plasticity, 0.1, spikes=(1.0, 0.0))
    run(plasticity, 0.5, gs=(1.0, 0.0))
    assert weights[0] == pytest.approx(2.0)
    plasticity.reset()
    run(plasticity, 0.1, ip3=(1.0, 0.0))
    run(plasticity, 0.1, spikes=(1.0, 0.0), ip3=(1.0, 0.0))
    assert weights[0] == pytest.approx(15.0)


def test_negative_weights_keep_their_sign():
    weights = np.array([-8.0])
    plasticity = SynapticPlasticity(
        weights=weights,
        synapse_index=np.array([0]),
        presynaptic_index=np.array([0]),
        compartment_index=np.array([0]),
        n_compartments=1,
        params=PlasticityParams(tau_eligibility=2.0, rate_depression=0.5, rate_potentiation=0.0),
        dt=1e-3,
    )
    for _ in range(200):
        plasticity.step(np.array([1.0]), np.zeros(1), np.zeros(1))
    for _ in range(500):
        plasticity.step(np.zeros(1), np.array([0.6]), np.zeros(1))
    assert -8.0 < weights[0] < 0.0


def test_recovery_returns_weights_towards_baseline():
    plasticity, weights = make_plasticity(tau_recovery=1.0)
    run(plasticity, 0.2, spikes=(1.0, 0.0))
    run(plasticity, 0.5, gs=(0.6, 0.6))
    depressed = weights[0]
    run(plasticity, 5.0, gs=(0.6, 0.6))
    assert depressed < weights[0]
    assert weights[0] == pytest.approx(10.0, rel=0.05)


def test_reset_restores_weights_and_traces():
    plasticity, weights = make_plasticity()
    run(plasticity, 0.2, spikes=(1.0, 1.0))
    run(plasticity, 0.5, gs=(0.6, 0.6))
    plasticity.reset()
    assert weights == pytest.approx([10.0, 10.0])
    assert plasticity.eligibility.tolist() == [0.0, 0.0]
    # after a reset, the Gs level that was present is treated as new again
    run(plasticity, 0.2, spikes=(1.0, 0.0), gs=(0.0, 0.0))
    run(plasticity, 0.1, gs=(0.6, 0.6))
    assert weights[0] < 10.0


def test_change_of_reports_learning_and_refuses_empty_selections():
    plasticity, _ = make_plasticity()
    run(plasticity, 0.2, spikes=(1.0, 0.0))
    run(plasticity, 0.5, gs=(0.6, 0.6))
    assert plasticity.change_of(np.array([True, False])) < 0
    assert plasticity.change_of(np.array([False, True])) == pytest.approx(0.0)
    assert math.isnan(plasticity.change_of(np.array([False, False])))


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        PlasticityParams(tau_eligibility=0.0, rate_depression=1.0, rate_potentiation=1.0)
    with pytest.raises(ValueError):
        PlasticityParams(
            tau_eligibility=1.0, rate_depression=1.0, rate_potentiation=1.0, min_fraction=1.2
        )
    params = PlasticityParams(tau_eligibility=1.0, rate_depression=1.0, rate_potentiation=1.0)
    with pytest.raises(ValueError, match="equal length"):
        SynapticPlasticity(
            np.array([1.0]), np.array([0, 1]), np.array([0]), np.array([0]), 1, params, 1e-3
        )
    with pytest.raises(ValueError, match="outside"):
        SynapticPlasticity(
            np.array([1.0]), np.array([0]), np.array([0]), np.array([3]), 1, params, 1e-3
        )
    plasticity, _ = make_plasticity()
    with pytest.raises(ValueError, match="shape"):
        plasticity.step(np.zeros(2), np.zeros(3), np.zeros(2))
