"""Dopamine-gated plasticity at Kenyon cell output synapses."""

import numpy as np
import pytest

from flyneuromod.neuromod.plasticity import PlasticityParams, SynapticPlasticity


def make_plasticity(n_synapses=2, presyn=(0, 1), compartment=(0, 1), **kwargs) -> tuple:
    """Two synapses from two Kenyon cells into two different compartments."""
    weights = np.array([10.0, 10.0])
    # rates are fractions of the weight per second: ~0.8/s depresses by about a
    # third during half a second of dopamine
    params = PlasticityParams(
        tau_eligibility=kwargs.pop("tau_eligibility", 5.0),
        rate_depression=kwargs.pop("rate_depression", 0.8),
        rate_potentiation=kwargs.pop("rate_potentiation", 0.4),
        **kwargs,
    )
    plasticity = SynapticPlasticity(
        weights=weights,
        synapse_index=np.arange(n_synapses),
        presynaptic_index=np.array(presyn),
        compartment_index=np.array(compartment),
        params=params,
        dt=1e-3,
    )
    return plasticity, weights


def test_no_dopamine_leaves_weights_untouched():
    plasticity, weights = make_plasticity()
    for _ in range(1000):
        plasticity.step(np.array([1.0, 1.0]), camp=np.zeros(2), calcium=np.zeros(2))
    assert weights == pytest.approx([10.0, 10.0])


def test_dopamine_without_presynaptic_activity_leaves_weights_untouched():
    """Dopamine alone does not depress: the rule needs the coincidence."""
    plasticity, weights = make_plasticity()
    for _ in range(1000):
        plasticity.step(np.zeros(2), camp=np.ones(2), calcium=np.zeros(2))
    assert weights == pytest.approx([10.0, 10.0])


def test_forward_pairing_depresses_the_active_synapse():
    """Kenyon cell activity followed by dopamine (cAMP) -> depression."""
    plasticity, weights = make_plasticity()
    for _ in range(100):  # odour: only Kenyon cell 0 spikes
        plasticity.step(np.array([1.0, 0.0]), camp=np.zeros(2), calcium=np.zeros(2))
    for _ in range(500):  # then dopamine in both compartments
        plasticity.step(np.zeros(2), camp=np.ones(2), calcium=np.zeros(2))
    assert weights[0] < 9.0
    assert weights[1] == pytest.approx(10.0)


def test_backward_pairing_potentiates():
    """Dopamine (calcium branch) followed by Kenyon cell activity -> potentiation."""
    plasticity, weights = make_plasticity()
    for _ in range(100):
        plasticity.step(np.array([1.0, 0.0]), camp=np.zeros(2), calcium=np.ones(2))
    assert weights[0] > 10.0
    assert weights[1] == pytest.approx(10.0)


def test_plasticity_is_compartment_specific():
    plasticity, weights = make_plasticity()
    for _ in range(100):
        plasticity.step(np.array([1.0, 1.0]), camp=np.zeros(2), calcium=np.zeros(2))
    for _ in range(500):
        plasticity.step(np.zeros(2), camp=np.array([1.0, 0.0]), calcium=np.zeros(2))
    assert weights[0] < 9.0
    assert weights[1] == pytest.approx(10.0)


def test_eligibility_decays_so_late_dopamine_has_less_effect():
    early, weights_early = make_plasticity(tau_eligibility=1.0)
    late, weights_late = make_plasticity(tau_eligibility=1.0)

    for plasticity in (early, late):
        for _ in range(100):
            plasticity.step(np.array([1.0, 0.0]), camp=np.zeros(2), calcium=np.zeros(2))
    for _ in range(3000):  # three eligibility time constants of waiting
        late.step(np.zeros(2), camp=np.zeros(2), calcium=np.zeros(2))
    for plasticity in (early, late):
        for _ in range(500):
            plasticity.step(np.zeros(2), camp=np.ones(2), calcium=np.zeros(2))

    assert weights_early[0] < weights_late[0] < 10.0


def test_weights_stay_within_bounds():
    plasticity, weights = make_plasticity(
        rate_depression=1e4, rate_potentiation=1e4, min_fraction=0.2, max_fraction=1.5
    )
    for _ in range(200):
        plasticity.step(np.array([1.0, 1.0]), camp=np.ones(2), calcium=np.zeros(2))
    assert weights[0] == pytest.approx(2.0)
    for _ in range(200):
        plasticity.step(np.array([1.0, 1.0]), camp=np.zeros(2), calcium=np.ones(2))
    assert weights[0] == pytest.approx(15.0)


def test_negative_weights_keep_their_sign():
    """Inhibitory output synapses are scaled, never flipped."""
    weights = np.array([-8.0])
    plasticity = SynapticPlasticity(
        weights=weights,
        synapse_index=np.array([0]),
        presynaptic_index=np.array([0]),
        compartment_index=np.array([0]),
        params=PlasticityParams(tau_eligibility=5.0, rate_depression=0.8, rate_potentiation=0.0),
        dt=1e-3,
    )
    for _ in range(100):
        plasticity.step(np.array([1.0]), camp=np.zeros(1), calcium=np.zeros(1))
    for _ in range(500):
        plasticity.step(np.zeros(1), camp=np.ones(1), calcium=np.zeros(1))
    assert -8.0 < weights[0] < 0.0


def test_recovery_returns_weights_towards_baseline():
    plasticity, weights = make_plasticity(tau_recovery=1.0)
    for _ in range(100):
        plasticity.step(np.array([1.0, 0.0]), camp=np.zeros(2), calcium=np.zeros(2))
    for _ in range(500):
        plasticity.step(np.zeros(2), camp=np.ones(2), calcium=np.zeros(2))
    depressed = weights[0]
    for _ in range(5000):  # five recovery time constants without dopamine
        plasticity.step(np.zeros(2), camp=np.zeros(2), calcium=np.zeros(2))
    assert depressed < weights[0] < 10.0
    assert weights[0] == pytest.approx(10.0, rel=0.05)


def test_reset_restores_baseline_weights():
    plasticity, weights = make_plasticity()
    for _ in range(100):
        plasticity.step(np.array([1.0, 1.0]), camp=np.ones(2), calcium=np.zeros(2))
    plasticity.reset()
    assert weights == pytest.approx([10.0, 10.0])
    assert plasticity.eligibility.tolist() == [0.0, 0.0]


def test_depression_fraction_reports_learning():
    plasticity, _ = make_plasticity()
    for _ in range(100):
        plasticity.step(np.array([1.0, 0.0]), camp=np.zeros(2), calcium=np.zeros(2))
    for _ in range(500):
        plasticity.step(np.zeros(2), camp=np.ones(2), calcium=np.zeros(2))
    assert plasticity.weight_factor[0] < 1.0
    assert plasticity.weight_factor[1] == pytest.approx(1.0)


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        PlasticityParams(tau_eligibility=0.0, rate_depression=1.0, rate_potentiation=1.0)
    with pytest.raises(ValueError):
        PlasticityParams(
            tau_eligibility=1.0, rate_depression=1.0, rate_potentiation=1.0, min_fraction=1.2
        )
    with pytest.raises(ValueError):
        SynapticPlasticity(
            weights=np.array([1.0]),
            synapse_index=np.array([0, 1]),  # longer than the other arrays
            presynaptic_index=np.array([0]),
            compartment_index=np.array([0]),
            params=PlasticityParams(
                tau_eligibility=1.0, rate_depression=1.0, rate_potentiation=1.0
            ),
            dt=1e-3,
        )
