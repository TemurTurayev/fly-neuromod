"""Extracellular dopamine: release, diffusion into fields, DAT uptake."""

import numpy as np
import pytest

from flyneuromod.neuromod.field import DopamineField, ReleaseKinetics


def single_field(**kwargs) -> DopamineField:
    kinetics = ReleaseKinetics(
        per_spike=kwargs.pop("per_spike", 0.05),
        v_max=kwargs.pop("v_max", 4.0),
        k_m=kwargs.pop("k_m", 0.2),
        **kwargs,
    )
    return DopamineField(n_fields=1, kinetics=kinetics, dt=1e-3)


def test_no_release_keeps_concentration_at_zero():
    field = single_field()
    for _ in range(100):
        field.step(np.zeros(1))
    assert field.concentration[0] == pytest.approx(0.0)


def test_single_spike_raises_concentration_by_quantum():
    field = single_field(per_spike=0.05, v_max=0.0)
    field.step(np.array([1.0]))
    assert field.concentration[0] == pytest.approx(0.05)


def test_release_scales_with_number_of_spikes():
    field = single_field(per_spike=0.05, v_max=0.0)
    field.step(np.array([4.0]))
    assert field.concentration[0] == pytest.approx(0.2)


def test_uptake_is_saturated_at_high_concentration():
    """Well above K_m, clearance runs at V_max (zero order)."""
    field = single_field(v_max=4.0, k_m=0.2)
    field.concentration[0] = 10.0
    field.step(np.zeros(1))
    assert field.concentration[0] == pytest.approx(10.0 - 4.0 * 1e-3, rel=1e-3)


def test_uptake_is_first_order_at_low_concentration():
    """Well below K_m, clearance is exponential with rate V_max / K_m."""
    field = single_field(v_max=4.0, k_m=0.2)
    field.concentration[0] = 0.001
    for _ in range(50):
        field.step(np.zeros(1))
    expected = 0.001 * np.exp(-(4.0 / 0.2) * 0.05)
    assert field.concentration[0] == pytest.approx(expected, rel=0.05)


def test_concentration_never_goes_negative():
    field = single_field(v_max=1e6, k_m=0.2)
    field.concentration[0] = 0.01
    field.step(np.zeros(1))
    assert field.concentration[0] >= 0.0


def test_tonic_release_reaches_steady_state():
    """Steady state where release rate equals Michaelis-Menten uptake."""
    field = single_field(per_spike=0.001, v_max=4.0, k_m=0.2)
    rate_per_step = np.array([0.5])  # spikes per step, i.e. 500 Hz onto the field
    for _ in range(20_000):
        field.step(rate_per_step)
    c = field.concentration[0]
    release_rate = 0.5 * 0.001 / 1e-3  # micromolar per second
    uptake = 4.0 * c / (0.2 + c)
    assert uptake == pytest.approx(release_rate, rel=0.02)


def test_fields_are_independent():
    kinetics = ReleaseKinetics(per_spike=0.05, v_max=0.0, k_m=0.2)
    field = DopamineField(n_fields=3, kinetics=kinetics, dt=1e-3)
    field.step(np.array([1.0, 0.0, 2.0]))
    assert field.concentration.tolist() == pytest.approx([0.05, 0.0, 0.1])


def test_uptake_block_slows_clearance():
    """A competitive DAT blocker raises the apparent K_m, so clearance gets slower."""
    baseline = single_field(v_max=4.0, k_m=0.2)
    blocked = single_field(v_max=4.0, k_m=2.0)
    baseline.concentration[0] = blocked.concentration[0] = 0.5
    for _ in range(100):
        baseline.step(np.zeros(1))
        blocked.step(np.zeros(1))
    assert blocked.concentration[0] > baseline.concentration[0]


def test_spike_count_shape_is_validated():
    field = single_field()
    with pytest.raises(ValueError):
        field.step(np.zeros(2))


def test_invalid_kinetics_are_rejected():
    with pytest.raises(ValueError):
        ReleaseKinetics(per_spike=-1.0, v_max=4.0, k_m=0.2)
    with pytest.raises(ValueError):
        ReleaseKinetics(per_spike=0.05, v_max=4.0, k_m=0.0)


def test_reset_clears_concentration():
    field = single_field()
    field.step(np.array([3.0]))
    field.reset()
    assert field.concentration[0] == pytest.approx(0.0)
