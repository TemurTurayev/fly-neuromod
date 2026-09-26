"""Tests for ExcitabilityParams and ExcitabilityEffector."""

import math

import numpy as np
import pytest

from flyneuromod.neuromod.excitability import ExcitabilityEffector, ExcitabilityParams

# Provenance: SI unit definitions
MILLIVOLT: float = 1e-3  # Provenance: SI unit definition (1 mV = 1e-3 V)
MILLISECOND: float = 1e-3  # Provenance: SI unit definition (1 ms = 1e-3 s)


def test_params_validation_and_evolve():
    """ExcitabilityParams validates inputs and supports immutability via evolve."""
    params = ExcitabilityParams(max_conductance=2.0, e_k=-75.0 * MILLIVOLT, tau_effector=0.5)

    assert params.max_conductance == 2.0
    assert params.e_k == -0.075
    assert params.tau_effector == 0.5

    evolved = params.evolve(max_conductance=3.0)
    assert evolved.max_conductance == 3.0
    assert params.max_conductance == 2.0

    with pytest.raises(ValueError, match="max_conductance must be non-negative"):
        ExcitabilityParams(max_conductance=-0.5)

    with pytest.raises(ValueError, match="e_k must be finite"):
        ExcitabilityParams(e_k=np.nan)

    with pytest.raises(ValueError, match="tau_effector must be positive"):
        ExcitabilityParams(tau_effector=0.0)


def test_effector_relaxation_towards_occupancy():
    """Activation relaxes toward gi_occupancy with exact exponential time constant."""
    params = ExcitabilityParams(max_conductance=2.0, e_k=-80.0 * MILLIVOLT, tau_effector=0.5)
    dt = 0.01
    effector = ExcitabilityEffector(params, target_indices=[0, 1], dt=dt)

    assert np.all(effector.activation == 0.0)

    # Step with 100% receptor occupancy
    k = effector.step(1.0)

    expected_activation = 1.0 - math.exp(-dt / params.tau_effector)
    assert np.allclose(effector.activation, expected_activation)
    assert np.allclose(k, params.max_conductance * expected_activation)


def test_effector_step_independence_of_dt():
    """Stepping twice at dt equals stepping once at 2*dt within 1e-12."""
    params = ExcitabilityParams(max_conductance=1.5, e_k=-80.0 * MILLIVOLT, tau_effector=0.4)
    target = [0, 1, 2]
    gi_occ = np.array([0.8, 0.5, 0.2])

    eff_fine = ExcitabilityEffector(params, target_indices=target, dt=0.001)
    eff_coarse = ExcitabilityEffector(params, target_indices=target, dt=0.002)

    # Fine integration: two steps of 1 ms
    eff_fine.step(gi_occ)
    k_fine = eff_fine.step(gi_occ)

    # Coarse integration: one step of 2 ms
    k_coarse = eff_coarse.step(gi_occ)

    assert np.all(np.abs(eff_fine.activation - eff_coarse.activation) < 1e-12)
    assert np.all(np.abs(k_fine - k_coarse) < 1e-12)


def test_effector_reset():
    """reset() returns activation state to zero."""
    params = ExcitabilityParams(max_conductance=2.0, e_k=-80.0 * MILLIVOLT, tau_effector=0.5)
    effector = ExcitabilityEffector(params, target_indices=[0, 1], dt=0.01)

    effector.step(0.9)
    assert np.all(effector.activation > 0.0)

    effector.reset()
    assert np.all(effector.activation == 0.0)


def test_effector_input_validation():
    """ExcitabilityEffector validates target_indices, dt, and occupancy."""
    params = ExcitabilityParams()

    with pytest.raises(ValueError, match="dt must be positive"):
        ExcitabilityEffector(params, target_indices=[0], dt=0.0)

    with pytest.raises(ValueError, match="target_indices must be 1D"):
        ExcitabilityEffector(params, target_indices=[[0]], dt=0.001)

    with pytest.raises(ValueError, match="target_indices must be non-negative"):
        ExcitabilityEffector(params, target_indices=[-1], dt=0.001)

    effector = ExcitabilityEffector(params, target_indices=[0, 1], dt=0.001)

    with pytest.raises(ValueError, match="gi_occupancy must be finite"):
        effector.step(np.nan)

    with pytest.raises(ValueError, match="gi_occupancy shape does not match"):
        effector.step([0.5, 0.5, 0.5])
