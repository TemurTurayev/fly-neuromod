"""Tests for per-neuron potassium leak conductance in the LIF engine."""

import numpy as np
import pytest
from scipy import sparse

from flyneuromod.engine.lif import LIFNetwork
from flyneuromod.engine.params import LIFParams, PoissonDrive

# Provenance: SI unit definitions
MILLIVOLT: float = 1e-3  # Provenance: SI unit definition (1 mV = 1e-3 V)
MILLISECOND: float = 1e-3  # Provenance: SI unit definition (1 ms = 1e-3 s)


def make_random_network(n_neurons: int = 15, seed: int = 42) -> tuple[LIFNetwork, LIFParams]:
    """Create a small random test network."""
    rng = np.random.default_rng(seed)
    dense = rng.uniform(-1.0, 1.0, size=(n_neurons, n_neurons))
    np.fill_diagonal(dense, 0.0)
    weights = sparse.csr_matrix(dense)
    params = LIFParams()
    net = LIFNetwork(weights, params, rng=np.random.default_rng(seed))
    return net, params


def test_bit_for_bit_unmodified_baseline():
    """With no conductance set, spike trains are identical to baseline."""
    net1, params = make_random_network(seed=123)
    net2, _ = make_random_network(seed=123)

    drives = {0: PoissonDrive(rate=200.0, weight_factor=300.0)}
    net1.set_poisson_drives(drives)
    net2.set_poisson_drives(drives)

    trains1 = net1.run(0.2)
    trains2 = net2.run(0.2)

    assert trains1.neuron.size > 0
    assert np.array_equal(trains1.neuron, trains2.neuron)
    assert np.array_equal(trains1.time, trains2.time)


def test_hyperpolarization_towards_effective_rest():
    """k > 0 hyperpolarizes a silent neuron toward (v_rest + k*E_K)/(1+k) exactly."""
    net, params = make_random_network(n_neurons=2, seed=1)
    e_k = -80.0 * MILLIVOLT
    k = 2.0

    v_expected = (params.v_rest + k * e_k) / (1.0 + k)
    net.set_potassium_conductance([0], k=k, e_k=e_k)
    net.reset_state()

    assert np.isclose(net.v_rest[0], v_expected)
    assert np.isclose(net.v[0], v_expected)

    # Step simulation without drive and verify voltage stays at effective rest
    for _ in range(50):
        net.step()
        assert np.isclose(net.v[0], v_expected, atol=1e-14)


def test_shunting_reduces_input_deflection():
    """The same input moves the membrane (1 + k) times less: a conductance shunts.

    The drive enters through the synaptic variable, whose steady state is
    ``drive * tau_synapse`` (1 V/s -> 5 mV), so 1 V/s keeps the neuron below its
    7 mV threshold gap and the deflection is a true steady state.
    """
    net, params = make_random_network(n_neurons=2, seed=1)
    net.set_tonic_drive({0: 1.0})

    net.reset_state()
    trains = net.run(0.3)
    deflection_k0 = net.v[0] - params.v_rest
    assert trains.neuron.size == 0

    e_k, k = -80.0 * MILLIVOLT, 2.0
    net.set_potassium_conductance([0], k=k, e_k=e_k)
    net.reset_state()
    net.run(0.3)
    v_eff = (params.v_rest + k * e_k) / (1.0 + k)
    deflection_k2 = net.v[0] - v_eff

    assert 0 < deflection_k2 < deflection_k0
    assert np.isclose(deflection_k2, deflection_k0 / (1.0 + k), rtol=1e-2)


def test_reversible_firing_suppression():
    """Neuron stops firing under high k and resumes firing when k returns to 0."""
    net, _ = make_random_network(n_neurons=2, seed=1)
    net.set_tonic_drive({0: 3.0})  # 15 mV of drive: fires at k = 0, not at k = 5

    # Baseline firing with k = 0
    t1 = net.run(0.1)
    assert t1.neuron.size > 0

    # Suppress firing with k = 5.0
    net.set_potassium_conductance([0], k=5.0, e_k=-80.0 * MILLIVOLT)
    net.reset_state()
    t2 = net.run(0.1)
    assert t2.neuron.size == 0

    # Restore firing with k = 0.0
    net.set_potassium_conductance([0], k=0.0, e_k=-80.0 * MILLIVOLT)
    net.reset_state()
    t3 = net.run(0.1)
    assert t3.neuron.size > 0
    assert np.array_equal(t1.neuron, t3.neuron)
    assert np.array_equal(t1.time, t3.time)


def test_sparse_conductance_update_touches_only_given_indices():
    """set_potassium_conductance updates ONLY the specified target indices."""
    net, _ = make_random_network(n_neurons=5, seed=1)

    decay_v_orig = net._decay_v.copy()
    g_to_v_orig = net._g_to_v.copy()
    v_rest_orig = net.v_rest.copy()

    target_indices = [1, 3]
    net.set_potassium_conductance(target_indices, k=1.5, e_k=-80.0 * MILLIVOLT)

    untouched = [0, 2, 4]
    assert np.array_equal(net._decay_v[untouched], decay_v_orig[untouched])
    assert np.array_equal(net._g_to_v[untouched], g_to_v_orig[untouched])
    assert np.array_equal(net.v_rest[untouched], v_rest_orig[untouched])

    assert not np.array_equal(net._decay_v[target_indices], decay_v_orig[target_indices])
    assert not np.array_equal(net._g_to_v[target_indices], g_to_v_orig[target_indices])
    assert not np.array_equal(net.v_rest[target_indices], v_rest_orig[target_indices])


def test_tau_eff_equals_tau_synapse_raises_value_error():
    """Setting k such that tau_eff == tau_synapse raises ValueError."""
    net, params = make_random_network(n_neurons=2, seed=1)
    # tau_m = 20 ms, tau_s = 5 ms -> tau_eff = 20 / (1 + 3) = 5 ms = tau_s
    k_singular = (params.tau_membrane / params.tau_synapse) - 1.0

    with pytest.raises(ValueError, match="tau_eff equals tau_synapse"):
        net.set_potassium_conductance([0], k=k_singular, e_k=-80.0 * MILLIVOLT)


def test_set_potassium_conductance_input_validation():
    """Invalid parameters for set_potassium_conductance raise appropriate errors."""
    net, _ = make_random_network(n_neurons=3, seed=1)

    with pytest.raises(ValueError, match="k must be non-negative"):
        net.set_potassium_conductance([0], k=-1.0, e_k=-80.0 * MILLIVOLT)

    with pytest.raises(ValueError, match="k must be finite"):
        net.set_potassium_conductance([0], k=np.nan, e_k=-80.0 * MILLIVOLT)

    with pytest.raises(ValueError, match="e_k must be finite"):
        net.set_potassium_conductance([0], k=1.0, e_k=np.nan)

    with pytest.raises(IndexError, match="outside the network"):
        net.set_potassium_conductance([10], k=1.0, e_k=-80.0 * MILLIVOLT)
