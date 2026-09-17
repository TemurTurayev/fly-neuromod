import numpy as np
import pytest
from scipy import sparse

from flyneuromod.engine.lif import LIFNetwork
from flyneuromod.engine.params import LIFParams, PoissonDrive


def empty_network(n: int = 3, params: LIFParams | None = None) -> LIFNetwork:
    weights = sparse.csr_matrix((n, n), dtype=np.float64)
    return LIFNetwork(weights, params or LIFParams(), rng=np.random.default_rng(0))


def analytic_v(t: float, v0: float, g0: float, p: LIFParams) -> float:
    """Exact solution of the subthreshold dynamics with no input."""
    decay_v = np.exp(-t / p.tau_membrane)
    decay_g = np.exp(-t / p.tau_synapse)
    return (
        p.v_rest
        + (v0 - p.v_rest) * decay_v
        + g0 * p.tau_synapse / (p.tau_synapse - p.tau_membrane) * (decay_g - decay_v)
    )


def test_membrane_relaxes_to_rest_without_input():
    p = LIFParams()
    net = empty_network(params=p)
    net.v[:] = p.v_rest + 3e-3
    net.run(0.2)
    assert net.v == pytest.approx(np.full(3, p.v_rest), abs=1e-6)


def test_subthreshold_trajectory_matches_exact_solution():
    p = LIFParams()
    net = empty_network(params=p)
    v0, g0 = p.v_rest, 1.0e-3
    net.g[:] = g0
    n_steps = 200
    net.run(n_steps * p.dt)
    expected = analytic_v(n_steps * p.dt, v0, g0, p)
    assert net.v[0] == pytest.approx(expected, rel=1e-12)
    assert net.g[0] == pytest.approx(g0 * np.exp(-n_steps * p.dt / p.tau_synapse), rel=1e-12)


def test_spike_is_emitted_and_state_is_reset():
    p = LIFParams()
    net = empty_network(params=p)
    net.v[0] = p.v_threshold + 1e-4
    net.g[0] = 5e-3
    spiking = net.step()
    assert spiking.tolist() == [0]
    assert net.v[0] == pytest.approx(p.v_reset)
    assert net.g[0] == pytest.approx(0.0)


def test_no_spike_during_refractory_period():
    p = LIFParams(dt=0.1e-3)
    net = empty_network(params=p)
    net.v[0] = p.v_threshold + 1e-4
    assert net.step().tolist() == [0]
    for _ in range(p.refractory_steps - 1):
        net.v[0] = p.v_threshold + 1e-4  # keep it above threshold
        assert net.step().size == 0
    net.v[0] = p.v_threshold + 1e-4
    assert net.step().tolist() == [0]


def test_synaptic_transmission_is_delayed_and_signed():
    p = LIFParams()
    weights = sparse.csr_matrix(np.array([[0.0, 0.0], [7.0, 0.0]]))  # pre 0 -> post 1, 7 synapses
    net = LIFNetwork(weights, p, rng=np.random.default_rng(0))
    net.v[0] = p.v_threshold + 1e-4
    net.step()  # neuron 0 spikes
    for _ in range(p.delay_steps - 1):
        net.step()
        assert net.g[1] == pytest.approx(0.0)
    net.step()
    assert net.g[1] == pytest.approx(7 * p.w_synapse)


def test_inhibitory_weight_decreases_drive():
    p = LIFParams()
    weights = sparse.csr_matrix(np.array([[0.0, 0.0], [-4.0, 0.0]]))
    net = LIFNetwork(weights, p, rng=np.random.default_rng(0))
    net.v[0] = p.v_threshold + 1e-4
    net.step()
    for _ in range(p.delay_steps):
        net.step()
    assert net.g[1] < 0


def test_poisson_drive_makes_a_neuron_fire():
    p = LIFParams()
    net = empty_network(params=p)
    net.set_poisson_drives({0: PoissonDrive(rate=150.0)})
    trains = net.run(0.5)
    rates = trains.rates()
    assert rates[0] > 50.0
    assert rates[1] == 0.0


def test_zero_rate_drive_produces_no_spikes():
    net = empty_network()
    net.set_poisson_drives({0: PoissonDrive(rate=0.0)})
    assert net.run(0.2).neuron.size == 0


def test_silencing_removes_outgoing_transmission():
    p = LIFParams()
    weights = sparse.csr_matrix(np.array([[0.0, 0.0], [7.0, 0.0]]))
    net = LIFNetwork(weights, p, rng=np.random.default_rng(0))
    net.silence([0])
    net.v[0] = p.v_threshold + 1e-4
    net.step()
    for _ in range(p.delay_steps + 2):
        net.step()
    assert net.g[1] == pytest.approx(0.0)


def test_callback_sees_every_step_and_spikes():
    p = LIFParams()
    net = empty_network(params=p)
    net.set_poisson_drives({0: PoissonDrive(rate=300.0)})
    seen: list[tuple[int, int]] = []
    net.run(0.05, callbacks=[lambda _net, step, spikes: seen.append((step, spikes.size))])
    assert len(seen) == 500
    assert sum(n for _, n in seen) > 0


def test_reset_state_clears_delayed_input():
    p = LIFParams()
    weights = sparse.csr_matrix(np.array([[0.0, 0.0], [7.0, 0.0]]))
    net = LIFNetwork(weights, p, rng=np.random.default_rng(0))
    net.v[0] = p.v_threshold + 1e-4
    net.step()
    net.reset_state()
    for _ in range(p.delay_steps + 2):
        net.step()
    assert net.g[1] == pytest.approx(0.0)
    assert net.v[1] == pytest.approx(p.v_rest)


def test_non_square_connectivity_is_rejected():
    with pytest.raises(ValueError):
        LIFNetwork(sparse.csr_matrix((2, 3)))


def test_spike_trains_report_times_per_neuron():
    p = LIFParams()
    net = empty_network(params=p)
    net.set_poisson_drives({1: PoissonDrive(rate=200.0)})
    trains = net.run(0.3)
    times = trains.times_of(1)
    assert times.size > 0
    assert np.all(np.diff(times) > 0)
    assert trains.times_of(0).size == 0


def test_tonic_drive_raises_the_membrane_and_can_make_a_neuron_fire():
    p = LIFParams()
    net = empty_network(params=p)
    net.set_tonic_drive({0: 2.0})  # volts per second of synaptic drive
    trains = net.run(0.5)
    assert trains.rates()[0] > 0
    assert trains.rates()[1] == 0.0


def test_tonic_drive_is_replaced_not_accumulated():
    net = empty_network()
    net.set_tonic_drive({0: 2.0})
    net.set_tonic_drive({1: 1.0})
    assert net.tonic_drive[0] == 0.0
    assert net.tonic_drive[1] == pytest.approx(1.0)


def test_tonic_drive_rejects_unknown_neurons():
    net = empty_network()
    with pytest.raises(IndexError):
        net.set_tonic_drive({99: 1.0})
