"""The engine reproduces the reference Brian 2 model spike for spike.

The fast layer here is a re-implementation of the published Brian 2 model. That
claim is only worth making if it is tested, so this compares the two simulators
on the same random network with the same parameters and no stochastic input:
every spike must match, to the time step.

Brian 2 is an optional dependency (``uv sync --extra brian``); the test skips
when it is absent.
"""

import numpy as np
import pytest
from scipy import sparse

from flyneuromod.engine.lif import LIFNetwork
from flyneuromod.engine.params import LIFParams

brian2 = pytest.importorskip("brian2", reason="brian2 not installed")

pytestmark = pytest.mark.slow

N_NEURONS = 40
DURATION = 0.1  # seconds


def random_weights(seed: int = 0) -> sparse.csr_matrix:
    """Signed synapse counts for a small random network."""
    rng = np.random.default_rng(seed)
    dense = rng.integers(0, 40, size=(N_NEURONS, N_NEURONS)).astype(np.float64)
    signs = rng.choice([1.0, -1.0], size=(N_NEURONS, N_NEURONS), p=[0.8, 0.2])
    dense *= signs
    np.fill_diagonal(dense, 0.0)
    return sparse.csr_matrix(dense)


def run_ours(weights: sparse.csr_matrix, params: LIFParams, seeded: np.ndarray):
    network = LIFNetwork(weights, params, rng=np.random.default_rng(0))
    network.v[seeded] = params.v_threshold + 1e-4
    trains = network.run(DURATION)
    return {
        (int(n), round(float(t) / params.dt))
        for n, t in zip(trains.neuron, trains.time, strict=True)
    }


def run_brian(weights: sparse.csr_matrix, params: LIFParams, seeded: np.ndarray):
    from brian2 import (
        Network,
        NeuronGroup,
        SpikeMonitor,
        Synapses,
        defaultclock,
        ms,
        mV,
        second,
    )

    brian2.prefs.codegen.target = "numpy"
    defaultclock.dt = params.dt * second

    namespace = {
        "v_0": params.v_rest * 1000 * mV,
        "t_mbr": params.tau_membrane * 1000 * ms,
        "tau": params.tau_synapse * 1000 * ms,
    }
    neurons = NeuronGroup(
        N=N_NEURONS,
        model="""
            dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
            dg/dt = -g / tau              : volt (unless refractory)
            rfc                           : second
        """,
        method="linear",
        threshold=f"v > {params.v_threshold * 1000} * mV",
        reset=f"v = {params.v_reset * 1000} * mV; g = 0 * mV",
        refractory="rfc",
        namespace=namespace,
    )
    neurons.v = params.v_rest * 1000 * mV
    neurons.g = 0 * mV
    neurons.rfc = params.t_refractory * 1000 * ms
    neurons.v[seeded] = (params.v_threshold + 1e-4) * 1000 * mV

    synapses = Synapses(
        neurons, neurons, "w : volt", on_pre="g += w", delay=params.t_delay * 1000 * ms
    )
    coo = sparse.coo_matrix(weights)
    synapses.connect(i=coo.col.tolist(), j=coo.row.tolist())  # weights are [post, pre]
    synapses.w = coo.data * params.w_synapse * 1000 * mV

    monitor = SpikeMonitor(neurons)
    Network(neurons, synapses, monitor).run(DURATION * second)
    return {
        (int(n), round(float(t / second) / params.dt))
        for n, t in zip(monitor.i, monitor.t, strict=True)
    }


def test_spike_trains_match_the_reference_simulator():
    params = LIFParams()
    weights = random_weights()
    seeded = np.array([0, 1, 2, 3, 4])

    ours = run_ours(weights, params, seeded)
    theirs = run_brian(weights, params, seeded)

    assert ours, "the test network produced no spikes at all"
    only_ours = ours - theirs
    only_theirs = theirs - ours
    assert not only_ours and not only_theirs, (
        f"{len(only_ours)} spikes only in this engine, {len(only_theirs)} only in Brian 2; "
        f"{len(ours & theirs)} shared"
    )
