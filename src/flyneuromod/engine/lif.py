"""Event-driven leaky integrate-and-fire network.

This is a NumPy/SciPy re-implementation of the Brian 2 model published by
Shiu et al. (2024). It uses the exact propagator of the linear subthreshold
dynamics, so with identical parameters it reproduces the reference simulator
step for step (see ``tests/test_brian2_equivalence.py``) while running fast
enough for the multi-second protocols that neuromodulation experiments need.

Dynamics (per neuron)::

    dv/dt = (v_rest - v + g) / tau_membrane      (frozen while refractory)
    dg/dt = -g / tau_synapse                     (frozen while refractory)

A presynaptic spike adds ``w`` to ``g`` of the postsynaptic neuron after a fixed
transmission delay; ``w`` is the signed number of anatomical synapses times
``w_synapse``. Crossing ``v_threshold`` emits a spike and resets ``v`` and ``g``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .params import LIFParams, PoissonDrive

logger = logging.getLogger(__name__)

StepCallback = Callable[["LIFNetwork", int, np.ndarray], None]
"""Called after every step as ``callback(network, step, spiking_indices)``."""


@dataclass(frozen=True, slots=True)
class SpikeTrains:
    """Spikes of one simulation run.

    Attributes
    ----------
    neuron:
        Index of the spiking neuron, one entry per spike.
    time:
        Spike time in seconds, aligned with ``neuron``.
    n_neurons:
        Size of the simulated network, needed to compute rates.
    duration:
        Simulated time in seconds.
    """

    neuron: np.ndarray
    time: np.ndarray
    n_neurons: int
    duration: float

    def rates(self) -> np.ndarray:
        """Mean firing rate per neuron in hertz."""
        counts = np.bincount(self.neuron, minlength=self.n_neurons)
        return counts / self.duration

    def times_of(self, index: int) -> np.ndarray:
        """Spike times of a single neuron in seconds."""
        return self.time[self.neuron == index]


class LIFNetwork:
    """Leaky integrate-and-fire network on a signed connectivity matrix.

    Parameters
    ----------
    weights:
        Sparse matrix of signed synapse counts with ``weights[post, pre]``
        entries. It is converted to a presynapse-major CSR layout internally;
        the per-connection voltage increment is ``weights * params.w_synapse``.
    params:
        Fast-layer parameters.
    rng:
        Random generator used for the Poisson drives.

    Notes
    -----
    The state vectors (``v``, ``g``) are updated in place: they are the hot loop
    of the simulation, and copying 140k-element arrays 10,000 times per
    simulated second would dominate the runtime. Everything that describes the
    *model* (parameters, connectivity, cell-type tables) is immutable.
    """

    def __init__(
        self,
        weights: sparse.spmatrix | sparse.sparray,
        params: LIFParams | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        params = params or LIFParams()
        if weights.shape[0] != weights.shape[1]:
            raise ValueError(f"connectivity must be square, got {weights.shape}")

        self.params = params
        self.n_neurons = int(weights.shape[0])
        self.rng = rng or np.random.default_rng()

        # presynapse-major layout: row i lists the targets of neuron i
        transposed = sparse.csr_matrix(weights).T.tocsr()
        transposed.sum_duplicates()
        self._indptr = transposed.indptr.astype(np.int64)
        self._targets = transposed.indices.astype(np.int64)
        self.synapse_weight = transposed.data.astype(np.float64) * params.w_synapse

        # exact propagator over one time step
        dt, tau_m, tau_s = params.dt, params.tau_membrane, params.tau_synapse
        self._decay_v = float(np.exp(-dt / tau_m))
        self._decay_g = float(np.exp(-dt / tau_s))
        self._g_to_v = float(tau_s / (tau_s - tau_m) * (self._decay_g - self._decay_v))

        self.v = np.full(self.n_neurons, params.v_rest, dtype=np.float64)
        self.g = np.zeros(self.n_neurons, dtype=np.float64)
        self._refractory_left = np.zeros(self.n_neurons, dtype=np.int32)
        # steps a neuron stays blocked *after* the step in which it spiked, so that it
        # may fire again exactly ``t_refractory`` later (Brian 2: t >= lastspike + rfc)
        self._refractory_steps = np.full(
            self.n_neurons, max(params.refractory_steps - 1, 0), dtype=np.int32
        )

        # ring buffer holding the delayed synaptic increments
        self._delay_buffer = np.zeros((params.delay_steps, self.n_neurons), dtype=np.float64)
        self._buffer_pos = 0
        self._step = 0

        self._poisson_targets = np.zeros(0, dtype=np.int64)
        self._poisson_lambda = np.zeros(0, dtype=np.float64)
        self._poisson_weight = np.zeros(0, dtype=np.float64)

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------
    def set_poisson_drives(self, drives: dict[int, PoissonDrive]) -> None:
        """Attach external Poisson drives (optogenetic-style activation).

        Driven neurons lose their refractory period, as in the reference model.
        Calling this replaces any previously attached drives.
        """
        self._refractory_left[:] = 0
        self._refractory_steps[:] = max(self.params.refractory_steps - 1, 0)

        targets = np.fromiter(drives.keys(), dtype=np.int64, count=len(drives))
        if targets.size and (targets.min() < 0 or targets.max() >= self.n_neurons):
            raise IndexError("Poisson target index outside the network")

        self._poisson_targets = targets
        self._poisson_lambda = np.array(
            [d.rate * self.params.dt for d in drives.values()], dtype=np.float64
        )
        self._poisson_weight = np.array(
            [d.weight(self.params) for d in drives.values()], dtype=np.float64
        )
        self._refractory_steps[targets] = 0

    def silence(self, indices: Sequence[int] | np.ndarray) -> None:
        """Set all outgoing weights of ``indices`` to zero (optogenetic silencing).

        Incoming connections are removed as well, mirroring the reference model,
        by zeroing the corresponding entries of the weight vector.
        """
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size == 0:
            return
        for i in indices:
            self.synapse_weight[self._indptr[i] : self._indptr[i + 1]] = 0.0
        self.synapse_weight[np.isin(self._targets, indices)] = 0.0

    # ------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------
    def reset_state(self) -> None:
        """Return membrane potentials, synaptic drive and buffers to rest."""
        self.v[:] = self.params.v_rest
        self.g[:] = 0.0
        self._refractory_left[:] = 0
        self._delay_buffer[:] = 0.0
        self._buffer_pos = 0
        self._step = 0

    def step(self) -> np.ndarray:
        """Advance the network by one ``dt`` and return the spiking indices."""
        active = self._refractory_left == 0

        # exact integration of the linear subthreshold dynamics
        v, g = self.v, self.g
        v[active] = (
            self.params.v_rest
            + (v[active] - self.params.v_rest) * self._decay_v
            + g[active] * self._g_to_v
        )
        g[active] *= self._decay_g
        np.subtract(
            self._refractory_left, 1, out=self._refractory_left, where=~active
        )

        # delayed synaptic input arriving in this step
        arriving = self._delay_buffer[self._buffer_pos]
        g += arriving
        arriving[:] = 0.0

        # external Poisson drive acts directly on the membrane potential
        if self._poisson_targets.size:
            events = self.rng.poisson(self._poisson_lambda)
            v[self._poisson_targets] += events * self._poisson_weight

        spiking = np.flatnonzero((v > self.params.v_threshold) & active)
        if spiking.size:
            v[spiking] = self.params.v_reset
            g[spiking] = 0.0
            self._refractory_left[spiking] = self._refractory_steps[spiking]
            self._schedule(spiking)

        self._buffer_pos = (self._buffer_pos + 1) % self._delay_buffer.shape[0]
        self._step += 1
        return spiking

    def outgoing_synapses(self, pre_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Locate the synapses leaving a set of neurons.

        Parameters
        ----------
        pre_indices:
            Presynaptic neuron indices.

        Returns
        -------
        tuple of numpy.ndarray
            ``(positions, presynaptic, postsynaptic)`` where ``positions`` index
            into :attr:`synapse_weight`. This is how the neuromodulation layer
            finds the synapses it is allowed to change.
        """
        pre_indices = np.asarray(pre_indices, dtype=np.int64)
        positions = self._flat_positions(pre_indices)
        counts = self._indptr[pre_indices + 1] - self._indptr[pre_indices]
        return positions, np.repeat(pre_indices, counts), self._targets[positions]

    def _flat_positions(self, pre_indices: np.ndarray) -> np.ndarray:
        """Positions in ``synapse_weight`` of all synapses leaving ``pre_indices``."""
        starts = self._indptr[pre_indices]
        counts = self._indptr[pre_indices + 1] - starts
        total = int(counts.sum())
        if total == 0:
            return np.zeros(0, dtype=np.int64)
        # flat gather indices for the concatenated CSR rows of all listed neurons
        offsets = np.repeat(starts - np.concatenate(([0], np.cumsum(counts)[:-1])), counts)
        return np.arange(total, dtype=np.int64) + offsets

    def _schedule(self, spiking: np.ndarray) -> None:
        """Add the postsynaptic increments of ``spiking`` to the delay buffer."""
        flat = self._flat_positions(spiking)
        if flat.size == 0:
            return
        # the slot just consumed in this step is read again exactly delay_steps later
        np.add.at(
            self._delay_buffer[self._buffer_pos], self._targets[flat], self.synapse_weight[flat]
        )

    def run(
        self,
        duration: float,
        callbacks: Iterable[StepCallback] = (),
        record: bool = True,
    ) -> SpikeTrains:
        """Simulate for ``duration`` seconds and return the spike trains.

        Parameters
        ----------
        duration:
            Simulated time in seconds.
        callbacks:
            Called after every step; this is how the neuromodulation layer reads
            spikes and writes back changes to synaptic weights.
        record:
            Set to ``False`` to skip spike recording (saves memory in long runs
            where a callback does the bookkeeping).
        """
        if duration <= 0:
            raise ValueError("duration must be positive")
        n_steps = int(round(duration / self.params.dt))
        callbacks = tuple(callbacks)

        neurons: list[np.ndarray] = []
        times: list[np.ndarray] = []
        for _ in range(n_steps):
            step_index = self._step
            spiking = self.step()
            if record and spiking.size:
                neurons.append(spiking)
                times.append(np.full(spiking.size, (step_index + 1) * self.params.dt))
            for callback in callbacks:
                callback(self, step_index, spiking)

        neuron_array = (
            np.concatenate(neurons) if neurons else np.zeros(0, dtype=np.int64)
        )
        time_array = np.concatenate(times) if times else np.zeros(0, dtype=np.float64)
        logger.debug(
            "simulated %.3f s, %d spikes, %d neurons", duration, neuron_array.size, self.n_neurons
        )
        return SpikeTrains(
            neuron=neuron_array,
            time=time_array,
            n_neurons=self.n_neurons,
            duration=n_steps * self.params.dt,
        )
