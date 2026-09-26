# Engineering Design Specification: Excitability Effector for Neuromodulation

> **Decision (supersedes the recommendation in section 1.4).** The effector is a
> potassium *conductance*, option A, not the tonic current of option C. Pimentel
> et al. (2016) measured a rise in a voltage-independent potassium leak (Sandman).
> A conductance both pulls the membrane toward E_K and lowers the input
> resistance, so the same synaptic drive moves the membrane (1 + k) times less; a
> current only shifts the membrane and leaves its response to input unchanged.
> Implemented as `LIFNetwork.set_potassium_conductance`, with per-neuron
> propagators that are bit-for-bit identical to the old ones when k = 0.

## Overview & Context

Currently, the dopamine layer ([dopamine.py](../src/flyneuromod/neuromod/dopamine.py#L1-L24)) has a single effector: it modifies fast synaptic weights (`synapse_weight` in [lif.py](../src/flyneuromod/engine/lif.py#L108)) via order-dependent plasticity at Kenyon cell to MBON terminals ([plasticity.py](../src/flyneuromod/neuromod/plasticity.py#L1-L31)). 

Biologically, neuromodulators such as dopamine also regulate intrinsic neuronal excitability. In the dorsal fan-shaped body (dFSB) of *Drosophila*, dopamine acting on $\text{G}_i$-coupled receptors switches sleep-control neurons from active firing to electrical silence (Pimentel et al. 2016, Nature 536:333).

This document outlines the software engineering design for incorporating a time-varying, per-neuron intrinsic excitability effector into `fly-neuromod` without compromising performance, step-independence, or breaking existing validation benchmarks.

---

## 1. The Crux: LIF Exact Propagators and State Invalidation

### 1.1 Current Architecture of `LIFNetwork`
The fast spiking engine [lif.py](../src/flyneuromod/engine/lif.py#L1-L17) integrates linear subthreshold dynamics using an exact propagator. In `LIFParams` ([params.py](../src/flyneuromod/engine/params.py#L50-L58)), all membrane parameters are global scalars:
- `v_rest = -52.0 mV` ([params.py](../src/flyneuromod/engine/params.py#L50))
- `v_reset = -52.0 mV` ([params.py](../src/flyneuromod/engine/params.py#L51))
- `v_threshold = -45.0 mV` ([params.py](../src/flyneuromod/engine/params.py#L52))
- `tau_membrane = 20.0 ms` ([params.py](../src/flyneuromod/engine/params.py#L53))
- `tau_synapse = 5.0 ms` ([params.py](../src/flyneuromod/engine/params.py#L54))

During initialization, `LIFNetwork` precomputes exact integration constants as global Python `float` scalars ([lif.py](../src/flyneuromod/engine/lif.py#L110-L114)):
```python
# src/flyneuromod/engine/lif.py lines 110-114
dt, tau_m, tau_s = params.dt, params.tau_membrane, params.tau_synapse
self._decay_v = float(np.exp(-dt / tau_m))
self._decay_g = float(np.exp(-dt / tau_s))
self._g_to_v = float(tau_s / (tau_s - tau_m) * (self._decay_g - self._decay_v))
```

In the fast simulation loop ([lif.py](../src/flyneuromod/engine/lif.py#L246-L250)), subthreshold updates perform scalar-array multiplications:
```python
# src/flyneuromod/engine/lif.py lines 246-250
v[active] = (
    self.params.v_rest
    + (v[active] - self.params.v_rest) * self._decay_v
    + g[active] * self._g_to_v
)
g[active] *= self._decay_g
```

### 1.2 Comparison of Excitability Implementation Mechanisms

An excitability effector must modulate how readily target neurons fire in response to synaptic input. We evaluate three primary mathematical abstractions:

| Mechanism | Biophysical Basis | Codebase Impact & Precomputed Arrays | Performance & Recomputation Overhead |
| :--- | :--- | :--- | :--- |
| **Option A: Membrane Leak ($\tau_m$)** | $\text{G}_i$-coupled GIRK channel opening increases membrane leak conductance $g_{\text{leak}}$, reducing input resistance $R_m$ and membrane time constant $\tau_m = C_m / g_{\text{leak}}$. | **Breaks global scalar decay constants.** `self._decay_v` ([lif.py#L112](../src/flyneuromod/engine/lif.py#L112)) and `self._g_to_v` ([lif.py#L114](../src/flyneuromod/engine/lif.py#L114)) become 1D `float64` arrays of shape `(n_neurons,)`. | Recomputes `_decay_v` and `_g_to_v` per slow step for target neurons. Fast step loop requires vector-vector updates `* self._decay_v[active]`. |
| **Option B: Resting Potential ($v_{\text{rest}}$)** | $\text{G}_i$-coupled activation hyperpolarizes the reversal potential towards the potassium equilibrium potential $E_K$. | `_decay_v` and `_g_to_v` **remain global scalars**. `self.v_rest` becomes a per-neuron 1D array (`shape=(n_neurons,)`) or `self.params.v_rest` is offset by a per-neuron array $\Delta v_{\text{rest}}$. | Zero recomputation of exact propagators. Fast step uses `self.v_rest[active]` in line 247. |
| **Option C: Tonic Drive / Hyperpolarizing Current ($I_{\text{tonic}}$)** | $\text{G}_i$ activation drives a sustained outward potassium current $I_{\text{GIRK}}$. | **No propagator changes whatsoever.** Modulates `self.tonic_drive` ([lif.py#L136](../src/flyneuromod/engine/lif.py#L136)), which already exists as a per-neuron array `self.tonic_drive` and is applied in line 260 ([lif.py#L260](../src/flyneuromod/engine/lif.py#L260)). | **Zero additional overhead.** Modulates existing array `self.tonic_drive` during the slow step update. |

### 1.3 Detailed Analysis of Option A (Membrane Leak $\tau_m$)

If biophysical accuracy demands modulating $\tau_m$ directly:
1. **What breaks:**
   - `self._decay_v` ([lif.py#L112](../src/flyneuromod/engine/lif.py#L112)) can no longer be a scalar `float`.
   - `self._g_to_v` ([lif.py#L114](../src/flyneuromod/engine/lif.py#L114)) can no longer be a scalar `float`.
   - Singularity check `tau_membrane == tau_synapse` ([params.py#L63-L67](../src/flyneuromod/engine/params.py#L63-L67)) must be guarded per neuron to avoid division by zero in `tau_s / (tau_s - tau_m)`.
2. **What must be recomputed per slow step:**
   - On each slow step (`_advance_slow_layer()` in [dopamine.py#L411](../src/flyneuromod/neuromod/dopamine.py#L411)), updated $\tau_m[i]$ values for target neurons require re-evaluating:
     $$\text{\_decay\_v}[i] = \exp\left(-\frac{\Delta t}{\tau_m[i]}\right)$$
     $$\text{\_g\_to\_v}[i] = \frac{\tau_s}{\tau_s - \tau_m[i]} \left( \text{\_decay\_g} - \text{\_decay\_v}[i] \right)$$
3. **Computational Cost for Whole-Brain Model (138,639 Neurons):**
   - **Dense update (all 138,639 neurons):** Evaluating `np.exp()` and vector arithmetic over 138,639 elements per slow step (`slow_dt = 1e-3` s, [dopamine.py#L89](../src/flyneuromod/neuromod/dopamine.py#L89)) requires ~0.2–0.5 ms per slow step (guess). Over 1 simulated second (1,000 slow steps), this introduces ~200–500 ms of CPU overhead.
   - **Sparse update (target neurons only, e.g., ~90 dFSB cells):** Updating a sliced array of 90 elements takes < 1 microsecond per slow step, which is negligible.

### 1.4 Recommendation
- **Primary Recommendation (Option C / B):** Modulating `tonic_drive` ([lif.py#L136](../src/flyneuromod/engine/lif.py#L136)) or `v_rest` ([lif.py#L116](../src/flyneuromod/engine/lif.py#L116)) delivers the exact electrical silencing observed by Pimentel et al. (2016) while leaving exact propagator scalars `_decay_v` and `_g_to_v` ([lif.py#L112-L114](../src/flyneuromod/engine/lif.py#L112-L114)) 100% untouched.
- **Alternative (Option A):** If $\tau_m$ shunting must be modeled, `_decay_v` and `_g_to_v` should be maintained as 1D arrays, with sparse updates restricted strictly to target neuron indices.

---

## 2. API Design & Architecture

The API follows the established pattern of frozen parameter dataclasses and decoupled effector objects (like `PlasticityParams` and `SynapticPlasticity` in [plasticity.py#L45-L150](../src/flyneuromod/neuromod/plasticity.py#L45-L150)).

### 2.1 Parameter Dataclass (`ExcitabilityParams`)
```python
# src/flyneuromod/neuromod/excitability.py

from dataclasses import dataclass, replace
from typing import Any

@dataclass(frozen=True, slots=True)
class ExcitabilityParams:
    """Parameters for receptor-driven intrinsic excitability modulation.
    
    Attributes
    ----------
    target_cell_types:
        Tuple of cell-type names (e.g. ("FB6A", "FB6B", ...)) whose excitability is modulated.
    coupling:
        Receptor coupling type driving the effect (default "Gi").
    max_delta_tonic:
        Maximum negative tonic drive shift (in V/s) at 100% receptor occupancy.
    tau_effector:
        Relaxation time constant of the effector activation (seconds).
    """
    target_cell_types: tuple[str, ...]
    coupling: str = "Gi"
    max_delta_tonic: float = -50.0  # V/s (guess, calibrated to silence target neurons)
    tau_effector: float = 0.5        # seconds (guess)

    def __post_init__(self) -> None:
        if not self.target_cell_types:
            raise ValueError("target_cell_types cannot be empty")
        if self.tau_effector <= 0:
            raise ValueError("tau_effector must be positive")

    def evolve(self, **changes: Any) -> ExcitabilityParams:
        return replace(self, **changes)
```

### 2.2 Effector Object (`ExcitabilityEffector`)
```python
# src/flyneuromod/neuromod/excitability.py

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..engine.lif import LIFNetwork

class ExcitabilityEffector:
    """Manages time-varying intrinsic excitability changes on target neurons."""

    def __init__(
        self,
        network: LIFNetwork,
        target_indices: np.ndarray,
        target_fields: np.ndarray,
        params: ExcitabilityParams,
        dt: float,
    ) -> None:
        self.network = network
        self.target_indices = np.asarray(target_indices, dtype=np.int64)
        self.target_fields = np.asarray(target_fields, dtype=np.int64)
        self.params = params
        self.dt = float(dt)

        self.activation = np.zeros(len(self.target_indices), dtype=np.float64)
        self._decay = float(np.exp(-dt / params.tau_effector))
        self._baseline_tonic = self.network.tonic_drive[self.target_indices].copy()

    def reset(self) -> None:
        """Reset effector state and restore baseline network excitability."""
        self.activation[:] = 0.0
        self.network.tonic_drive[self.target_indices] = self._baseline_tonic

    def step(self, receptor_occupancy: np.ndarray) -> None:
        """Advance effector state by one slow step dt given receptor occupancy per field."""
        # Map field occupancy to target neurons
        target_occ = receptor_occupancy[self.target_fields]
        
        # Exact exponential relaxation update towards receptor occupancy target
        self.activation[:] = target_occ + (self.activation - target_occ) * self._decay
        
        # Apply hyperpolarizing drive to network tonic_drive
        delta_drive = self.params.max_delta_tonic * self.activation
        self.network.tonic_drive[self.target_indices] = self._baseline_tonic + delta_drive
```

### 2.3 Integration into `DopamineConfig` and `DopamineLayer`

1. **`DopamineConfig` ([dopamine.py#L62-L96](../src/flyneuromod/neuromod/dopamine.py#L62-L96)):**
   Add an optional `excitability` attribute:
   ```python
   # In DopamineConfig (src/flyneuromod/neuromod/dopamine.py)
   excitability: ExcitabilityParams | None = None
   ```

2. **`DopamineTargets` ([dopamine.py#L143-L174](../src/flyneuromod/neuromod/dopamine.py#L143-L174)):**
   Add fields for excitability targets:
   ```python
   excitability_target_index: np.ndarray
   excitability_target_field: np.ndarray
   ```

3. **`DopamineLayer._advance_slow_layer()` ([dopamine.py#L411-L438](../src/flyneuromod/neuromod/dopamine.py#L411-L438)):**
   Extract $\text{G}_i$ receptor occupancy and drive the effector:
   ```python
   # In DopamineLayer._advance_slow_layer()
   gi_occ = self._sum_coupling(occupancy, "Gi")
   if self.excitability is not None:
       self.excitability.step(gi_occ)
   ```

---

## 3. Preservation of Zero-Target No-Op (Bit-for-Bit Determinism)

A critical requirement is that configuring no excitability targets must remain a strict no-op, preserving 100% bit-for-bit equivalence for all existing mushroom body validation tests ([validation.md#L11-L56](../docs/validation.md#L11-L56)).

### Guarantee Mechanism:
1. **Default Configuration:** In `DopamineConfig` ([dopamine.py#L62](../src/flyneuromod/neuromod/dopamine.py#L62)), `excitability: ExcitabilityParams | None = None` defaults to `None`.
2. **Layer Initialization:** In `DopamineLayer._build_state()` ([dopamine.py#L246](../src/flyneuromod/neuromod/dopamine.py#L246)), if `config.excitability is None` or `len(targets.excitability_target_index) == 0`, `self.excitability` is set to `None`.
3. **Execution Path:** In `_advance_slow_layer()` ([dopamine.py#L411](../src/flyneuromod/neuromod/dopamine.py#L411)), the `if self.excitability is not None:` block is bypassed entirely.
4. **Engine Isolation:** `LIFNetwork` ([lif.py#L67](../src/flyneuromod/engine/lif.py#L67)) incurs zero modifications to its precomputed propagators (`_decay_v`, `_decay_g`, `_g_to_v` at lines 112–114) or fast loop execution path ([lif.py#L240-L276](../src/flyneuromod/engine/lif.py#L240-L276)).

Thus, for all existing simulation runs, memory layouts, random numbers, and arithmetic operations remain identical to the current baseline.

---

## 4. Integration Step Independence

To ensure that simulation results do not depend on the choice of integration time step (`slow_dt`), the excitability state update is formulated using exact subthreshold propagator dynamics, exactly as done in `ReceptorPopulation` ([receptors.py#L149-L156](../src/flyneuromod/neuromod/receptors.py#L149-L156)) and `SecondMessenger` ([receptors.py#L217-L225](../src/flyneuromod/neuromod/receptors.py#L217-L225)).

### Mathematical Formulation
For target activation $A(t)$ relaxing towards steady-state receptor drive $O_{\text{Gi}}(t)$ with time constant $\tau_e$:
$$\frac{dA}{dt} = \frac{O_{\text{Gi}}(t) - A(t)}{\tau_e}$$

Over a slow step of length $\Delta t = \text{slow\_dt}$, the exact solution is:
$$A(t + \Delta t) = O_{\text{Gi}}(t) + \left( A(t) - O_{\text{Gi}}(t) \right) e^{-\frac{\Delta t}{\tau_e}}$$

Because $e^{-\Delta t / \tau_e}$ is an exact exponential propagator (computed as `self._decay = float(np.exp(-dt / params.tau_effector))`), stepping by $\Delta t$ twice produces the exact same numerical result as stepping by $2\Delta t$ once:
$$e^{-\frac{\Delta t}{\tau_e}} \cdot e^{-\frac{\Delta t}{\tau_e}} = e^{-\frac{2\Delta t}{\tau_e}}$$

This guarantees that the continuous trajectory $A(t)$ is independent of `slow_dt` without Euler truncation errors.

---

## 5. Fan-Shaped Body Types and Subnetwork Integration

### 5.1 FlyWire v783 FB Cell Types
As identified in FlyWire v783 annotations (Schlegel et al. 2024), the dorsal fan-shaped body contains approximately 90 cells across layer 6 and 7 cell types:
- **FB6 Types:** `FB6A` through `FB6Z`, `FB6A_c`, `FB6`, and `FB6J`.
- **FB7 Types:** `FB7A` through `FB7C` and beyond.
- Cell count per type ranges from 1 to 9 cells.

### 5.2 Transmitter Mix & Subnetwork Connectivity
The predicted primary neurotransmitters for these types are:
- **Glutamate:** Predicted for almost all FB6 types (e.g., `FB6A`..`FB6L`, `FB6O`..`FB6Z`).
- **GABA:** Predicted specifically for `FB6M`.
- **Serotonin (5-HT):** Predicted specifically for `FB6N`.

### 5.3 Engineering Implications for `LIFNetwork` Integration

1. **Signed Connectivity Matrix Signs (`weights`):**
   - In `LIFNetwork` ([lif.py#L72-L108](../src/flyneuromod/engine/lif.py#L72-L108)), synapses are represented in a signed CSR matrix where the sign determines excitatory versus inhibitory action:
     - **GABA (`FB6M`):** Unambiguously inhibitory (-1 sign in `weights`). Silencing `FB6M` via dopamine $\text{G}_i$ activation causes **disinhibition** of downstream targets.
     - **Glutamate (`FB6A`..`FB6Z` except M/N):** In *Drosophila*, glutamate acts via ionotropic AMPA/kainate receptors (excitatory, +1) or inhibitory GluCl channels (inhibitory, -1). The connectome compiler assigns signs based on postsynaptic receptor annotations. Silencing excitatory glutamatergic FB6 neurons removes tonic excitatory drive to downstream arousal centers.
     - **Serotonin (`FB6N`):** Serotonin acts primarily via metabotropic receptors. In the fast spiking `LIFNetwork`, fast serotonergic synapses should either be assigned an explicit fast transmitter sign or zeroed via an option analogous to `remove_fast_dopamine_synapses` ([dopamine.py#L80](../src/flyneuromod/neuromod/dopamine.py#L80)), allowing slow serotonergic modulatory layers to handle them separately.

2. **Subnetwork Calibration & Tonic Drive (`tonic_drive`):**
   - Extracting a subnetwork around dFSB neurons removes inputs from the rest of the brain. As documented in `LIFNetwork.set_tonic_drive()` ([lif.py#L201-L212](../src/flyneuromod/engine/lif.py#L201-L212)), a baseline calibration `tonic_drive` must be set for FB6/FB7 neurons to maintain physiological baseline firing rates (e.g., 5–15 Hz, guess) prior to dopamine release.
   - When dopamine is released (e.g. from PPL1 or PPL2 DANs innervating the dFSB), $\text{G}_i$ receptor activation drives `ExcitabilityEffector`, suppressing `tonic_drive` and pulling FB6 neurons into electrical silence.
