# Validation

What the model is checked against, and how it does. A claim like "the layer
reproduces fly learning" should be a table with numbers in it, not a sentence.

Fast checks: `uv run pytest`. Experiments on the full FlyWire v783 mushroom body
(5,608 neurons, 523,784 connections): `uv run pytest -m slow`, or
`uv run python scripts/validate_mushroom_body.py` for the report below (~5 min on
a laptop).

## Headline results on the real connectome

**One pairing in γ1pedc** — 5 s odour on 10% of Kenyon cells, PPL1-γ1pedc driven
at 20 Hz from 0.2 s.

| Quantity | Fly (Hige et al. 2015) | Model |
| --- | --- | --- |
| Weight of the trained odour's synapses | about −90% | **−90%** (calibrated) |
| Weight of all γ1pedc synapses | — | −9% (only the trained odour's cells change) |
| Same pairing, Dop1R1 null | learning abolished | **+0%** |
| Trained odour response of MBON-γ1pedc | about −80% | falls below spontaneous (103.5 → 0.0 Hz, −141% of the evoked response) |
| Control odour response (20% overlap) | about −25% | 94.5 → 57.0 Hz (−58%) |

**The interval curve in γ5** — PAM-γ5 driven for 1 s at intervals from a 2 s
odour, as in Handler et al. 2019.

| Dopamine onset relative to odour | Fly | Model |
| --- | --- | --- |
| −1.2 s (dopamine first) | potentiation | **+14.6%** |
| −0.5 s | near the crossover | **−9.7%** |
| 0 s | depression | **−16.9%** |
| +0.5 s | depression | **−14.7%** |
| +6 s | no change | **+3.0%** |

The sign flip is not fitted. The potentiation rate sets *where* it happens, but
no value of it can make a symmetric rule flip; the flip comes from the order
preferences of the two coincidence detectors.

## Passing (fast suite)

| Test | Expectation | Source |
| --- | --- | --- |
| Engine equivalence | Spike-for-spike identity with the Brian 2 reference model | Shiu et al. 2024 |
| Dopamine transient | A compartment's population at 20 Hz holds 0.3–0.5 µM | Shin & Venton 2022 |
| Clearance | Half-decay 1.4–2.7 s | Shin & Venton 2022 |
| Autoreceptor feedback | Presynaptic Dop2R autoinhibition suppresses release by ~3–4x; Flupentixol block restores unsuppressed level | Shin & Venton 2022 |
| Transporter block | Slower clearance; a transporter null is slower still but still clears | Makos et al. 2010 |
| Coincidence requirement | Dopamine alone and odour alone change nothing | Hige et al. 2015 |
| Order | Calcium then Gs depresses; IP₃ then calcium potentiates; Gs already present when a cell starts firing does not depress | Yovell & Abrams 1992; Bezprozvanny et al. 1991 |
| Step independence | The amount learned does not depend on the integration step | — |
| Cell, compartment and hemisphere specificity | Only the cells, compartment and side that were paired change | Hige et al. 2015; Aso et al. 2014 |
| γ1pedc exception | No potentiation on backward pairing | Hige et al. 2015 |
| Receptor null, antagonist | Null abolishes depression; a competitive antagonist weakens it | Handler et al. 2019 |
| Protocol timeline | Every interval delivered exactly, including gaps and simultaneous onset | — |
| Atlas | Every type exists in FlyWire v783; every curated dopaminergic type is accounted for | Schlegel et al. 2024 |

## Calibrated, not predicted

| Quantity | Target | Fitted parameter |
| --- | --- | --- |
| Peak dopamine | 0.35–0.4 µM at 20 Hz (with autoreceptor active) | release per spike |
| Autoreceptor feedback | ~4-fold rise in evoked dopamine upon Dop2R block | Dop2R EC50 (0.20 µM) and max_suppression (0.90) |
| Half-decay | ~2 s | transporter V_max and diffusion rate |
| Synaptic depression after one pairing | −90% | depression rate |
| Position of the sign flip | between −1.2 and 0 s | ratio of potentiation to depression rate |
| Output neuron spontaneous rate | 30 Hz | tonic drive (stands in for inputs outside the mushroom body) |

## Not reproduced yet

| Quantity | Why |
| --- | --- |
| Size of the output neuron's response change | The isolated mushroom body leaves its output neuron without the rest of its inputs. Even held at a fitted 30 Hz, it loses more spikes for a given loss of synaptic drive than the fly's does, so the trained odour silences it and the control odour loses 58% rather than 25%. The synaptic readout is the reliable one. |
| MBON-γ5 odour response | In the subnetwork the odour does not raise MBON-γ5 above its spontaneous rate, so only the synaptic change is reported for γ5. |
| Exact crossover of the timing curve | The model flips between −1.2 and −0.5 s; Handler et al. place it between −0.5 and 0 s. |
| Kenyon cell firing rates | The circuit needs tens of hertz of drive; APL and DPM are graded neurons that a spiking model makes fire at hundreds of hertz. |
| Compartment-specific rates and retention | Measured only for γ1, γ4 and γ5; one rate is used everywhere. |

## A finding along the way

During the odour, Kenyon cells excite PPL1-γ1pedc through about 14,000 synapses.
A 20 Hz optogenetic-style drive becomes ~47 Hz of firing and ~0.9 µM of
dopamine. This is the reciprocal Kenyon cell → dopaminergic neuron loop that
Cervantes-Sandoval et al. (2017) showed is needed for learning; it appears in
the model from the connectome alone. It is also why the learning rate had to be
calibrated in the network rather than on an isolated compartment.
