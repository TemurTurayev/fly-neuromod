# Parameters and where they come from

Every number in `src/flyneuromod/neuromod/constants.py` appears here with its
source and a confidence label.

| Label | Meaning |
| --- | --- |
| **measured** | Measured in *Drosophila*. |
| **calibrated** | Fitted so the model reproduces a measured observable, which is named. |
| **assumed** | No fly measurement exists. A placeholder with reasoning; treat as a free parameter. |

## Fast spiking layer

Taken unchanged from the reference model so that results stay comparable to it:
Shiu et al. 2024, *Nature* 634:210, doi:10.1038/s41586-024-07763-9.

| Parameter | Value | Confidence | Source |
| --- | --- | --- | --- |
| Resting / reset potential | −52 mV | measured | Kakaria & de Bivort 2017, doi:10.3389/fnbeh.2017.00008 |
| Spike threshold | −45 mV | measured | same |
| Membrane time constant | 20 ms | measured | same (10 MΩ × 2 nF) |
| Synaptic time constant | 5 ms | measured | Jürgensen et al. 2021, doi:10.1088/2634-4386/ac3ba6 |
| Refractory period | 2.2 ms | measured | Lazar et al. 2021, doi:10.7554/eLife.62362 |
| Transmission delay | 1.8 ms | measured | Paul et al. 2015, doi:10.3389/fncel.2015.00029 |
| Weight per synapse | 0.275 mV | calibrated | free parameter of the reference model |

## Dopamine receptors

| Parameter | Value | Confidence | Source |
| --- | --- | --- | --- |
| Dop1R1 coupling | Gs | measured | Himmelreich et al. 2017, *Cell Rep* 21:2074, doi:10.1016/j.celrep.2017.10.104 |
| Dop1R1 EC50 | 0.61 µM | measured | same (BRET, Gs activation). A cAMP readout gives ~0.3 µM: Sugamori et al. 1995, *FEBS Lett* 362:131 |
| Dop1R1 activation rate | 0.46 s⁻¹ → τ 2.2 s | measured | Himmelreich et al. 2017 |
| Dop1R1 deactivation | τ 5 s | assumed | no fly measurement |
| Dop1R2 coupling | Gq ≫ Gs; Gi/o in sleep neurons | measured | Himmelreich et al. 2017; Pimentel et al. 2016, *Nature* 536:333 |
| Dop1R2 EC50 (Gq) | 0.057 µM | measured | Himmelreich et al. 2017 |
| Dop1R2 activation rate | 2.71 s⁻¹ → τ 0.37 s | measured | same |
| Dop2R coupling | Gi/o | measured | Hearn et al. 2002, *PNAS* 99:14554 |
| Dop2R EC50 | 0.5 µM | **assumed** | Hearn et al. report only a rank order; 0.1–1 µM is the defensible range |
| Hill coefficients | 1.0 | assumed | no cooperativity reported |

Dop2R is **not** placed on Kenyon cell terminals in the default configuration.
Its documented role in the mushroom body is autoinhibition of release by the
dopaminergic neurons themselves. Adding it postsynaptically with an arbitrary
gain simply cancels the Gs branch, which silently removes learning — a mistake
worth naming, because it is easy to make.

## Release and clearance

| Parameter | Value | Confidence | Source |
| --- | --- | --- | --- |
| Peak dopamine in a compartment | 0.3–0.5 µM | measured | Shin & Venton 2022, *Angew Chem* 61:e202207399 (feeding and cholinergic stimulation, adult mushroom body) |
| Half-decay of the transient | 1.4–2.7 s | measured | same |
| Transporter K_m | 1.3 µM | measured | Vickrey et al. 2013, *ACS Chem Neurosci* 4:832 (larval, diffusion-corrected). No adult value exists |
| Transporter V_max | 0.45 µM/s | **calibrated** | set so V_max/K_m reproduces the adult half-decay. The measured larval V_max of 0.11 µM/s gives ~12 s, far slower than the adult brain shows |
| Release per spike | 0.0053 µM | **calibrated** | so that 20 Hz firing of one typical neuron holds ~0.4 µM |
| Release per neuron | scaled by synapse count | measured (anatomy) | number of synapses onto Kenyon cells in the compartment, from the connectome, normalised to the median |

## Intracellular signalling

| Parameter | Value | Confidence | Source |
| --- | --- | --- | --- |
| cAMP time constant | 4 s | assumed/calibrated | cAMP imaging evolves over tens of seconds (Tomchik & Davis 2009, *Neuron* 64:510); pairing experiments average over ~4 s windows (Handler et al. 2019) |
| Calcium time constant | 1 s | assumed | the Gq/IP3 branch is stimulus-locked and faster than cAMP |
| cAMP and calcium units | normalised | — | absolute concentrations in fly neurons are unknown; all imaging is a sensor ratio |

## Plasticity

| Parameter | Value | Confidence | Source |
| --- | --- | --- | --- |
| Eligibility time constant | 2 s | measured (window) | pairing works at 0.1–1 s and is gone by ~6 s: Handler et al. 2019, *Cell* 178:60. τ = 2 s leaves 5% of the trace at 6 s |
| Sign depends on order | forward → depression, backward → potentiation | measured | Handler et al. 2019 (γ4, γ5) |
| No potentiation in γ1pedc | backward pairing produced no change | measured | Hige et al. 2015, *Neuron* 88:985 |
| Depression magnitude target | 90 ± 4% of the response, one pairing | measured | Hige et al. 2015 |
| Control-odour loss | ~25% | measured | same; caused by shared Kenyon cells, not by an unspecific rule |
| Depression persistence | ≥ 40 min | measured | same |
| Depression / potentiation rates | calibrated | **calibrated** | fitted to the two numbers above in `experiments/conditioning.py` |
| Weight bounds | 0 to 1.5 × anatomical | assumed | physiological bounds are unmeasured |
| Passive forgetting | off | measured (as a choice) | forgetting in the fly is dopamine-driven through Dop1R2, not passive decay: Berry et al. 2012, *Neuron* 74:530 |

## Known unknowns

These are free parameters, not facts. They are the honest limits of the layer:

- Quantal size, release probability and number of vesicles at dopaminergic terminals.
- Adult transporter kinetics (only larval measurements exist).
- Extracellular volume fraction and tortuosity of fly neuropil.
- Receptor density per cell type; only relative fluorescence measurements exist (Kudo et al. 2025, *eLife* 14:RP98358).
- Receptor desensitisation and internalisation kinetics.
- The sign of the Dop1R2 effect outside the mushroom body — it is Gq in Kenyon cells and Gi/o in sleep-control neurons, so it cannot be given a global sign.
- Plasticity magnitudes in all compartments except γ1 (electrophysiology) and γ4/γ5 (imaging).
- Postsynaptic plasticity of acetylcholine receptors on output neurons (Pribbenow et al. 2022, *eLife* 11:e80445) — real, and not modelled here.
