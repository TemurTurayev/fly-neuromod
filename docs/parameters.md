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
| Dop2R EC50 | 0.20 µM | **calibrated** | Hearn et al. report only a rank order; 0.20 µM calibrated so presynaptic autoreceptor feedback suppresses steady-state release by ~3-4x |
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
| Transporter V_max | 0.30 µM/s | **calibrated** | set so that uptake plus diffusion reproduce the adult half-decay. The measured larval V_max of 0.11 µM/s alone gives ~12 s, far slower than the adult brain shows |
| Diffusion out of the compartment | 0.12 s⁻¹ | **assumed** | uptake alone saturates at V_max, so a compartment driven harder than that would accumulate dopamine without bound. Transporter-null flies still clear dopamine (Makos et al. 2010), so this term cannot be zero; the split between uptake and diffusion is not measured |
| Release per spike | 0.0145 µM | **calibrated** | so that 20 Hz firing of a compartment's dopaminergic population holds ~0.35-0.4 µM with the presynaptic Dop2R autoreceptor active (only meaningful with the autoreceptor active) |
| Dop2R max_suppression | 0.90 | **calibrated** | reproduces the ~4-fold rise in evoked dopamine when Dop2R is blocked by flupentixol (Shin & Venton 2022) |
| Dop2R min_gain | 0.1 | **assumed** | prevents total shutoff of release under strong stimulation |
| Release per neuron | share of the compartment's synapses onto Kenyon cells | measured (anatomy) | from the connectome; the shares within a compartment sum to one, so the calibration applies to compartments rather than to a single neuron |

## Intracellular signalling

| Parameter | Value | Confidence | Source |
| --- | --- | --- | --- |
| cAMP time constant | 4 s | assumed/calibrated | cAMP imaging evolves over tens of seconds (Tomchik & Davis 2009, *Neuron* 64:510); pairing experiments average over ~4 s windows (Handler et al. 2019) |
| Calcium time constant | 1 s | assumed | the Gq/IP3 branch is stimulus-locked and faster than cAMP |
| cAMP and calcium units | normalised | — | absolute concentrations in fly neurons are unknown; all imaging is a sensor ratio |

## Plasticity

The rule is two order-selective coincidence detectors, not a product of
"dopamine × activity". A symmetric product was scanned over receptor affinities,
kinetics and rate ratios and never produced the measured sign flip: dopamine
lingers for seconds, so both orders look like overlap to it.

| Parameter | Value | Confidence | Source |
| --- | --- | --- | --- |
| Depression detector | Gs activation arriving onto a primed (recently active) terminal | measured (biochemistry) | the Ca²⁺/calmodulin cyclase responds more when calcium precedes the transmitter: Yovell & Abrams 1992, *PNAS* 89:6526; rutabaga is that cyclase in the fly: Levin et al. 1992, *Cell* 68:479 |
| Potentiation detector | calcium arriving onto IP₃ that is already present, scaled by the receptors not yet inhibited by calcium | measured (biochemistry) | IP₃ receptors need IP₃ bound before calcium and have a bell-shaped calcium dependence: Bezprozvanny et al. 1991, *Nature* 351:751; the Dop1R2/Gq/ER-calcium route: Handler et al. 2019 |
| Calcium (eligibility) time constant | 2 s | measured (window) | pairing works at 0.1–1 s and is gone by ~6 s: Handler et al. 2019, *Cell* 178:60 |
| IP₃ signal time constant | 1 s | assumed | the Gq branch is stimulus-locked and faster than cAMP |
| Depression rate | 1.5 per unit of arriving Gs activation | **calibrated** | one pairing removes ~90% (-89.8%) of the trained synapses' weight on the full mushroom body: Hige et al. 2015, *Neuron* 88:985. Calibrated in the network because Kenyon cells excite the dopaminergic neuron during the odour (Cervantes-Sandoval et al. 2017), more than doubling its firing |
| Potentiation rate | 0.6 (0.4 × depression) | **calibrated (sign only)** | rescaled alongside depression rate to preserve the 0.4 ratio that sets where the sign flips: dopamine 1.2 s before the odour potentiates, 0.5 s after depresses (Handler et al. 2019). The size of potentiation is not measured |
| No potentiation in γ1pedc | backward pairing produced no change | measured | Hige et al. 2015 |
| Control-odour loss | ~25% | measured | Hige et al. 2015; caused by shared Kenyon cells |
| Weight bounds | 0 to 1.5 × anatomical | assumed | physiological bounds are unmeasured |
| Passive forgetting | off | measured (as a choice) | forgetting in the fly is dopamine-driven through Dop1R2: Berry et al. 2012, *Neuron* 74:530 |
| Hemispheres | separate dopamine fields | anatomy | the left and right mushroom bodies are separate volumes |

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
