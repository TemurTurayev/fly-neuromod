# Validation

What the model is checked against, and how it currently does. The point of this
file is that a claim like "the layer reproduces fly learning" should be a table
with numbers in it, not a sentence.

Run the fast checks with `uv run pytest`; the ones that simulate the mushroom
body are marked `slow` and run with `uv run pytest -m slow`.

## Passing

| Test | Expectation | Source |
| --- | --- | --- |
| Engine equivalence | Spike-for-spike identity with the Brian 2 reference model on a random network | Shiu et al. 2024 |
| Dopamine transient | 20 Hz of a compartment's dopaminergic population holds 0.3–0.5 µM | Shiu & Venton 2022 |
| Clearance | Half-decay of a transient is 1.4–2.7 s | Shin & Venton 2022 |
| Transporter block | Cocaine-like block prolongs the transient; a transporter null is slower still but still clears | Makos et al. 2010 |
| Coincidence requirement | Dopamine alone and odour alone leave weights unchanged | Hige et al. 2015 |
| Cell specificity | Only the Kenyon cells that carried the odour are depressed | Hige et al. 2015 |
| Compartment specificity | Only the compartment where dopamine was released changes | Aso et al. 2014 |
| Order dependence | Odour → dopamine depresses; dopamine → odour potentiates; a 6 s gap does nothing | Handler et al. 2019 |
| γ1pedc exception | Backward pairing does not potentiate in γ1pedc | Hige et al. 2015 |
| Receptor knockout | Removing the Gs receptor abolishes learning | Kim et al. 2007; Handler et al. 2019 |
| Compartment table | Every cell type in the atlas exists in FlyWire v783, and every curated dopaminergic type is accounted for | Schlegel et al. 2024 |

## Calibrated, not predicted

These are fitted, so they cannot count as successes — they are the targets the
free parameters were set against.

| Quantity | Target | Fitted parameter |
| --- | --- | --- |
| Peak dopamine in a compartment | 0.4 µM at 20 Hz | release per spike |
| Half-decay of the transient | ~2 s | transporter V_max and the diffusion rate |
| Depression after one pairing | 90% of the trained synapses | depression rate |

## Not reproduced yet

| Quantity | Fly | Model | Why |
| --- | --- | --- | --- |
| Loss of the trained odour response | 80–90% | matches at the synapse, steeper at the output neuron | the isolated mushroom body leaves the output neuron near threshold; a fitted tonic drive helps but does not replace its real input |
| Loss of the control odour response | ~25% | lower or higher depending on the assumed odour overlap | the overlap between two odours is a free parameter until the antennal lobe pathway is modelled |
| Kenyon cell firing rates | a few spikes per odour | tens of hertz needed to drive the circuit | APL and DPM are graded neurons; a spiking model makes them fire at hundreds of hertz and dominate |
| Compartment-specific rates and retention | differ strongly between compartments | one rate for all | measured only for γ1, γ4 and γ5 |

## Protocols

**Conditioning** (`flyneuromod conditioning`). Measure the response of the
compartment's output neuron to a trained and a control odour, pair the trained
odour with dopaminergic activation, wait, measure again. Every phase is
separated by a rest long compared with both the eligibility trace and cAMP —
without it the control odour is trained by accident, in two different ways.

**Timing** (`flyneuromod timing`). The same pairing at a range of intervals
between odour and dopamine, from dopamine first to dopamine seconds later.
