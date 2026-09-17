# fly-neuromod

A biologically grounded **neuromodulation layer** for whole-brain connectome models of
*Drosophila melanogaster*. First modulator: **dopamine**.

Connectome-based brain models — the leaky integrate-and-fire model of
[Shiu et al. 2024](https://doi.org/10.1038/s41586-024-07763-9) and everything built on top of it
during the 2026 wave of fly-brain simulations — treat the brain as a graph of fast,
point-to-point synapses. Real brains are not only that. Dopamine, serotonin, octopamine and
neuropeptides act through **G-protein-coupled receptors**, on timescales of seconds to minutes,
over volumes rather than single contacts, and they change how the fast network behaves: which
synapses get stronger, how excitable a cell is, whether the fly learns anything at all.

This project adds that missing layer, and tries to be honest about the difference between what is
measured, what is inferred and what is assumed.

## Status

Early work in progress. See [`docs/DESIGN.md`](docs/DESIGN.md) for the architecture and
[`docs/parameters.md`](docs/parameters.md) for the provenance of every constant.

## Why this is not "just a reward signal"

Most hobby projects implement dopamine as "add current to the dopamine neurons when something
good happens". Three things go wrong:

1. **Dopamine is not a fast synapse.** No ionotropic dopamine receptor is known in *Drosophila*;
   all four receptors (Dop1R1, Dop1R2, Dop2R, DopEcR) are GPCRs. A dopaminergic connection in the
   connectome should not inject current on the millisecond timescale — it should change the state
   of the postsynaptic cell over seconds.
2. **Neurotransmitter predictions are not cell identities.** In the FlyWire v783 annotations,
   5,909 neurons carry the predicted transmitter `dopamine` — but 5,172 of them are Kenyon cells,
   which are cholinergic. The curated dopaminergic class (`cell_class == "DAN"`) has **331**
   neurons in 27 types (PAM01-15, PPL101-108, PPL201-204). Selecting by predicted transmitter
   silently turns the entire mushroom body into a dopamine source.
3. **Learning happens at specific synapses under specific rules.** In the fly, the
   best-characterised dopamine-dependent plasticity is at Kenyon cell → mushroom body output
   neuron synapses, it is compartment-specific, and its sign depends on the *timing* between odour
   and dopamine.

## Data and credit

This repository contains **code only**. Connectome data is downloaded on first use:

- FlyWire v783 connectivity and completeness tables, as packaged by the
  [Shiu et al. brain model](https://github.com/philshiu/Drosophila_brain_model) (MIT).
- FlyWire cell-type annotations from
  [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations)
  (Schlegel et al. 2024).

If you use this work, cite the connectome and model papers listed in
[`docs/references.md`](docs/references.md), not just this repository.

## Licence

MIT for the code in this repository. The downloaded connectome data keeps its own licence and
citation requirements.
