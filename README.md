# fly-neuromod

A biologically grounded **neuromodulation layer** for whole-brain connectome models of
*Drosophila melanogaster*. First modulator: **dopamine**.

Connectome-based models — the leaky integrate-and-fire model of
[Shiu et al. 2024](https://doi.org/10.1038/s41586-024-07763-9) and everything built on it during
the 2026 wave of fly-brain simulations — treat the brain as a graph of fast, point-to-point
synapses. Real brains are not only that. Dopamine, serotonin, octopamine and neuropeptides act
through **G-protein-coupled receptors**, over seconds to minutes, on volumes rather than single
contacts, and they change what the fast network does: which synapses get weaker, how excitable a
cell is, whether the fly learns anything at all.

This adds that layer, and tries to be explicit about the difference between what is measured,
what is inferred, and what is assumed.

## What it does

```
spikes (0.1 ms)                    chemistry (1 ms)
┌──────────────────────┐          ┌────────────────────────────────────────┐
│ LIF network          │ spikes   │ dopamine layer                         │
│  138,639 neurons     ├─────────►│  pool dopaminergic spikes per          │
│  15.1 M connections  │          │  compartment → concentration →         │
│                      │          │  receptors → cAMP / calcium →          │
│  synaptic weights ◄──┼──────────┤  plasticity at Kenyon cell synapses    │
└──────────────────────┘          └────────────────────────────────────────┘
```

- **Dopamine is not current.** No ionotropic dopamine receptor is known in the fly, so
  dopaminergic connections are removed from the spiking layer and routed through the slow one.
- **Release, diffusion, uptake.** Concentration per mushroom body compartment, with
  Michaelis-Menten transporter kinetics, calibrated against the 0.3–0.5 µM peak and 1.4–2.7 s
  half-decay measured in the adult mushroom body.
- **Receptors with real affinities.** Dop1R1 (Gs, EC50 0.61 µM) and Dop1R2 (Gq, EC50 0.057 µM) —
  a tenfold difference in sensitivity, which is what makes the sign of learning depend on timing.
- **A learning rule from the fly literature.** Presynaptic, compartment-specific, three-factor,
  and order-dependent: odour then dopamine depresses, dopamine then odour potentiates.
- **Drugs and mutants as parameters.** Transporter blockers, synthesis inhibition, receptor
  knockouts — and SCH-23390 as a deliberately inert control, because mammalian D1 pharmacology
  does not transfer to the fly.

## Install and run

```bash
git clone https://github.com/TemurTurayev/fly-neuromod && cd fly-neuromod
uv sync
uv run flyneuromod download      # ~130 MB of connectome data, not stored in the repo
uv run flyneuromod info          # what the connectome contains
uv run flyneuromod conditioning  # pair an odour with dopamine, measure the output neuron
uv run flyneuromod timing        # the sign of plasticity against the pairing interval
```

## Five things that went wrong, and what they teach

Every one of these was found by running the model, and each is a trap for anyone building on a
connectome.

**1. Predicted transmitters are not cell identities.** In FlyWire v783, 5,909 neurons carry the
predicted transmitter `dopamine` — and 5,172 of them are Kenyon cells, which are cholinergic. The
curated dopaminergic class has **331** neurons in 27 types. Picking dopamine sources by predicted
transmitter turns the entire mushroom body into a dopamine source.

**2. The same error has a second victim.** DPM, the single large modulatory neuron of the
mushroom body, is also predicted dopaminergic. It actually releases GABA and serotonin, so in the
unmodified model it *excites* every Kenyon cell instead of damping them. Curated corrections live
in [`transmitter_overrides.yaml`](src/flyneuromod/data/transmitter_overrides.yaml), each with its
source.

**3. Receptors cannot be dropped in by name.** Putting Dop2R on Kenyon cell terminals alongside
Dop1R1, at equal gain, silently cancels the Gs branch and abolishes learning. Dop2R's documented
role in the mushroom body is autoinhibition of release by the dopaminergic neurons themselves.

**4. Uptake alone lets dopamine run away.** Transporter uptake saturates at `v_max`, so a
compartment driven harder than that accumulates dopamine without bound — the model reached 9 µM
against a measured peak of 0.3–0.5 µM. Clearance in the fly is mixed: transporter-null (`fumin`)
flies still clear dopamine, so a diffusion term belongs in the model.

**5. Eligibility traces make the order of an experiment matter.** Measuring the control odour
shortly before the pairing trains it too: its Kenyon cells still carry an eligibility trace when
dopamine arrives. The symptom is a control odour that looks almost as depressed as the trained
one — which reads like a broken learning rule, but is a broken protocol.

## What it reproduces, and what it does not

Reproduced:

- Dopamine alone changes nothing; odour alone changes nothing; only the pairing writes anything.
- Depression is specific to the Kenyon cells that carried the odour, and to the compartment where
  dopamine was released.
- The sign of the change follows the order of odour and dopamine, and vanishes when they are far
  apart in time.
- Removing the Gs receptor abolishes learning, as in `dumb` mutants.
- Blocking the transporter prolongs the dopamine transient.

Not yet:

- **The size of the response change.** At the synapse the model reaches the depression measured by
  Hige et al. (2015); at the output neuron the change is steeper than in the fly, because the
  isolated mushroom body gives that neuron no input from the rest of the brain and it sits close to
  threshold. The synaptic readout is the honest one for now.
- **Physiological Kenyon cell firing rates.** The model needs stronger drive than the fly uses,
  because APL and DPM are graded neurons that a spiking model represents poorly — they fire at
  hundreds of hertz here and dominate the circuit.
- Compartment-specific learning rates and retention, the Dop2R autoreceptor loop, excitability
  modulation outside the mushroom body, co-transmission, and any modulator other than dopamine.

## Documentation

- [`docs/DESIGN.md`](docs/DESIGN.md) — architecture and the decisions worth arguing about.
- [`docs/parameters.md`](docs/parameters.md) — every constant with its source and confidence.
- [`docs/references.md`](docs/references.md) — the papers this is built on.

## Data and credit

This repository contains **code only**. Connectome data is downloaded on first use: FlyWire v783
connectivity as packaged by the [Shiu et al. brain model](https://github.com/philshiu/Drosophila_brain_model)
(MIT), and cell-type annotations from
[flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations)
(Schlegel et al. 2024). If you use this work, cite those papers and the experiments in
[`docs/references.md`](docs/references.md), not just this repository.

## Licence

MIT for the code here. The downloaded data keeps its own licence and citation requirements.
