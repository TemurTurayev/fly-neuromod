# Design

## The problem

Connectome-based models of the fly brain simulate a graph of fast synapses. Real
brains also run a slower chemistry on top of that graph, and in *Drosophila* the
best understood part of it is dopamine: it is what turns the mushroom body from a
fixed classifier into something that learns.

A dopamine layer that deserves the name has to get four things right.

1. **Dopamine is not current.** No ionotropic dopamine receptor is known in the
   fly. Every documented effect runs through G-protein-coupled receptors over
   seconds. So dopaminergic connections must leave the fast layer and enter a
   slow one.
2. **Dopamine acts on a volume, not on a wire.** It is released into a mushroom
   body compartment, and every Kenyon cell terminal in that compartment sees it.
   The compartment, not the synapse, is the unit.
3. **Receptors differ.** Dop1R2 responds to tens of nanomolar through Gq, Dop1R1
   to hundreds through Gs. The same dopamine transient means different things to
   the two branches, and that difference is what makes the sign of learning
   depend on timing.
4. **The identity of a dopaminergic neuron is curated, not predicted.** Selecting
   sources by the predicted transmitter of the connectome would make every Kenyon
   cell a dopamine source.

## Layers

```
spikes (0.1 ms)                    chemistry (1 ms)
┌──────────────────────┐          ┌────────────────────────────────────────┐
│ LIFNetwork           │ spikes   │ DopamineLayer                          │
│  138,639 neurons     ├─────────►│  pool DAN spikes per compartment       │
│  15.1 M connections  │          │  DopamineField   release + DAT uptake  │
│                      │          │  ReceptorPopulation  Dop1R1 / Dop1R2   │
│  synapse_weight  ◄───┼──────────┤  SecondMessenger  cAMP, calcium        │
│  (written in place)  │ weights  │  SynapticPlasticity  KC → MBON         │
└──────────────────────┘          └────────────────────────────────────────┘
```

The two layers run at different steps because their timescales differ by four
orders of magnitude. The dopamine layer is a step callback: the network calls it
after every step, it accumulates spikes, and every tenth call it advances its own
state and writes new weights into the array the network is already using. There
is no copying and no second simulator.

## Components

| Module | Responsibility |
| --- | --- |
| `engine/params.py` | Immutable fast-layer constants; validation. |
| `engine/lif.py` | Event-driven spiking network with the exact propagator; locating synapses. |
| `data/connectome.py` | FlyWire tables to a signed sparse matrix; subnetworks. |
| `data/annotations.py` | Curated cell identity, kept separate from predicted transmitter. |
| `neuromod/field.py` | Extracellular dopamine: release, Michaelis-Menten uptake. |
| `neuromod/receptors.py` | Hill-curve occupancy with kinetics; cAMP and calcium. |
| `neuromod/plasticity.py` | The three-factor, order-dependent rule. |
| `neuromod/mb_atlas.py` | The 16 compartments and their cell types. |
| `neuromod/pharmacology.py` | Drugs and mutants as parameter changes. |
| `neuromod/dopamine.py` | Wiring all of the above to a running network. |
| `experiments/` | Protocols that reproduce published experiments. |

Each has one job, its own tests, and no knowledge of the others' internals.

## Decisions worth arguing about

**A new simulator instead of Brian 2.** The reference model is written in Brian 2
and we match its equations exactly, using the closed-form solution of the linear
subthreshold dynamics. Brian 2 is the better tool for a single one-second trial;
it is the wrong tool for a conditioning protocol that has to simulate a minute of
brain time per condition, many times over, while a Python-level modulator writes
weights every millisecond. The equivalence is a test, not a claim.

**Order-selective detectors instead of a product.** The first rule multiplied
cAMP by a presynaptic trace. It learned, and it could never tell odour-then-
dopamine from dopamine-then-odour: dopamine lingers for seconds, so both orders
look like overlap. A parameter scan confirmed that no receptor kinetics fix
this. The rule now uses the two molecular detectors in the terminal with their
known order preferences - the calcium-primed cyclase for depression, the
IP₃-primed receptor for potentiation - and the sign flip follows.

**Receptor activation per field, specificity per synapse.** Receptors and IP₃
are tracked per compartment and hemisphere, because dopamine covers the whole
compartment. What makes learning cell-specific is the calcium trace, which is
per synapse.

**Hemispheres are separate.** Each compartment exists twice. A dopaminergic
neuron releases into its own side, and a Kenyon cell synapse belongs to its
cell's side.

**Protocols are timelines.** A pairing is built as a list of segments before
anything is simulated, which made it possible to test the schedule on its own.
The first version computed it inline and silently delivered the wrong
intervals.

**Plasticity as a factor on the anatomical weight.** Weights are stored as the
anatomical synapse count times a learned factor in [0, 1.5]. Inhibitory synapses
keep their sign, learning cannot invert a connection, and the anatomy is always
recoverable.

**Fast dopaminergic synapses are removed by default.** This is right for
dopamine, and slightly wrong for the dopaminergic neurons that genuinely
co-release GABA or glutamate (Yamazaki et al. 2023). The switch is exposed;
per-type co-transmission is future work.

## Out of scope in v0.1, and why

- **Odour input through the antennal lobe.** An odour here is a random sparse set
  of Kenyon cells. Real odour identity needs the projection neuron pathway and
  receptor response data; it does not change the plasticity rule being tested.
- **The Dop2R autoreceptor loop.** Blocking it quadruples released dopamine, so
  it matters; until the loop is implemented, that drug is modelled as increased
  release, which is stated where it happens.
- **Excitability modulation outside the mushroom body.** The sleep-control
  neurons are the one quantified case (Pimentel et al. 2016) and are the natural
  next target.
- **Serotonin, octopamine, neuropeptides.** The interfaces are deliberately
  generic — a modulator is a field, a receptor set, a signal and an effector —
  but nothing else is implemented yet.

## Roadmap

1. Validation suite against published experiments: timing dependence, receptor
   knockouts, compartment-specific retention, output-neuron valence.
2. The Dop2R autoreceptor and release-dependent depression.
3. Excitability as a second effector, starting with the sleep-control circuit.
4. Co-transmission per dopaminergic cell type.
5. Octopamine and serotonin on the same scaffolding.
