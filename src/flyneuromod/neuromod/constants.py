"""Literature-derived defaults for the dopamine layer.

Every constant carries its source and a confidence label:

``measured``
    Directly measured in *Drosophila*.
``calibrated``
    Fitted so that the model reproduces a measured observable (the observable is
    named).
``assumed``
    No fly measurement exists; the value is a defensible placeholder and the
    reasoning is given.

Do not treat ``assumed`` values as facts. ``docs/parameters.md`` holds the full
table, including the numbers we deliberately did not use.

Units: micromolar for concentrations, seconds for time.
"""

from __future__ import annotations

from .field import ReleaseKinetics
from .plasticity import PlasticityParams
from .receptors import ReceptorSpec, SignalingParams

# ---------------------------------------------------------------------------
# receptors
# ---------------------------------------------------------------------------

DOP1R1 = ReceptorSpec(
    name="Dop1R1",
    # measured: EC50 0.61 +/- 0.07 uM for dopamine-driven Gs activation (BRET),
    # Himmelreich et al. 2017, Cell Rep 21:2074, doi:10.1016/j.celrep.2017.10.104.
    # A cAMP-readout study gives ~0.3 uM (Sugamori et al. 1995, FEBS Lett 362:131).
    ec50=0.61,
    hill=1.0,  # assumed: single-site binding, no cooperativity reported
    # measured: Gs activation rate 0.46 +/- 0.01 1/s (Himmelreich et al. 2017)
    tau_on=2.2,
    tau_off=5.0,  # assumed: no fly deactivation kinetics published
    coupling="Gs",
)

DOP1R2_GQ = ReceptorSpec(
    name="Dop1R2(Gq)",
    # measured: EC50 56.7 +/- 7.7 nM at the Gq arm, ten times more sensitive than
    # Dop1R1 (Himmelreich et al. 2017). Dop1R2 is not a plain "D1-like" receptor:
    # its Gs arm has EC50 7.37 uM, and in sleep-control neurons it couples to Gi/o
    # (Pimentel et al. 2016, Nature 536:333, doi:10.1038/nature19055).
    ec50=0.0567,
    hill=1.0,  # assumed
    # measured: Gq activation rate 2.71 +/- 0.04 1/s (Himmelreich et al. 2017)
    tau_on=0.37,
    tau_off=2.0,  # assumed
    coupling="Gq",
)

DOP2R = ReceptorSpec(
    name="Dop2R",
    # assumed: Hearn et al. 2002, PNAS 99:14554 report only that dopamine is the
    # most potent agonist, with no EC50. 0.5 uM sits in the defensible 0.1-1 uM
    # range implied by their nanomolar-potency agonists.
    ec50=0.5,
    hill=1.0,  # assumed
    tau_on=1.0,  # assumed
    tau_off=5.0,  # assumed
    coupling="Gi",
)

DOPAMINE_RECEPTORS = (DOP1R1, DOP1R2_GQ, DOP2R)

# ---------------------------------------------------------------------------
# release and clearance in a mushroom body compartment
# ---------------------------------------------------------------------------

MB_COMPARTMENT_RELEASE = ReleaseKinetics(
    # calibrated so that 20 Hz firing of one dopaminergic neuron holds the
    # compartment near 0.4 uM, the peak measured in the adult mushroom body
    # during sugar feeding and cholinergic stimulation
    # (Shin & Venton 2022, Angew Chem 61:e202207399, doi:10.1002/anie.202207399).
    per_spike=0.0053,
    # calibrated: with k_m fixed at the measured value, v_max is set so that the
    # low-concentration clearance rate v_max / k_m = 0.35 1/s reproduces the
    # measured half-decay of 1.4-2.7 s in the adult mushroom body (Shin & Venton
    # 2022). The directly measured larval v_max of 0.11 uM/s (Vickrey et al.
    # 2013) would give a ~12 s decay, far slower than the adult brain shows.
    v_max=0.45,
    # measured: diffusion-corrected K_m of the dopamine transporter, 1.3 +/- 0.6 uM
    # in the larval CNS (Vickrey et al. 2013, ACS Chem Neurosci 4:832,
    # doi:10.1021/cn400019q). No adult measurement exists.
    k_m=1.3,
    baseline=0.0,  # tonic dopamine emerges from tonic firing of the neurons
)

# ---------------------------------------------------------------------------
# intracellular signalling
# ---------------------------------------------------------------------------

KC_SIGNALING = SignalingParams(
    # assumed/calibrated: cAMP imaging in the mushroom body evolves over tens of
    # seconds (Tomchik & Davis 2009, Neuron 64:510) and pairing experiments
    # average cAMP over ~4 s windows (Handler et al. 2019, Cell 178:60).
    tau_camp=4.0,
    # assumed: the Gq/IP3 calcium branch is stimulus-locked and roughly an order
    # of magnitude faster than cAMP.
    tau_calcium=1.0,
    gain_gs=1.0,
    gain_gi=1.0,
    gain_gq=1.0,
    basal_camp=0.0,
)

# ---------------------------------------------------------------------------
# plasticity
# ---------------------------------------------------------------------------

KC_TO_MBON_PLASTICITY = PlasticityParams(
    # measured window: pairing is effective when dopamine follows odour by
    # 0.1-1 s and has no effect by ~6 s (Handler et al. 2019). tau = 2 s leaves
    # 5% of the trace at 6 s.
    tau_eligibility=2.0,
    # calibrated: see experiments/conditioning.py; the target is the ~30-50%
    # depression of the Kenyon cell to output neuron response measured after
    # paired odour and dopaminergic activation (Hige et al. 2015, Neuron 88:985).
    rate_depression=0.008,
    rate_potentiation=0.004,
    min_fraction=0.0,
    max_fraction=1.5,
    # Forgetting in the fly is an active, dopamine-driven process through Dop1R2
    # rather than passive decay (Berry et al. 2012, Neuron 74:530), so the
    # passive drift is off by default.
    tau_recovery=None,
)
