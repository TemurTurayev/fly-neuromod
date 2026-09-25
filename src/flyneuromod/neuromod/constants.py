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

from .autoreceptor import AutoreceptorParams
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

DAN_AUTORECEPTOR = ReceptorSpec(
    name="Dop2R",
    # calibrated: EC50 0.20 uM sits in the defensible 0.1-1 uM range for
    # Gi-coupled dopamine receptors (Hearn et al. 2002, PNAS 99:14554), calibrated so that
    # presynaptic autoreceptor feedback suppresses steady-state release by ~3-4x.
    ec50=0.20,
    hill=1.0,  # assumed
    tau_on=1.0,  # assumed
    tau_off=5.0,  # assumed
    coupling="Gi",
)
"""Dop2R, the Gi-coupled receptor.

Its documented role in the mushroom body is presynaptic autoinhibition of the
dopaminergic neurons themselves: blocking it with flupentixol raises evoked
dopamine about fourfold (Shin & Venton 2022). It is configured as a presynaptic
autoreceptor feedback loop on dopaminergic terminals via AutoreceptorFeedback
and DAN_AUTORECEPTOR_FEEDBACK rather than being part of the default receptor set
on Kenyon cell terminals.
"""

DAN_AUTORECEPTOR_FEEDBACK = AutoreceptorParams(
    spec=DAN_AUTORECEPTOR,
    # calibrated: max_suppression=0.90 reproduces the ~4-fold rise in evoked dopamine
    # when Dop2R is blocked by flupentixol (Shin & Venton 2022).
    max_suppression=0.90,
    # assumed: min_gain=0.1 prevents total shutoff of release under strong stimulation.
    min_gain=0.1,
)

KC_TERMINAL_RECEPTORS = (DOP1R1, DOP1R2_GQ)
"""Receptors modelled on Kenyon cell terminals: the Gs and Gq branches.

Both are expressed there (Kudo et al. 2025, eLife 14:RP98358) and together they
produce the order-dependent sign of plasticity measured by Handler et al. 2019.
"""

DOPAMINE_RECEPTORS = KC_TERMINAL_RECEPTORS

# ---------------------------------------------------------------------------
# release and clearance in a mushroom body compartment
# ---------------------------------------------------------------------------

MB_COMPARTMENT_RELEASE = ReleaseKinetics(
    # calibrated: with the presynaptic Dop2R autoreceptor feedback loop active,
    # 20 Hz firing of a compartment's whole dopaminergic population holds it near 0.35-0.4 uM,
    # reproducing the peak measured in the adult mushroom body during sugar feeding
    # and cholinergic stimulation (Shin & Venton 2022, Angew Chem 61:e202207399,
    # doi:10.1002/anie.202207399).
    per_spike=0.0145,
    # calibrated: with k_m fixed at the measured value, transporter uptake and
    # diffusion together give a low-concentration clearance rate of
    # v_max / k_m + k_diffusion = 0.35 1/s, reproducing the measured half-decay of
    # 1.4-2.7 s in the adult mushroom body (Shin & Venton 2022). The directly
    # measured larval v_max of 0.11 uM/s (Vickrey et al. 2013) would give a ~12 s
    # decay on its own, far slower than the adult brain shows.
    v_max=0.30,
    # measured: diffusion-corrected K_m of the dopamine transporter, 1.3 +/- 0.6 uM
    # in the larval CNS (Vickrey et al. 2013, ACS Chem Neurosci 4:832,
    # doi:10.1021/cn400019q). No adult measurement exists.
    k_m=1.3,
    baseline=0.0,  # tonic dopamine emerges from tonic firing of the neurons
    # assumed: about a third of the clearance is diffusion out of the
    # compartment rather than uptake. The split is not measured; what is
    # measured is that transporter-null flies still clear dopamine, so it cannot
    # be zero, and that blocking uptake slows but does not abolish clearance
    # (Makos et al. 2010).
    k_diffusion=0.12,
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
    # calibrated on the full FlyWire mushroom body against the synaptic
    # measurement of Hige et al. 2015, Neuron 88:985: rate_depression=1.5 gives
    # -89.8% trained synapse weight change onto MBON-gamma1pedc after one pairing,
    # matching the measured ~-90% (1.7 already saturates at -100%).
    # Units: fraction of weight per unit of Gs activation arriving onto a fully primed terminal.
    #
    # The calibration has to be done in the network, not in isolation: during
    # the odour, Kenyon cells excite PPL1-gamma1pedc through ~14,000 synapses, so
    # a 20 Hz drive becomes ~47 Hz of firing and ~0.9 uM of dopamine. That is the
    # reciprocal Kenyon cell -> dopaminergic neuron loop of Cervantes-Sandoval et
    # al. 2017 (eLife 6:e23789), appearing in the model without being put there.
    rate_depression=1.5,
    # calibrated on the sign of the timing curve: rate_potentiation=0.6 keeps the 0.4
    # ratio to depression rate that places the timing-curve crossover (dopamine 1.2 s
    # before odour potentiates, 0.5 s after depresses: Handler et al. 2019).
    rate_potentiation=0.6,
    min_fraction=0.0,
    max_fraction=1.5,
    # Forgetting in the fly is an active, dopamine-driven process through Dop1R2
    # rather than passive decay (Berry et al. 2012, Neuron 74:530), so the
    # passive drift is off by default.
    tau_recovery=None,
)
