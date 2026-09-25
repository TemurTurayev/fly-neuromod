"""Drugs and mutants act on the same parameters an experimenter would change."""

import numpy as np
import pytest

from flyneuromod.neuromod import pharmacology
from flyneuromod.neuromod.autoreceptor import AutoreceptorFeedback
from flyneuromod.neuromod.constants import DAN_AUTORECEPTOR_FEEDBACK, DOP1R1, MB_COMPARTMENT_RELEASE
from flyneuromod.neuromod.field import DopamineField


def clearance_halftime(kinetics, start=0.4, dt=1e-3) -> float:
    """Time for a concentration transient to fall to half, in seconds."""
    field = DopamineField(n_fields=1, kinetics=kinetics, dt=dt)
    field.concentration[0] = start
    for step in range(200_000):
        field.step(np.zeros(1))
        if field.concentration[0] <= start / 2:
            return (step + 1) * dt
    raise AssertionError("concentration did not halve")


def test_default_release_matches_measured_mushroom_body_transient():
    """20 Hz of one dopaminergic neuron holds ~0.3-0.5 uM, as measured in vivo."""
    auto = AutoreceptorFeedback(DAN_AUTORECEPTOR_FEEDBACK, n_fields=1, dt=1e-3)
    field = DopamineField(n_fields=1, kinetics=MB_COMPARTMENT_RELEASE, dt=1e-3)
    spikes_per_step = np.array([20.0 * 1e-3])
    gain = np.ones(1)
    for _ in range(30_000):
        c = field.step(spikes_per_step, gain=gain)
        gain = auto.step(c)
    assert 0.3 <= field.concentration[0] <= 0.5


def test_default_clearance_matches_measured_half_decay():
    """Half-decay of 1.4-2.7 s in the adult mushroom body (Shin & Venton 2022)."""
    assert 1.4 <= clearance_halftime(MB_COMPARTMENT_RELEASE) <= 2.7


def test_transporter_block_slows_clearance():
    control = clearance_halftime(MB_COMPARTMENT_RELEASE)
    cocaine = clearance_halftime(pharmacology.COCAINE.apply_to(MB_COMPARTMENT_RELEASE))
    assert cocaine > 1.5 * control


def test_transporter_null_is_slower_than_a_blocker():
    cocaine = clearance_halftime(pharmacology.COCAINE.apply_to(MB_COMPARTMENT_RELEASE))
    fumin = clearance_halftime(pharmacology.FUMIN.apply_to(MB_COMPARTMENT_RELEASE))
    assert fumin > cocaine


def test_autoreceptor_block_targets_dop2r():
    assert pharmacology.FLUPENTIXOL.blocks == {"Dop2R": pytest.approx(1000.0)}
    assert pharmacology.FLUPENTIXOL.release_factor == 1.0


def test_synthesis_inhibition_lowers_release():
    treated = pharmacology.IODOTYROSINE.apply_to(MB_COMPARTMENT_RELEASE)
    assert treated.per_spike < MB_COMPARTMENT_RELEASE.per_spike


def test_mammalian_antagonist_is_inert_in_the_fly():
    """SCH-23390 blocks mammalian D1 receptors but not the fly ones."""
    treated = pharmacology.SCH23390.apply_to(MB_COMPARTMENT_RELEASE)
    assert treated == MB_COMPARTMENT_RELEASE
    assert pharmacology.SCH23390.receptor_block == ()
    assert pharmacology.SCH23390.receptor_knockout == ()


def test_manipulations_are_immutable_and_hashable():
    """Review finding M1: a shared dict let one caller change a catalogue entry for everyone."""
    blocks = pharmacology.CONTROL.blocks
    blocks["Dop1R1"] = 1e6
    assert pharmacology.get("control").blocks == {}
    assert hash(pharmacology.COCAINE) != hash(pharmacology.CONTROL)


def test_knockouts_name_existing_receptors():
    names = {DOP1R1.name, "Dop1R2(Gq)", "Dop2R"}
    for manipulation in pharmacology.CATALOGUE.values():
        assert manipulation.receptors_named <= names


def test_every_manipulation_documents_its_source():
    for name, manipulation in pharmacology.CATALOGUE.items():
        if name == "control":
            continue
        assert len(manipulation.note) > 40, f"{name} lacks a provenance note"


def test_lookup_reports_unknown_names():
    assert pharmacology.get("cocaine") is pharmacology.COCAINE
    assert pharmacology.get("dop1r1-null") is pharmacology.DOP1R1_KNOCKOUT
    with pytest.raises(KeyError, match="available"):
        pharmacology.get("bromocriptine")


def test_invalid_manipulation_is_rejected():
    with pytest.raises(ValueError):
        pharmacology.Manipulation(name="bad", k_m_factor=0.0)
    with pytest.raises(ValueError):
        pharmacology.Manipulation(name="bad", release_factor=-1.0)
