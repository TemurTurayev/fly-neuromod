"""The compartment table is checked against the connectome, not trusted."""

import pytest

from flyneuromod.data.annotations import load_annotations
from flyneuromod.neuromod.mb_atlas import Compartment, MushroomBodyAtlas, load_atlas

from .conftest import ANNOTATIONS, requires_data


def test_packaged_atlas_covers_the_known_compartments():
    atlas = load_atlas()
    assert len(atlas) == 16
    assert "gamma1pedc" in atlas.names
    assert atlas.get("gamma1pedc").dan_types == ("PPL101",)
    assert atlas.get("gamma1pedc").mbon_types == ("MBON11",)


def test_lookup_by_cell_type():
    atlas = load_atlas()
    assert atlas.compartment_of_dan("PPL101") == "gamma1pedc"
    assert atlas.compartment_of_mbon("MBON01") == "gamma5"
    assert atlas.compartment_of_mbon("MBON99") is None
    assert atlas.index_of("gamma5") == atlas.names.index("gamma5")
    with pytest.raises(KeyError):
        atlas.index_of("nonexistent")


def test_valence_follows_the_transmitter_rule():
    """Aso et al. 2014b: aversion-driving output neurons are glutamatergic."""
    atlas = load_atlas()
    for compartment in atlas.compartments:
        if compartment.valence == "avoidance":
            assert compartment.mbon_transmitter == "glutamate"
        if compartment.valence == "approach":
            assert compartment.mbon_transmitter in ("gaba", "acetylcholine")


def test_potentiation_labels_carry_evidence():
    atlas = load_atlas()
    assert atlas.get("gamma1pedc").potentiation == "measured_no"
    assert not atlas.get("gamma1pedc").potentiates
    assert atlas.get("gamma4").potentiation == "measured_yes"
    assert atlas.get("gamma4").potentiates


def test_each_output_neuron_belongs_to_one_compartment():
    with pytest.raises(ValueError, match="one compartment"):
        MushroomBodyAtlas(
            compartments=(
                Compartment("a", "a", ("PAM01",), ("MBON01",), "glutamate", "avoidance", "unknown"),
                Compartment("b", "b", ("PAM02",), ("MBON01",), "glutamate", "avoidance", "unknown"),
            )
        )


def test_compartment_requires_a_dopaminergic_source():
    with pytest.raises(ValueError, match="DAN type"):
        Compartment("a", "a", (), ("MBON01",), "glutamate", "avoidance", "unknown")


def test_invalid_labels_are_rejected():
    with pytest.raises(ValueError, match="potentiation"):
        Compartment("a", "a", ("PAM01",), (), None, None, "maybe")
    with pytest.raises(ValueError, match="valence"):
        Compartment("a", "a", ("PAM01",), (), None, "tasty", "unknown")


def test_missing_from_reports_absent_types():
    atlas = load_atlas()
    report = atlas.missing_from({"PPL101"})
    assert "PAM01" in report["dan"]
    assert "MBON11" in report["mbon"]


@requires_data
def test_every_atlas_type_exists_in_the_connectome():
    """The curated table must match the FlyWire v783 cell types exactly."""
    annotations = load_annotations(ANNOTATIONS)
    known = set(annotations.table["cell_type"].dropna().unique())
    atlas = load_atlas()
    assert atlas.missing_from(known) == {"dan": [], "mbon": []}


@requires_data
def test_atlas_accounts_for_every_dopaminergic_type():
    """No curated dopaminergic type is silently dropped."""
    annotations = load_annotations(ANNOTATIONS)
    dan_types = set(annotations.cell_types_of(annotations.dopaminergic()).dropna().unique())
    atlas = load_atlas()
    accounted = set(atlas.dan_types()) | set(atlas.unassigned_dan_types)
    assert dan_types == accounted
