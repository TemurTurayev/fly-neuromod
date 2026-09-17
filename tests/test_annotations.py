import pytest

from flyneuromod.data.annotations import load_annotations

from .conftest import ANNOTATIONS, requires_data


def test_dopaminergic_selection_ignores_predicted_transmitter(tiny_annotations):
    """The core trap: Kenyon cells are predicted 'dopamine' but are not dopamine sources."""
    annotations = load_annotations(tiny_annotations)
    assert annotations.dopaminergic().tolist() == [1000]
    assert sorted(annotations.predicted_transmitter("dopamine").tolist()) == [1000, 1001, 1002]


def test_cell_type_selectors(tiny_annotations):
    annotations = load_annotations(tiny_annotations)
    assert annotations.kenyon_cells().tolist() == [1001, 1002]
    assert annotations.mbons().tolist() == [1003]
    assert annotations.by_cell_type("PPL101").tolist() == [1000]
    assert annotations.by_type_prefix("KC").tolist() == [1001, 1002]
    assert annotations.by_cell_type().tolist() == []


def test_cell_types_of_and_counts(tiny_annotations):
    annotations = load_annotations(tiny_annotations)
    types = annotations.cell_types_of([1003, 1000])
    assert types.tolist() == ["MBON01", "PPL101"]
    counts = annotations.types_in([1001, 1002, 1002])
    assert counts["KCab"] == 2


def test_unknown_root_ids_yield_missing_values(tiny_annotations):
    annotations = load_annotations(tiny_annotations)
    assert annotations.cell_types_of([424242]).isna().all()


def test_restricted_to_subset(tiny_annotations):
    annotations = load_annotations(tiny_annotations)
    subset = annotations.restricted_to([1000, 1003])
    assert len(subset.table) == 2
    assert subset.dopaminergic().tolist() == [1000]


def test_missing_file_raises_with_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="download"):
        load_annotations(tmp_path / "missing.tsv")


@requires_data
def test_real_annotations_confirm_the_dopamine_trap():
    """Regression test for the numbers quoted in the README."""
    annotations = load_annotations(ANNOTATIONS)
    dans = annotations.dopaminergic()
    predicted = annotations.predicted_transmitter("dopamine")
    kenyon = set(annotations.kenyon_cells().tolist())

    assert len(dans) == 331
    assert len(predicted) == 5_909
    assert len(kenyon & set(predicted.tolist())) == 5_172
    assert not (set(dans.tolist()) & kenyon)
    assert annotations.types_in(dans).index.str.match("PAM|PPL").all()
