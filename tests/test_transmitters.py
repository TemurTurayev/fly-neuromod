"""Curated transmitter identities override the connectome's predictions."""

import numpy as np
import pytest

from flyneuromod.data.annotations import load_annotations
from flyneuromod.data.connectome import load_connectome
from flyneuromod.data.transmitters import (
    TransmitterOverride,
    apply_transmitter_overrides,
    load_overrides,
)

from .conftest import ANNOTATIONS, COMPLETENESS, CONNECTIVITY, requires_data


def test_packaged_overrides_cover_the_mushroom_body_modulators():
    overrides = load_overrides()
    by_value = {o.value: o for o in overrides}
    assert by_value["Kenyon_Cell"].sign == 1
    assert by_value["DPM"].sign == -1
    assert by_value["APL"].sign == -1
    assert all(len(o.source) > 40 for o in overrides)


def test_override_validation():
    with pytest.raises(ValueError, match="select"):
        TransmitterOverride("colour", "DPM", "gaba", -1, "source text long enough")
    with pytest.raises(ValueError, match="sign"):
        TransmitterOverride("cell_type", "DPM", "gaba", 0, "source text long enough")
    with pytest.raises(ValueError, match="source"):
        TransmitterOverride("cell_type", "DPM", "gaba", -1, "")


def test_signs_are_corrected_without_touching_synapse_counts(tiny_tables, tiny_annotations):
    """The KC in the fixture has an excitatory outgoing connection; DPM-style flip works."""
    connectome = load_connectome(*tiny_tables)
    annotations = load_annotations(tiny_annotations)
    override = TransmitterOverride(
        select="cell_class",
        value="Kenyon_Cell",
        transmitter="gaba (test)",
        sign=-1,
        source="a source long enough to pass validation",
    )
    corrected = apply_transmitter_overrides(connectome, annotations, (override,))

    before = connectome.weights.toarray()
    after = corrected.weights.toarray()
    assert np.array_equal(np.abs(before), np.abs(after))  # anatomy unchanged
    assert after[2, 1] == -abs(before[2, 1])  # Kenyon cell output now inhibitory
    assert after[1, 0] == before[1, 0]  # other cells untouched
    assert connectome.weights.toarray()[2, 1] == before[2, 1]  # input not modified


def test_release_label_records_the_correction(tiny_tables, tiny_annotations):
    connectome = load_connectome(*tiny_tables)
    annotations = load_annotations(tiny_annotations)
    corrected = apply_transmitter_overrides(connectome, annotations, ())
    assert corrected.release.endswith("curated_nt")


@requires_data
def test_dpm_becomes_inhibitory_in_the_real_connectome():
    """DPM is predicted dopaminergic, so the base model has it exciting all Kenyon cells."""
    connectome = load_connectome(COMPLETENESS, CONNECTIVITY)
    annotations = load_annotations(ANNOTATIONS)

    dpm_ids = annotations.by_cell_type("DPM")
    dpm = connectome.indices_of(dpm_ids)
    kenyon = connectome.indices_of(annotations.kenyon_cells())

    before = connectome.weights[kenyon][:, dpm]
    assert before.data.max() > 0  # excitatory in the unmodified model

    corrected = apply_transmitter_overrides(connectome, annotations)
    after = corrected.weights[kenyon][:, dpm]
    assert after.data.max() <= 0
    assert np.array_equal(np.abs(before.data), np.abs(after.data))
