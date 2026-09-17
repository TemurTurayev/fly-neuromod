"""Protocol handling, and the full experiments against the real connectome.

The experiments marked ``slow`` simulate tens of seconds of brain time and take
minutes. Run them with ``pytest -m slow``.
"""

import numpy as np
import pytest

from flyneuromod.data.annotations import load_annotations
from flyneuromod.data.connectome import load_connectome
from flyneuromod.experiments.conditioning import run_conditioning
from flyneuromod.experiments.mushroom_body import extract_mushroom_body, sparse_odour
from flyneuromod.experiments.protocol import ConditioningProtocol
from flyneuromod.experiments.timing import timing_curve

from .conftest import ANNOTATIONS, COMPLETENESS, CONNECTIVITY, requires_data


def test_sparse_odour_picks_distinct_cells():
    kenyon = np.arange(1000)
    odour = sparse_odour(kenyon, fraction=0.1, rng=np.random.default_rng(0))
    assert len(odour) == 100
    assert len(set(odour.tolist())) == 100
    with pytest.raises(ValueError):
        sparse_odour(kenyon, fraction=0)


def test_sparse_odour_is_reproducible():
    kenyon = np.arange(1000)
    first = sparse_odour(kenyon, 0.1, np.random.default_rng(7))
    second = sparse_odour(kenyon, 0.1, np.random.default_rng(7))
    assert first.tolist() == second.tolist()


@requires_data
def test_mushroom_body_extraction_has_the_expected_cells():
    connectome = load_connectome(COMPLETENESS, CONNECTIVITY)
    annotations = load_annotations(ANNOTATIONS)
    mushroom_body = extract_mushroom_body(connectome, annotations)

    assert mushroom_body.n_neurons == 5608
    assert len(mushroom_body.kenyon_indices()) == 5177
    assert len(mushroom_body.indices_of_type("PPL101")) == 2
    assert len(mushroom_body.indices_of_type("MBON11")) == 2
    # connections among the selected cells are preserved
    assert mushroom_body.connectome.n_connections > 100_000


@pytest.mark.slow
@requires_data
def test_pairing_depresses_the_trained_odour_more_than_the_control():
    connectome = load_connectome(COMPLETENESS, CONNECTIVITY)
    annotations = load_annotations(ANNOTATIONS)
    mushroom_body = extract_mushroom_body(connectome, annotations)

    result = run_conditioning(mushroom_body, protocol=ConditioningProtocol())

    assert result.trained_weight_change < -0.7  # Hige et al. 2015: about -90%
    assert result.trained_depression > result.control_depression


@pytest.mark.slow
@requires_data
def test_the_sign_of_plasticity_follows_the_pairing_order():
    """Handler et al. 2019: forward pairing depresses, backward pairing potentiates."""
    connectome = load_connectome(COMPLETENESS, CONNECTIVITY)
    annotations = load_annotations(ANNOTATIONS)
    mushroom_body = extract_mushroom_body(connectome, annotations)

    points = timing_curve(mushroom_body, intervals=(-1.2, 0.0, 0.5, 6.0))
    by_interval = {p.interval: p.weight_change for p in points}

    assert by_interval[0.5] < 0  # odour first: depression
    assert by_interval[-1.2] > 0  # dopamine first: potentiation
    assert abs(by_interval[6.0]) < abs(by_interval[0.5])  # too late to matter
