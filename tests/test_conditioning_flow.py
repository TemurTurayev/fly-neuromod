"""The conditioning protocol end to end, on a toy mushroom body.

This runs the same code path as the real experiment - measure, pair, rest,
measure - but on a network small enough to finish in a second, so the protocol
logic itself is covered without waiting for a full simulation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flyneuromod.data.annotations import load_annotations
from flyneuromod.data.connectome import load_connectome
from flyneuromod.experiments.conditioning import (
    ConditioningProtocol,
    _overlapping_odour,
    run_conditioning,
)
from flyneuromod.experiments.mushroom_body import MushroomBody, extract_mushroom_body

N_KENYON = 24


@pytest.fixture
def toy_mushroom_body(tmp_path: Path) -> MushroomBody:
    """One dopaminergic neuron, 24 Kenyon cells, one output neuron."""
    root_ids = [3000 + i for i in range(N_KENYON + 2)]
    dan, mbon = 0, N_KENYON + 1
    kenyon = list(range(1, N_KENYON + 1))

    pd.DataFrame({"Completed": [True] * len(root_ids)}, index=root_ids).to_csv(
        tmp_path / "completeness.csv"
    )

    pre = kenyon + [dan] * N_KENYON
    post = [mbon] * N_KENYON + kenyon
    counts = [12] * N_KENYON + [4] * N_KENYON
    pd.DataFrame(
        {
            "Presynaptic_ID": [root_ids[i] for i in pre],
            "Postsynaptic_ID": [root_ids[i] for i in post],
            "Presynaptic_Index": pre,
            "Postsynaptic_Index": post,
            "Connectivity": counts,
            "Excitatory": [1] * len(counts),
            "Excitatory x Connectivity": counts,
        }
    ).to_parquet(tmp_path / "connectivity.parquet")

    pd.DataFrame(
        {
            "root_id": root_ids,
            "cell_class": ["DAN"] + ["Kenyon_Cell"] * N_KENYON + [None],
            "cell_type": ["PPL101"] + ["KCg-m"] * N_KENYON + ["MBON11"],
            "top_nt": ["dopamine"] * (N_KENYON + 1) + ["gaba"],
            "side": ["left"] * len(root_ids),
        }
    ).to_csv(tmp_path / "annotations.tsv", sep="\t", index=False)

    connectome = load_connectome(
        tmp_path / "completeness.csv", tmp_path / "connectivity.parquet"
    )
    annotations = load_annotations(tmp_path / "annotations.tsv")
    return extract_mushroom_body(connectome, annotations)


def quick_protocol(**changes) -> ConditioningProtocol:
    base = ConditioningProtocol(
        odour_duration=0.3,
        test_duration=0.2,
        dopamine_duration=0.3,
        rest=0.5,
        odour_fraction=0.5,
        odour_overlap=0.25,
        readout_baseline_rate=0.0,  # no operating-point fitting in the toy network
    )
    return base.evolve(**changes)


def test_extraction_keeps_the_toy_circuit(toy_mushroom_body):
    assert toy_mushroom_body.n_neurons == N_KENYON + 2
    assert len(toy_mushroom_body.kenyon_indices()) == N_KENYON
    assert len(toy_mushroom_body.indices_of_type("PPL101")) == 1


def test_forward_pairing_depresses_the_trained_odour(toy_mushroom_body):
    result = run_conditioning(toy_mushroom_body, protocol=quick_protocol())
    assert result.trained_weight_change < 0
    assert result.compartment == "gamma1pedc"
    assert "gamma1pedc" in result.summary()


def test_backward_pairing_does_not_depress_in_gamma1pedc(toy_mushroom_body):
    """This compartment has no potentiating arm, so nothing should be written."""
    result = run_conditioning(
        toy_mushroom_body, protocol=quick_protocol(dopamine_onset=-0.5)
    )
    assert result.trained_weight_change <= 0


def test_silent_dopaminergic_neurons_write_nothing(toy_mushroom_body):
    result = run_conditioning(
        toy_mushroom_body, protocol=quick_protocol(dopamine_rate=0.0)
    )
    assert result.trained_weight_change == pytest.approx(0.0)


def test_missing_cell_types_are_reported(toy_mushroom_body):
    with pytest.raises(ValueError, match="MBON99"):
        run_conditioning(
            toy_mushroom_body, readout_type="MBON99", protocol=quick_protocol()
        )
    with pytest.raises(ValueError, match="PAM01"):
        run_conditioning(toy_mushroom_body, dan_type="PAM01", protocol=quick_protocol())


def test_control_odour_shares_the_requested_fraction():
    kenyon = np.arange(100)
    trained = np.arange(20)
    protocol = quick_protocol(odour_overlap=0.5)
    control = _overlapping_odour(kenyon, trained, protocol, np.random.default_rng(0))
    assert len(control) == len(trained)
    assert len(np.intersect1d(control, trained)) == 10


def test_fractional_loss_of_a_silent_neuron_is_zero(toy_mushroom_body):
    result = run_conditioning(toy_mushroom_body, protocol=quick_protocol(odour_rate=0.0))
    assert result.trained_depression == 0.0
    assert result.control_depression == 0.0
