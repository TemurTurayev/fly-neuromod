"""The dopamine layer wired to a spiking network, on hand-built toy circuits."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flyneuromod.data.annotations import load_annotations
from flyneuromod.data.connectome import load_connectome
from flyneuromod.engine.lif import LIFNetwork
from flyneuromod.engine.params import LIFParams
from flyneuromod.neuromod import pharmacology
from flyneuromod.neuromod.constants import DAN_AUTORECEPTOR, DOP1R1, DOP1R2_GQ
from flyneuromod.neuromod.dopamine import DopamineConfig, DopamineLayer


def build_circuit(tmp_path: Path, cells, connections):
    """Write and load a toy connectome.

    ``cells`` is a list of ``(cell_class, cell_type, side)``; ``connections`` a
    list of ``(pre, post, synapse_count)`` using positions in ``cells``.
    """
    root_ids = [5000 + i for i in range(len(cells))]
    pd.DataFrame({"Completed": [True] * len(cells)}, index=root_ids).to_csv(
        tmp_path / "completeness.csv"
    )
    pre, post, counts = zip(*connections, strict=True) if connections else ((), (), ())
    pd.DataFrame(
        {
            "Presynaptic_ID": [root_ids[i] for i in pre],
            "Postsynaptic_ID": [root_ids[i] for i in post],
            "Presynaptic_Index": list(pre),
            "Postsynaptic_Index": list(post),
            "Connectivity": list(counts),
            "Excitatory": [1] * len(counts),
            "Excitatory x Connectivity": list(counts),
        }
    ).to_parquet(tmp_path / "connectivity.parquet")
    pd.DataFrame(
        {
            "root_id": root_ids,
            "cell_class": [c[0] for c in cells],
            "cell_type": [c[1] for c in cells],
            "top_nt": ["dopamine"] * len(cells),
            "side": [c[2] for c in cells],
        }
    ).to_csv(tmp_path / "annotations.tsv", sep="\t", index=False)
    return (
        load_connectome(tmp_path / "completeness.csv", tmp_path / "connectivity.parquet"),
        load_annotations(tmp_path / "annotations.tsv"),
    )


# gamma1pedc on the left: one PPL101, three Kenyon cells, one MBON11
DAN, KC_A, KC_B, KC_C, MBON = 0, 1, 2, 3, 4


@pytest.fixture
def gamma1pedc(tmp_path):
    cells = [
        ("DAN", "PPL101", "left"),
        ("Kenyon_Cell", "KCg-m", "left"),
        ("Kenyon_Cell", "KCg-m", "left"),
        ("Kenyon_Cell", "KCab", "left"),
        (None, "MBON11", "left"),
    ]
    connections = [
        (KC_A, MBON, 10),
        (KC_B, MBON, 10),
        (KC_C, MBON, 10),
        (DAN, KC_A, 5),
        (DAN, KC_B, 5),
        (DAN, KC_C, 5),
    ]
    return build_circuit(tmp_path, cells, connections)


@pytest.fixture
def gamma5(tmp_path):
    """gamma5, where backward pairing is measured to potentiate."""
    cells = [
        ("DAN", "PAM01", "left"),
        ("Kenyon_Cell", "KCg-m", "left"),
        (None, "MBON01", "left"),
    ]
    return build_circuit(tmp_path, cells, [(1, 2, 10), (0, 1, 5)])


def attach(circuit, **config):
    connectome, annotations = circuit
    network = LIFNetwork(connectome.weights, LIFParams(), rng=np.random.default_rng(0))
    layer = DopamineLayer(network, connectome, annotations, config=DopamineConfig(**config))
    return network, layer


def drive(network, layer, neurons, seconds):
    """Force the listed neurons to spike at 100 Hz for the given time."""
    steps = int(round(seconds / network.params.dt))
    for step in range(steps):
        if neurons and step % 100 == 0:
            network.v[list(neurons)] = network.params.v_threshold + 1e-4
        layer(network, step, network.step())


def plastic_weights(network, layer):
    return network.synapse_weight[layer.targets.synapse_position].copy()


def test_layer_finds_the_circuit(gamma1pedc):
    _, layer = attach(gamma1pedc)
    assert layer.targets.dan_index.tolist() == [DAN]
    assert layer.targets.kenyon_index.tolist() == [KC_A, KC_B, KC_C]
    assert len(layer.targets.synapse_position) == 3
    field = 2 * layer.atlas.index_of("gamma1pedc")  # left side
    assert set(layer.targets.synapse_field.tolist()) == {field}


def test_dopamine_stays_in_its_compartment_and_hemisphere(gamma1pedc):
    network, layer = attach(gamma1pedc)
    drive(network, layer, [DAN], 1.0)
    assert layer.concentration("gamma1pedc", "left") > 0.01
    assert layer.concentration("gamma1pedc", "right") == pytest.approx(0.0)
    assert layer.concentration("gamma5") == pytest.approx(0.0)
    assert set(layer.concentrations()) >= {"gamma1pedc/left", "gamma1pedc/right"}


def test_dopamine_is_not_a_fast_synapse(gamma1pedc):
    network, layer = attach(gamma1pedc)
    network.v[DAN] = network.params.v_threshold + 1e-4
    for step in range(network.params.delay_steps + 2):
        layer(network, step, network.step())
    assert network.g[KC_A] == pytest.approx(0.0)


def test_fast_synapses_can_be_kept_for_cotransmission(gamma1pedc):
    network, _ = attach(gamma1pedc, remove_fast_dopamine_synapses=False)
    network.v[DAN] = network.params.v_threshold + 1e-4
    for _ in range(network.params.delay_steps + 1):
        network.step()
    assert network.g[KC_A] > 0.0


def test_forward_pairing_depresses_only_the_paired_cells(gamma1pedc):
    network, layer = attach(gamma1pedc)
    baseline = plastic_weights(network, layer)
    drive(network, layer, [KC_A, KC_B], 0.5)  # odour carried by two of three cells
    drive(network, layer, [KC_A, KC_B, DAN], 1.0)  # dopamine arrives during the odour
    weights = plastic_weights(network, layer)
    presynaptic = layer.targets.kenyon_index[layer.targets.synapse_presynaptic_slot]
    paired = np.isin(presynaptic, [KC_A, KC_B])
    assert np.all(weights[paired] < baseline[paired])
    assert weights[~paired] == pytest.approx(baseline[~paired])
    assert layer.weight_change_of_neurons(np.array([KC_A, KC_B]), "gamma1pedc") < 0
    assert layer.weight_change_of_neurons(np.array([KC_C]), "gamma1pedc") == pytest.approx(0.0)


def test_dopamine_alone_and_odour_alone_change_nothing(gamma1pedc):
    network, layer = attach(gamma1pedc)
    baseline = plastic_weights(network, layer)
    drive(network, layer, [DAN], 1.0)
    drive(network, layer, [], 10.0)
    drive(network, layer, [KC_A, KC_B, KC_C], 1.0)
    assert plastic_weights(network, layer) == pytest.approx(baseline)


def test_gamma1pedc_does_not_potentiate(gamma1pedc):
    network, layer = attach(gamma1pedc)
    baseline = plastic_weights(network, layer)
    drive(network, layer, [DAN], 1.0)
    drive(network, layer, [KC_A], 1.0)
    assert np.all(plastic_weights(network, layer) <= baseline + 1e-12)


def test_gamma5_potentiates_when_dopamine_comes_first(gamma5):
    """Review finding H5: potentiation through the whole layer was never exercised."""
    network, layer = attach(gamma5)
    baseline = plastic_weights(network, layer)
    drive(network, layer, [0], 1.0)  # dopamine first
    drive(network, layer, [1], 1.0)  # then the Kenyon cell
    assert np.all(plastic_weights(network, layer) > baseline)


def test_receptor_null_abolishes_depression(gamma1pedc):
    network, layer = attach(gamma1pedc, manipulation=pharmacology.DOP1R1_KNOCKOUT)
    baseline = plastic_weights(network, layer)
    drive(network, layer, [KC_A, KC_B], 0.5)
    drive(network, layer, [KC_A, KC_B, DAN], 1.0)
    assert plastic_weights(network, layer) == pytest.approx(baseline)


def test_competitive_antagonist_weakens_learning(gamma1pedc):
    """Review finding H5: the receptor-block path had never run."""
    changes = []
    for manipulation in (
        pharmacology.CONTROL,
        pharmacology.Manipulation(name="antagonist", receptor_block=(("Dop1R1", 20.0),)),
    ):
        network, layer = attach(gamma1pedc, manipulation=manipulation)
        drive(network, layer, [KC_A, KC_B], 0.5)
        drive(network, layer, [KC_A, KC_B, DAN], 1.0)
        changes.append(layer.weight_change("gamma1pedc"))
    control, blocked = changes
    assert control < blocked <= 0


def test_manipulations_naming_absent_receptors_are_refused():
    """Review finding H3: a misspelt knockout used to do nothing, silently."""
    typo = pharmacology.Manipulation(name="typo", receptor_knockout=("Dop1R2",))
    with pytest.raises(ValueError, match="not configured"):
        DopamineConfig(manipulation=typo)
    absent_block = pharmacology.Manipulation(name="x", receptor_block=(("DopEcR", 1.0),))
    with pytest.raises(ValueError, match="not configured"):
        DopamineConfig(manipulation=absent_block)
    autoreceptor_block = pharmacology.Manipulation(name="x", receptor_block=(("Dop2R", 1.0),))
    with pytest.raises(ValueError, match="not configured"):
        DopamineConfig(autoreceptor=None, manipulation=autoreceptor_block)


def test_duplicate_receptor_names_are_refused():
    with pytest.raises(ValueError, match="unique"):
        DopamineConfig(receptors=(DOP1R1, DOP1R1.evolve(ec50=0.3), DOP1R2_GQ))


def test_a_gs_receptor_is_required():
    with pytest.raises(ValueError, match="Gs"):
        DopamineConfig(receptors=(DAN_AUTORECEPTOR,))


def test_coarse_slow_step_is_refused():
    with pytest.raises(ValueError, match="too coarse"):
        DopamineConfig(slow_dt=1.0)


def test_slow_step_must_be_a_multiple_of_the_network_step(gamma1pedc):
    connectome, annotations = gamma1pedc
    network = LIFNetwork(connectome.weights, LIFParams(dt=1e-4))
    with pytest.raises(ValueError, match="multiple"):
        DopamineLayer(network, connectome, annotations, config=DopamineConfig(slow_dt=0.00015))


def test_release_shares_follow_synapse_counts_and_ignore_live_weights(tmp_path):
    """Review finding M2: shares were read from weights the layer itself changes."""
    cells = [
        ("DAN", "PAM01", "left"),
        ("DAN", "PAM01", "left"),
        ("Kenyon_Cell", "KCg-m", "left"),
        (None, "MBON01", "left"),
    ]
    circuit = build_circuit(tmp_path, cells, [(0, 2, 30), (1, 2, 10), (2, 3, 10)])
    connectome, annotations = circuit
    network = LIFNetwork(connectome.weights, LIFParams())
    network.silence([0, 1])  # live weights now say nothing about anatomy
    layer = DopamineLayer(network, connectome, annotations)
    shares = dict(
        zip(layer.targets.dan_index.tolist(), layer.targets.dan_release_weight, strict=True)
    )
    assert shares[0] == pytest.approx(0.75)
    assert shares[1] == pytest.approx(0.25)


def test_hemispheres_are_separate_fields(tmp_path):
    cells = [
        ("DAN", "PPL101", "left"),
        ("DAN", "PPL101", "right"),
        ("Kenyon_Cell", "KCg-m", "left"),
        ("Kenyon_Cell", "KCg-m", "right"),
        (None, "MBON11", "left"),
        (None, "MBON11", "right"),
    ]
    circuit = build_circuit(tmp_path, cells, [(2, 4, 10), (3, 5, 10), (0, 2, 5), (1, 3, 5)])
    network, layer = attach(circuit)
    drive(network, layer, [2, 3], 0.5)
    drive(network, layer, [2, 3, 0], 1.0)  # only the left dopaminergic neuron
    assert layer.weight_change("gamma1pedc", "left") < 0
    assert layer.weight_change("gamma1pedc", "right") == pytest.approx(0.0)


def test_one_dopamine_layer_per_network_and_detach_restores(gamma1pedc):
    connectome, annotations = gamma1pedc
    network, layer = attach(gamma1pedc)
    with pytest.raises(RuntimeError, match="already attached"):
        DopamineLayer(network, connectome, annotations)
    layer.detach()
    network.v[DAN] = network.params.v_threshold + 1e-4
    for _ in range(network.params.delay_steps + 1):
        network.step()
    assert network.g[KC_A] > 0.0  # fast synapses are back
    DopamineLayer(network, connectome, annotations)  # and a new layer may attach


def test_layer_refuses_a_foreign_network(gamma1pedc):
    connectome, _ = gamma1pedc
    _, layer = attach(gamma1pedc)
    other = LIFNetwork(connectome.weights, LIFParams())
    with pytest.raises(RuntimeError, match="different network"):
        layer(other, 0, np.zeros(0, dtype=np.int64))


def test_transporter_block_prolongs_the_transient(gamma1pedc, tmp_path):
    levels = []
    for manipulation in (pharmacology.CONTROL, pharmacology.COCAINE):
        network, layer = attach(gamma1pedc, manipulation=manipulation)
        drive(network, layer, [DAN], 0.5)
        drive(network, layer, [], 3.0)
        levels.append(layer.concentration("gamma1pedc", "left"))
        layer.detach()
    assert levels[1] > levels[0]


def test_reset_restores_weights_and_state(gamma1pedc):
    network, layer = attach(gamma1pedc)
    baseline = plastic_weights(network, layer)
    drive(network, layer, [KC_A], 0.5)
    drive(network, layer, [KC_A, DAN], 1.0)
    layer.reset()
    assert plastic_weights(network, layer) == pytest.approx(baseline)
    assert layer.concentration("gamma1pedc") == pytest.approx(0.0)


def test_empty_compartment_reports_nan_not_zero(gamma1pedc):
    _, layer = attach(gamma1pedc)
    assert math.isnan(layer.weight_change("gamma5"))
