"""The dopamine layer wired to a spiking network."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flyneuromod.data.annotations import load_annotations
from flyneuromod.data.connectome import load_connectome
from flyneuromod.engine.lif import LIFNetwork
from flyneuromod.engine.params import LIFParams
from flyneuromod.neuromod import pharmacology
from flyneuromod.neuromod.dopamine import DopamineConfig, DopamineLayer

# toy mushroom body: one PPL101 dopaminergic neuron, three Kenyon cells, one MBON11
DAN, KC_A, KC_B, KC_C, MBON = 0, 1, 2, 3, 4
ROOT_IDS = [2000, 2001, 2002, 2003, 2004]


@pytest.fixture
def toy_mushroom_body(tmp_path: Path):
    """Kenyon cells drive the output neuron; the dopaminergic neuron innervates them."""
    completeness = tmp_path / "completeness.csv"
    pd.DataFrame({"Completed": [True] * 5}, index=ROOT_IDS).to_csv(completeness)

    pre = [KC_A, KC_B, KC_C, DAN, DAN, DAN]
    post = [MBON, MBON, MBON, KC_A, KC_B, KC_C]
    counts = [10, 10, 10, 5, 5, 5]
    connections = pd.DataFrame(
        {
            "Presynaptic_ID": [ROOT_IDS[i] for i in pre],
            "Postsynaptic_ID": [ROOT_IDS[i] for i in post],
            "Presynaptic_Index": pre,
            "Postsynaptic_Index": post,
            "Connectivity": counts,
            "Excitatory": [1] * 6,
            "Excitatory x Connectivity": counts,
        }
    )
    connectivity = tmp_path / "connectivity.parquet"
    connections.to_parquet(connectivity)

    annotations_path = tmp_path / "annotations.tsv"
    pd.DataFrame(
        {
            "root_id": ROOT_IDS,
            "cell_class": ["DAN", "Kenyon_Cell", "Kenyon_Cell", "Kenyon_Cell", None],
            "cell_type": ["PPL101", "KCg-m", "KCg-m", "KCab", "MBON11"],
            "top_nt": ["dopamine"] * 4 + ["gaba"],
            "side": ["left"] * 5,
        }
    ).to_csv(annotations_path, sep="\t", index=False)

    connectome = load_connectome(completeness, connectivity)
    annotations = load_annotations(annotations_path)
    return connectome, annotations


def build(toy, **config_kwargs):
    connectome, annotations = toy
    params = LIFParams()
    network = LIFNetwork(connectome.weights, params, rng=np.random.default_rng(0))
    layer = DopamineLayer(
        network, connectome, annotations, config=DopamineConfig(**config_kwargs)
    )
    return network, layer


def drive(network, layer, neurons, seconds):
    """Force the listed neurons to spike every step for the given time."""
    steps = int(round(seconds / network.params.dt))
    for _ in range(steps):
        network.v[list(neurons)] = network.params.v_threshold + 1e-4
        spiking = network.step()
        layer(network, 0, spiking)


def test_layer_finds_the_mushroom_body_structure(toy_mushroom_body):
    _, layer = build(toy_mushroom_body)
    assert layer.targets.dan_index.tolist() == [DAN]
    assert layer.targets.kenyon_index.tolist() == [KC_A, KC_B, KC_C]
    assert len(layer.targets.synapse_position) == 3  # three Kenyon cell -> MBON synapses
    assert set(layer.targets.synapse_compartment.tolist()) == {
        layer.atlas.index_of("gamma1pedc")
    }


def test_dopaminergic_spikes_raise_concentration_in_their_compartment(toy_mushroom_body):
    network, layer = build(toy_mushroom_body)
    drive(network, layer, [DAN], 0.5)
    concentrations = layer.concentrations()
    assert concentrations["gamma1pedc"] > 0.1
    assert concentrations["gamma5"] == pytest.approx(0.0)


def test_dopamine_does_not_act_as_a_fast_synapse(toy_mushroom_body):
    """Dopaminergic connections are removed from the spiking layer."""
    network, layer = build(toy_mushroom_body)
    drive(network, layer, [DAN], 0.05)
    assert network.g[KC_A] == pytest.approx(0.0)
    assert layer.concentrations()["gamma1pedc"] > 0.0


def test_fast_synapses_can_be_kept_for_cotransmission(toy_mushroom_body):
    network, _ = build(toy_mushroom_body, remove_fast_dopamine_synapses=False)
    network.v[DAN] = network.params.v_threshold + 1e-4
    network.step()
    for _ in range(network.params.delay_steps):
        network.step()
    assert network.g[KC_A] > 0.0


def test_forward_pairing_depresses_only_the_paired_kenyon_cells(toy_mushroom_body):
    """Odour then dopamine: the cells that carried the odour lose their output."""
    network, layer = build(toy_mushroom_body)
    baseline = network.synapse_weight[layer.targets.synapse_position].copy()

    drive(network, layer, [KC_A, KC_B], 0.5)  # odour: two of three Kenyon cells
    drive(network, layer, [DAN], 2.0)  # then the teaching signal

    weights = network.synapse_weight[layer.targets.synapse_position]
    paired = np.isin(layer.targets.synapse_presynaptic_slot, [0, 1])
    # this toy network only checks the mechanism and its specificity; the size of
    # the change is calibrated against Hige et al. 2015 in experiments/conditioning.py
    assert np.all(weights[paired] < baseline[paired])
    assert weights[~paired] == pytest.approx(baseline[~paired])
    assert layer.weight_change("gamma1pedc") < -0.01
    assert layer.weight_change_of_cells(np.array([2]), "gamma1pedc") == pytest.approx(0.0)


def test_dopamine_alone_changes_nothing(toy_mushroom_body):
    network, layer = build(toy_mushroom_body)
    baseline = network.synapse_weight[layer.targets.synapse_position].copy()
    drive(network, layer, [DAN], 2.0)
    assert network.synapse_weight[layer.targets.synapse_position] == pytest.approx(baseline)


def test_odour_alone_changes_nothing(toy_mushroom_body):
    network, layer = build(toy_mushroom_body)
    baseline = network.synapse_weight[layer.targets.synapse_position].copy()
    drive(network, layer, [KC_A, KC_B, KC_C], 2.0)
    assert network.synapse_weight[layer.targets.synapse_position] == pytest.approx(baseline)


def test_gamma1pedc_does_not_potentiate_on_backward_pairing(toy_mushroom_body):
    """Hige et al. 2015 found no change for backward pairing in this compartment."""
    network, layer = build(toy_mushroom_body)
    baseline = network.synapse_weight[layer.targets.synapse_position].copy()
    drive(network, layer, [DAN], 1.0)  # dopamine first
    drive(network, layer, [KC_A], 0.2)  # then the odour
    weights = network.synapse_weight[layer.targets.synapse_position]
    assert np.all(weights <= baseline + 1e-12)


def test_receptor_knockout_abolishes_learning(toy_mushroom_body):
    """Without the Gs receptor there is no cAMP, so pairing writes nothing."""
    network, layer = build(toy_mushroom_body, manipulation=pharmacology.DOP1R1_KNOCKOUT)
    baseline = network.synapse_weight[layer.targets.synapse_position].copy()
    drive(network, layer, [KC_A, KC_B], 0.5)
    drive(network, layer, [DAN], 2.0)
    assert network.synapse_weight[layer.targets.synapse_position] == pytest.approx(baseline)


def test_transporter_block_prolongs_the_dopamine_transient(toy_mushroom_body):
    control_net, control = build(toy_mushroom_body)
    cocaine_net, cocaine = build(toy_mushroom_body, manipulation=pharmacology.COCAINE)
    for net, layer in ((control_net, control), (cocaine_net, cocaine)):
        drive(net, layer, [DAN], 0.2)
        drive(net, layer, [], 2.0)
    assert cocaine.concentrations()["gamma1pedc"] > control.concentrations()["gamma1pedc"]


def test_reset_restores_weights_and_state(toy_mushroom_body):
    network, layer = build(toy_mushroom_body)
    baseline = network.synapse_weight[layer.targets.synapse_position].copy()
    drive(network, layer, [KC_A], 0.5)
    drive(network, layer, [DAN], 2.0)
    layer.reset()
    assert network.synapse_weight[layer.targets.synapse_position] == pytest.approx(baseline)
    assert layer.concentrations()["gamma1pedc"] == pytest.approx(0.0)


def test_slow_step_must_be_a_multiple_of_the_network_step(toy_mushroom_body):
    connectome, annotations = toy_mushroom_body
    network = LIFNetwork(connectome.weights, LIFParams(dt=1e-4))
    with pytest.raises(ValueError, match="multiple"):
        DopamineLayer(
            network, connectome, annotations, config=DopamineConfig(slow_dt=0.00015)
        )


def test_config_requires_a_gs_receptor():
    from flyneuromod.neuromod.constants import DAN_AUTORECEPTOR as DOP2R

    with pytest.raises(ValueError, match="Gs"):
        DopamineConfig(receptors=(DOP2R,))
