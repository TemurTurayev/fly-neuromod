import numpy as np
import pandas as pd
import pytest

from flyneuromod.data.connectome import load_connectome
from flyneuromod.engine.lif import LIFNetwork
from flyneuromod.engine.params import LIFParams

from .conftest import COMPLETENESS, CONNECTIVITY, requires_data


def test_loads_neurons_and_signed_weights(tiny_tables):
    connectome = load_connectome(*tiny_tables)
    assert connectome.n_neurons == 4
    assert connectome.n_connections == 3
    weights = connectome.weights.toarray()
    assert weights[1, 0] == 5  # [post, pre]
    assert weights[2, 1] == -3  # inhibitory connection keeps its sign
    assert weights[0, 1] == 0  # no reverse connection


def test_index_lookup_and_missing_ids(tiny_tables):
    connectome = load_connectome(*tiny_tables)
    assert connectome.indices_of([1002, 1000]).tolist() == [2, 0]
    with pytest.raises(KeyError):
        connectome.indices_of([999999])


def test_subnetwork_keeps_internal_connections_only(tiny_tables):
    connectome = load_connectome(*tiny_tables)
    sub = connectome.subnetwork([1001, 1002])
    assert sub.n_neurons == 2
    assert sub.weights.toarray()[1, 0] == -3  # 1001 -> 1002 survives
    assert sub.weights.nnz == 1  # 1000 -> 1001 is dropped


def test_subnetwork_rejects_duplicates(tiny_tables):
    connectome = load_connectome(*tiny_tables)
    with pytest.raises(ValueError):
        connectome.subnetwork([1001, 1001])


def test_missing_file_raises_with_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="download"):
        load_connectome(tmp_path / "nope.csv", tmp_path / "nope.parquet")


def test_missing_columns_are_reported(tmp_path, tiny_tables):
    completeness, _ = tiny_tables
    broken = tmp_path / "broken.parquet"
    pd.DataFrame({"Presynaptic_ID": [1]}).to_parquet(broken)
    with pytest.raises(ValueError, match="missing columns"):
        load_connectome(completeness, broken)


def test_mismatched_releases_are_detected(tmp_path, tiny_tables):
    completeness, _ = tiny_tables
    out_of_range = pd.DataFrame(
        {
            "Presynaptic_ID": [1],
            "Postsynaptic_ID": [2],
            "Presynaptic_Index": [0],
            "Postsynaptic_Index": [99],
            "Connectivity": [1],
            "Excitatory": [1],
            "Excitatory x Connectivity": [1],
        }
    )
    path = tmp_path / "mismatch.parquet"
    out_of_range.to_parquet(path)
    with pytest.raises(ValueError, match="different connectome releases"):
        load_connectome(completeness, path)


def test_connectome_drives_a_simulation(tiny_tables):
    """The loaded matrix plugs straight into the network: 0 -> 1 transmission works."""
    connectome = load_connectome(*tiny_tables)
    params = LIFParams()
    net = LIFNetwork(connectome.weights, params, rng=np.random.default_rng(0))
    net.v[0] = params.v_threshold + 1e-4
    net.step()
    for _ in range(params.delay_steps):
        net.step()
    assert net.g[1] == pytest.approx(5 * params.w_synapse)


@requires_data
def test_real_release_has_expected_size():
    connectome = load_connectome(COMPLETENESS, CONNECTIVITY)
    assert connectome.n_neurons == 138_639
    assert connectome.n_connections == 15_091_983
