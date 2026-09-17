"""Shared fixtures: tiny synthetic connectomes and the real data files when present."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
COMPLETENESS = DATA_DIR / "Completeness_783.csv"
CONNECTIVITY = DATA_DIR / "Connectivity_783.parquet"
ANNOTATIONS = DATA_DIR / "flywire_annotations_v783.tsv"

requires_data = pytest.mark.skipif(
    not (COMPLETENESS.is_file() and CONNECTIVITY.is_file() and ANNOTATIONS.is_file()),
    reason="connectome data not downloaded (run `flyneuromod download`)",
)


@pytest.fixture
def tiny_tables(tmp_path: Path) -> tuple[Path, Path]:
    """Four neurons: 0 -> 1 (excitatory), 1 -> 2 (inhibitory), 2 -> 3 (excitatory)."""
    root_ids = [1000, 1001, 1002, 1003]
    completeness = tmp_path / "completeness.csv"
    pd.DataFrame({"Completed": [True] * 4}, index=root_ids).to_csv(completeness)

    connections = pd.DataFrame(
        {
            "Presynaptic_ID": [1000, 1001, 1002],
            "Postsynaptic_ID": [1001, 1002, 1003],
            "Presynaptic_Index": [0, 1, 2],
            "Postsynaptic_Index": [1, 2, 3],
            "Connectivity": [5, 3, 7],
            "Excitatory": [1, -1, 1],
            "Excitatory x Connectivity": [5, -3, 7],
        }
    )
    connectivity = tmp_path / "connectivity.parquet"
    connections.to_parquet(connectivity)
    return completeness, connectivity


@pytest.fixture
def tiny_annotations(tmp_path: Path) -> Path:
    """Annotations mixing a real DAN, Kenyon cells mislabelled as dopaminergic, and an MBON."""
    table = pd.DataFrame(
        {
            "root_id": [1000, 1001, 1002, 1003],
            "cell_class": ["DAN", "Kenyon_Cell", "Kenyon_Cell", None],
            "cell_type": ["PPL101", "KCg-m", "KCab", "MBON01"],
            "top_nt": ["dopamine", "dopamine", "dopamine", "glutamate"],
            "side": ["left", "left", "right", "right"],
        }
    )
    path = tmp_path / "annotations.tsv"
    table.to_csv(path, sep="\t", index=False)
    return path
