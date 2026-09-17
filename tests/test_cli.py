"""Command line interface."""

import pytest

from flyneuromod.cli import build_parser, main
from flyneuromod.data.download import FILES, total_size_mb


def test_parser_requires_a_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_conditioning_defaults():
    args = build_parser().parse_args(["conditioning"])
    assert args.compartment == "gamma1pedc"
    assert args.readout == "MBON11"
    assert args.dan == "PPL101"
    assert args.manipulation == "control"


def test_timing_defaults_to_the_compartment_where_the_sign_flip_was_measured():
    args = build_parser().parse_args(["timing"])
    assert args.compartment == "gamma5"
    assert args.dan == "PAM01"


def test_unknown_manipulation_is_reported(tmp_path, capsys):
    exit_code = main(
        ["--data-dir", str(tmp_path), "conditioning", "--manipulation", "nonexistent"]
    )
    assert exit_code == 1
    assert "error" in capsys.readouterr().err


def test_missing_data_is_reported(tmp_path, capsys):
    exit_code = main(["--data-dir", str(tmp_path), "info"])
    assert exit_code == 1
    assert "download" in capsys.readouterr().err


def test_download_manifest_is_complete():
    assert len(FILES) == 3
    assert total_size_mb() > 100
    for data_file in FILES:
        assert data_file.url.startswith("https://")
        assert data_file.source
