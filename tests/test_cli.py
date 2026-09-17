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


def test_unknown_manipulation_is_rejected_before_any_data_is_loaded(tmp_path, capsys):
    """Review finding L8: a typo used to surface only after loading 130 MB."""
    with pytest.raises(SystemExit) as exit_info:
        main(["--data-dir", str(tmp_path), "conditioning", "--manipulation", "nonexistent"])
    assert exit_info.value.code == 2
    assert "dop1r1-null" in capsys.readouterr().err


def test_timing_accepts_custom_intervals():
    args = build_parser().parse_args(["timing", "--intervals", "-1", "0", "2.5"])
    assert args.intervals == [-1.0, 0.0, 2.5]


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


def test_downloads_are_pinned_and_checksummed():
    """Review finding M6: URLs pointed at a moving branch and checksums went unused."""
    for data_file in FILES:
        assert "/main/" not in data_file.url
        assert len(data_file.sha256) == 64


def test_existing_files_are_verified_against_the_pinned_checksum(tmp_path, caplog):
    from flyneuromod.data.download import download, sha256_of

    for data_file in FILES:
        (tmp_path / data_file.name).write_text("not the real data")
    with caplog.at_level("WARNING"):
        paths = download(tmp_path)
    assert len(paths) == len(FILES)
    assert "does not match the pinned checksum" in caplog.text
    assert sha256_of(paths[FILES[0].name]) != FILES[0].sha256
