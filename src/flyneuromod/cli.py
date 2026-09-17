"""Command line entry point: ``flyneuromod``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .data.annotations import load_annotations
from .data.connectome import load_connectome
from .data.download import DEFAULT_DATA_DIR, FILES, download, total_size_mb
from .experiments.conditioning import ConditioningProtocol, run_conditioning
from .experiments.mushroom_body import extract_mushroom_body
from .experiments.timing import describe, timing_curve
from .neuromod import pharmacology
from .neuromod.dopamine import DopamineConfig
from .neuromod.mb_atlas import load_atlas

logger = logging.getLogger("flyneuromod")


def _load(data_dir: Path):
    connectome = load_connectome(
        data_dir / "Completeness_783.csv", data_dir / "Connectivity_783.parquet"
    )
    annotations = load_annotations(data_dir / "flywire_annotations_v783.tsv")
    return connectome, annotations


def _cmd_download(args: argparse.Namespace) -> int:
    print(f"Downloading {len(FILES)} files (~{total_size_mb()} MB) into {args.data_dir}")
    for data_file in FILES:
        print(f"  {data_file.name}: {data_file.source}")
    paths = download(args.data_dir, force=args.force)
    for name, path in paths.items():
        print(f"  ready: {name} ({path.stat().st_size / 1e6:.0f} MB)")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    connectome, annotations = _load(args.data_dir)
    atlas = load_atlas()
    dans = annotations.dopaminergic()
    predicted = annotations.predicted_transmitter("dopamine")
    kenyon = set(annotations.kenyon_cells().tolist())

    print(f"connectome           {connectome.n_neurons} neurons, {connectome.n_connections} pairs")
    print(f"curated dopaminergic {len(dans)} neurons in {annotations.types_in(dans).size} types")
    print(f"predicted dopamine   {len(predicted)} neurons")
    print(
        f"  of which Kenyon cells: {len(kenyon & set(predicted.tolist()))}"
        "  <- why predicted transmitter must not be used to pick dopamine sources"
    )
    print(f"Kenyon cells         {len(kenyon)}")
    print(f"output neurons       {len(annotations.mbons())}")
    print(f"compartments         {len(atlas)}")
    for compartment in atlas.compartments:
        print(
            f"  {compartment.label:<10} DAN {','.join(compartment.dan_types):<16}"
            f" MBON {','.join(compartment.mbon_types) or '-':<18} {compartment.valence or ''}"
        )
    return 0


def _cmd_conditioning(args: argparse.Namespace) -> int:
    connectome, annotations = _load(args.data_dir)
    mushroom_body = extract_mushroom_body(connectome, annotations)
    config = DopamineConfig(manipulation=pharmacology.get(args.manipulation))
    result = run_conditioning(
        mushroom_body,
        compartment=args.compartment,
        readout_type=args.readout,
        dan_type=args.dan,
        protocol=ConditioningProtocol(dopamine_onset=args.onset),
        config=config,
        seed=args.seed,
    )
    print(result.summary())
    return 0


def _cmd_timing(args: argparse.Namespace) -> int:
    connectome, annotations = _load(args.data_dir)
    mushroom_body = extract_mushroom_body(connectome, annotations)
    points = timing_curve(
        mushroom_body,
        compartment=args.compartment,
        readout_type=args.readout,
        dan_type=args.dan,
        seed=args.seed,
    )
    print(describe(points))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flyneuromod",
        description="Dopamine neuromodulation for connectome models of the fly brain",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--quiet", action="store_true", help="only print results")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="fetch the connectome files")
    download_parser.add_argument("--force", action="store_true", help="re-download existing files")
    download_parser.set_defaults(func=_cmd_download)

    info_parser = subparsers.add_parser("info", help="summarise the loaded connectome")
    info_parser.set_defaults(func=_cmd_info)

    conditioning_parser = subparsers.add_parser(
        "conditioning", help="pair an odour with dopamine and measure the output neuron"
    )
    conditioning_parser.add_argument("--compartment", default="gamma1pedc")
    conditioning_parser.add_argument("--readout", default="MBON11")
    conditioning_parser.add_argument("--dan", default="PPL101")
    conditioning_parser.add_argument(
        "--onset", type=float, default=0.2, help="seconds from odour to dopamine (negative: before)"
    )
    conditioning_parser.add_argument(
        "--manipulation", default="control", help=f"one of {sorted(pharmacology.CATALOGUE)}"
    )
    conditioning_parser.add_argument("--seed", type=int, default=0)
    conditioning_parser.set_defaults(func=_cmd_conditioning)

    timing_parser = subparsers.add_parser(
        "timing", help="measure the sign of plasticity against the pairing interval"
    )
    timing_parser.add_argument("--compartment", default="gamma5")
    timing_parser.add_argument("--readout", default="MBON01")
    timing_parser.add_argument("--dan", default="PAM01")
    timing_parser.add_argument("--seed", type=int, default=0)
    timing_parser.set_defaults(func=_cmd_timing)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s"
    )
    try:
        return int(args.func(args))
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
