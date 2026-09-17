"""Run the mushroom body validation suite and print a report.

Three experiments on the full FlyWire v783 mushroom body:

1. one forward pairing in γ1pedc, compared with Hige et al. 2015;
2. the same pairing without the Gs receptor (Dop1R1 null);
3. the pairing interval curve in γ5, compared with Handler et al. 2019.

Takes about ten minutes on a laptop. Usage::

    uv run python scripts/validate_mushroom_body.py [--data-dir data/raw]
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from flyneuromod.data.annotations import load_annotations
from flyneuromod.data.connectome import load_connectome
from flyneuromod.experiments.conditioning import run_conditioning
from flyneuromod.experiments.mushroom_body import extract_mushroom_body
from flyneuromod.experiments.protocol import ConditioningProtocol
from flyneuromod.experiments.timing import describe, timing_curve
from flyneuromod.neuromod import pharmacology
from flyneuromod.neuromod.dopamine import DopamineConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    start = time.time()
    connectome = load_connectome(
        args.data_dir / "Completeness_783.csv", args.data_dir / "Connectivity_783.parquet"
    )
    annotations = load_annotations(args.data_dir / "flywire_annotations_v783.tsv")
    mushroom_body = extract_mushroom_body(connectome, annotations)
    protocol = ConditioningProtocol()

    control = run_conditioning(mushroom_body, protocol=protocol, seed=args.seed)
    print("1. forward pairing, γ1pedc (Hige et al. 2015: ~90% synaptic, ~80% spiking loss)")
    print("   " + control.summary(), flush=True)

    knockout = run_conditioning(
        mushroom_body,
        protocol=protocol,
        config=DopamineConfig(manipulation=pharmacology.DOP1R1_KNOCKOUT),
        seed=args.seed,
    )
    print("2. same pairing, Dop1R1 null (expected: no learning)")
    print("   " + knockout.summary(), flush=True)

    points = timing_curve(mushroom_body, intervals=(-1.2, -0.5, 0.0, 0.5, 6.0), seed=args.seed)
    print("3. pairing interval, γ5 (Handler et al. 2019: backward +, forward -, late 0)")
    print("   " + describe(points).replace("\n", "\n   "), flush=True)

    print(f"\ndone in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
