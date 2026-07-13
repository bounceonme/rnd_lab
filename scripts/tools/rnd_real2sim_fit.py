#!/usr/bin/env python3
"""Fit an encoder-domain equivalent actuator model from an RND dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


_TOOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOL_DIR))

from rnd_real2sim.config import Real2SimConfigError, load_experiment_config
from rnd_real2sim.dataset import DatasetError, load_dataset
from rnd_real2sim.identification import (
    IdentificationError,
    identify_dataset,
    model_report,
    save_model,
)


DEFAULT_EXPERIMENT = _TOOL_DIR / "config" / "rnd_real2sim.toml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit delay, effective backlash, response, and current-domain friction without Omniverse."
    )
    parser.add_argument("dataset", help="Completed .npz produced by rnd_real2sim_collect.py.")
    parser.add_argument("--config", default=str(DEFAULT_EXPERIMENT), help="Identification thresholds TOML.")
    parser.add_argument("--output", help="Output model JSON. Defaults next to the dataset.")
    parser.add_argument(
        "--allow-dry-run",
        action="store_true",
        help="Allow fitting synthetic data for pipeline tests. Synthetic output must not be used for training.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Fit an aborted dataset for diagnostics. Quality may be insufficient.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        experiment = load_experiment_config(args.config)
        dataset = load_dataset(args.dataset, allow_incomplete=args.allow_incomplete)
        if dataset.metadata.get("dry_run") and not args.allow_dry_run:
            raise IdentificationError(
                "Refusing to fit synthetic data without --allow-dry-run. Synthetic parameters are pipeline tests only."
            )
        output = Path(args.output) if args.output else dataset.path.with_name(f"{dataset.path.stem}_model.json")
        model = identify_dataset(dataset, experiment.identification)
        saved = save_model(model, output)
        print(model_report(model))
        print(f"\nSaved model: {saved}")
        return 0
    except (Real2SimConfigError, DatasetError, IdentificationError, OSError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
