#!/usr/bin/env python3
"""Prepare or execute the independent M7 SearchQA development path."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from textskill_optimizer.paper.searchqa_experiment import (
    detect_coco_model,
    prepare_searchqa_mechanism_smoke,
    prepare_zero_call_searchqa_experiment,
    run_searchqa_experiment,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    zero = subparsers.add_parser(
        "zero-call",
        help="freeze and execute the scripted full-call-graph dry-run",
    )
    _add_data_args(zero)
    zero.add_argument(
        "--mechanism-smoke-scope",
        action="store_true",
        help="use the non-claim-eligible 2-epoch plan that the paid smoke will use",
    )

    prepare = subparsers.add_parser(
        "prepare-smoke",
        help="freeze the first paid mechanism smoke without executing it",
    )
    _add_data_args(prepare)
    prepare.add_argument(
        "--zero-cost-receipt",
        type=Path,
        required=True,
        help="authorized M6 receipt bound to the current clean commit",
    )
    prepare.add_argument(
        "--mechanism-dry-run-receipt",
        type=Path,
        required=True,
        help="two-epoch zero-call receipt used to derive paid caps",
    )
    prepare.add_argument("--target-reasoning", default="not_configured")
    prepare.add_argument("--optimizer-reasoning", default="medium")
    prepare.add_argument("--safety-factor", type=float, default=1.5)

    execute = subparsers.add_parser(
        "run",
        help="execute one already-frozen preregistration",
    )
    execute.add_argument("--preregistration", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "zero-call":
            preregistration = prepare_zero_call_searchqa_experiment(
                run_dir=args.run_dir,
                train_path=args.train,
                selection_path=args.selection,
                materialization_receipt_path=args.materialization_receipt,
                mechanism_smoke_scope=args.mechanism_smoke_scope,
            )
            output = run_searchqa_experiment(preregistration)
        elif args.command == "prepare-smoke":
            target_model = detect_coco_model()
            optimizer_model = os.environ.get("EXTERNAL_LLM_MODEL", "").strip()
            output = prepare_searchqa_mechanism_smoke(
                run_dir=args.run_dir,
                train_path=args.train,
                selection_path=args.selection,
                target_model=target_model,
                target_reasoning=args.target_reasoning,
                optimizer_model=optimizer_model,
                optimizer_reasoning=args.optimizer_reasoning,
                safety_factor=args.safety_factor,
                zero_cost_receipt_path=args.zero_cost_receipt,
                materialization_receipt_path=args.materialization_receipt,
                mechanism_dry_run_receipt_path=args.mechanism_dry_run_receipt,
            )
        else:
            output = run_searchqa_experiment(args.preregistration)
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 2
    print(str(output))
    return 0


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--materialization-receipt", type=Path, required=True)


if __name__ == "__main__":
    raise SystemExit(main())
