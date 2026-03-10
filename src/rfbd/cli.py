"""Top-level CLI entrypoint for rfbd."""

from __future__ import annotations

import argparse

from .dataset_build import add_dataset_subparser
from .eval import add_eval_subparser
from .extract import add_extract_subparser
from .no_drone_batches import add_no_drone_batches_subparser
from .training import add_train_subparser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rfbd", description="RFBinaryDetect CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_extract_subparser(subparsers)
    add_no_drone_batches_subparser(subparsers)
    add_dataset_subparser(subparsers)
    add_train_subparser(subparsers)
    add_eval_subparser(subparsers)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return
    func(args)


if __name__ == "__main__":
    main()
