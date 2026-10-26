"""Additional command-line entry points for BinVAST."""

from __future__ import annotations

import argparse
from pathlib import Path

from .model import predict_labels


def predict_main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict labels for prepared query data using a trained BinVAST checkpoint."
    )
    parser.add_argument("--target-expression", type=Path, required=True)
    parser.add_argument("--target-coordinates", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=Path("target_predicted_labels.csv"))
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--target-input-normalized", choices=("counts", "log"))
    args = parser.parse_args()
    predict_labels(
        target_count_csv=args.target_expression,
        target_coord_csv=args.target_coordinates,
        ckpt_path=args.checkpoint,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
        target_input_normalized=args.target_input_normalized,
    )
