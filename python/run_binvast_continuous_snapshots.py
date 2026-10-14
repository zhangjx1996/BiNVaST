#!/usr/bin/env python3
"""Run one continuous BinVAST training trajectory for one batch size."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import pandas as pd

from full_synthetic_binvast import run_full_synthetic_binvast_epoch_grid


REQUIRED_TARGET_INPUTS = (
    "target_expression_log1p_cp10k.csv",
    "target_coordinates.csv",
)

REQUIRED_SYNTHETIC_INPUTS = (
    "synthetic_expression.csv",
    "synthetic_labels.csv",
    "synthetic_spatial_coords.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train BinVAST continuously to the largest requested epoch and save "
            "exact model/prediction/latent snapshots at each requested epoch."
        )
    )
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--target-data-dir", type=Path, required=True)
    parser.add_argument("--synthetic-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--snapshot-epochs", type=int, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--graph-k", type=int, default=30)
    parser.add_argument("--expr-latent-dim", type=int, default=4)
    parser.add_argument("--graph-latent-dim", type=int, default=4)
    parser.add_argument("--seed", type=int, default=259)
    return parser.parse_args()


def validate_inputs(input_dir: Path, required_names: tuple[str, ...]) -> None:
    missing = [
        str(input_dir / name)
        for name in required_names
        if not (input_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))


def aggregate_target_proportions(output_dir: Path) -> Path | None:
    columns = []
    for csv_path in sorted(output_dir.glob("epochs_*_batch_*/target_predicted_label_counts_with_legend_percent.csv")):
        table = pd.read_csv(csv_path)
        required = {"predicted_label", "percentage"}
        if not required.issubset(table.columns):
            continue
        columns.append(
            table.set_index("predicted_label")["percentage"].rename(csv_path.parent.name)
        )
    if not columns:
        return None

    summary = pd.concat(columns, axis=1).fillna(0.0).sort_index()
    summary.index.name = "cell_type"
    destination = output_dir / "target_cell_type_proportions_by_epoch_and_batch.csv"
    temporary = output_dir / f".{destination.name}.{os.getpid()}.tmp"
    summary.to_csv(temporary)
    os.replace(temporary, destination)
    return destination


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.expanduser().resolve()
    target_data_dir = args.target_data_dir.expanduser().resolve()
    synthetic_data_dir = args.synthetic_data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    epochs = sorted(set(args.snapshot_epochs))
    if not epochs or epochs[0] < 1:
        raise ValueError("--snapshot-epochs must contain positive integers")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    validate_inputs(target_data_dir, REQUIRED_TARGET_INPUTS)
    validate_inputs(synthetic_data_dir, REQUIRED_SYNTHETIC_INPUTS)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(
        json.dumps(
            {
                "host": socket.gethostname(),
                "lsf_job_id": os.environ.get("LSB_JOBID"),
                "project_dir": str(project_dir),
                "target_data_dir": str(target_data_dir),
                "synthetic_data_dir": str(synthetic_data_dir),
                "output_dir": str(output_dir),
                "batch_size": args.batch_size,
                "snapshot_epochs": epochs,
                "continuous_training_max_epoch": max(epochs),
                "graph_k": args.graph_k,
                "expr_latent_dim": args.expr_latent_dim,
                "graph_latent_dim": args.graph_latent_dim,
                "seed": args.seed,
            },
            indent=2,
        ),
        flush=True,
    )

    result = run_full_synthetic_binvast_epoch_grid(
        project_root=project_dir,
        target_dir_name=target_data_dir,
        synthetic_dir_name=synthetic_data_dir,
        output_name=output_dir,
        snapshot_epochs=epochs,
        batch_size=args.batch_size,
        graph_k=args.graph_k,
        expr_latent_dim=args.expr_latent_dim,
        graph_latent_dim=args.graph_latent_dim,
        seed=args.seed,
    )

    proportion_csv = aggregate_target_proportions(output_dir)
    manifest = {
        "status": "completed",
        "host": socket.gethostname(),
        "lsf_job_id": os.environ.get("LSB_JOBID"),
        "project_dir": str(project_dir),
        "target_data_dir": str(target_data_dir),
        "synthetic_data_dir": str(synthetic_data_dir),
        "output_dir": str(output_dir),
        "batch_size": args.batch_size,
        "snapshot_epochs": epochs,
        "continuous_training_max_epoch": max(epochs),
        "elapsed_minutes": (time.time() - started) / 60.0,
        "proportion_csv": str(proportion_csv) if proportion_csv else None,
        "snapshot_output_dirs": {
            str(epoch): str(snapshot["output_dir"])
            for epoch, snapshot in result["snapshots"].items()
        },
    }
    manifest_path = output_dir / f"job_manifest_batch_{args.batch_size:05d}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
