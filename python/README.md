# Python model workflow

The Python portion of BinVAST performs distribution alignment, synthetic-data generation, dual-branch VAE training, spatial-graph integration, internal validation, and query-cell prediction.

Run the workflow from the repository root:

```bash
python python/generate_synthetic_csv.py \
  --input-data-dir /path/to/binvast_preprocessed_input \
  --output-dir /path/to/binvast_synthetic_input

python python/run_binvast_continuous_snapshots.py \
  --project-dir /path/to/binvast_project \
  --target-data-dir /path/to/binvast_preprocessed_input \
  --synthetic-data-dir /path/to/binvast_synthetic_input \
  --output-dir /path/to/binvast_output \
  --snapshot-epochs 50 100 150 200 \
  --batch-size 512
```

Files:

- `generate_synthetic_csv.py`: generates query-matched synthetic expression, coordinates, and labels.
- `binvast_model_core.py`: contains preprocessing, graph construction, the expression and graph VAEs, classification heads, losses, training, evaluation, and prediction utilities.
- `full_synthetic_binvast.py`: orchestrates platform-neutral full-synthetic training, snapshot export, and query prediction.
- `run_binvast_continuous_snapshots.py`: command-line entry point for continuous training and epoch snapshots.

See the repository-level README for setup, input schemas, parameters, and outputs.
