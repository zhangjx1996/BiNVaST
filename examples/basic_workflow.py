"""Minimal import-based BiNVaST workflow."""

from binvast import generate_synthetic_data, train_and_predict


synthetic_result = generate_synthetic_data(
    input_data_dir="/path/to/preprocessed_input",
    output_dir="/path/to/synthetic_input",
    pca_base="reference",
    n_pcs=50,
    tau=1.0,
    seed=259,
)

model_result = train_and_predict(
    project_dir="/path/to/binvast_project",
    target_data_dir="/path/to/preprocessed_input",
    synthetic_data_dir=synthetic_result["output_dir"],
    output_dir="/path/to/binvast_output",
    snapshot_epochs=[50, 100, 150, 200],
    batch_size=512,
    graph_k=30,
    seed=259,
)

print(model_result["manifest"]["final_predicted_labels"])
