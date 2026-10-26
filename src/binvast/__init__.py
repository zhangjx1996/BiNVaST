"""Public Python interface for BiNVaST."""

from .model import BinVAST, TrainConfig, load_trained_model, predict_labels
from .pipeline import run_full_synthetic_binvast, run_full_synthetic_binvast_epoch_grid
from .synthetic import generate_synthetic_data
from .workflow import train_and_predict

__version__ = "0.1.0"

__all__ = [
    "BinVAST",
    "TrainConfig",
    "generate_synthetic_data",
    "load_trained_model",
    "predict_labels",
    "run_full_synthetic_binvast",
    "run_full_synthetic_binvast_epoch_grid",
    "train_and_predict",
]
