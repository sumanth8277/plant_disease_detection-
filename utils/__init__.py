"""
Utilities and helper functions package (Phases 2 & 3).
Exports dataset auditing, splitting, training, evaluation, and single-image inference utilities.
"""

from utils.dataset_inspector import PLANTVILLAGE_SOURCE_INFO, inspect_dataset
from utils.dataset_splitter import prepare_dataset_split
from utils.evaluate_model import evaluate_trained_model
from utils.inspect_dataset import inspect_dataset_directory
from utils.predict_image import predict_single_leaf_image
from utils.prepare_dataset import prepare_dataset_manifest
from utils.train_model import train_disease_classifier

__all__ = [
    "inspect_dataset",
    "inspect_dataset_directory",
    "prepare_dataset_split",
    "prepare_dataset_manifest",
    "train_disease_classifier",
    "evaluate_trained_model",
    "predict_single_leaf_image",
    "PLANTVILLAGE_SOURCE_INFO",
]
