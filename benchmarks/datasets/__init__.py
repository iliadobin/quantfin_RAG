"""
Dataset generators and validators for benchmark datasets.

Provides tools to:
- Generate synthetic examples for each dataset type
- Validate dataset quality and consistency
- Load/save datasets in standard format
"""
from .generator import DatasetGenerator
from .validator import DatasetValidator
from .loader import load_dataset, save_dataset

__all__ = [
    'DatasetGenerator',
    'DatasetValidator',
    'load_dataset',
    'save_dataset',
]

