"""
Dataset loading and saving utilities.

Handles reading/writing datasets in JSON format with validation.
"""
import json
from pathlib import Path
from typing import Union, Any
from benchmarks.schemas import (
    DS1Dataset, DS2Dataset, DS3Dataset, DS4Dataset, DS5Dataset,
    BenchmarkDatasets
)


DATASET_TYPES = {
    'ds1': DS1Dataset,
    'ds2': DS2Dataset,
    'ds3': DS3Dataset,
    'ds4': DS4Dataset,
    'ds5': DS5Dataset,
}


def load_dataset(
    path: Union[str, Path],
    dataset_type: str = 'auto'
) -> Any:
    """
    Load a dataset from JSON file.
    
    Args:
        path: Path to dataset file
        dataset_type: Type of dataset ('ds1', 'ds2', etc.) or 'auto'
        
    Returns:
        Dataset object of appropriate type
    """
    path = Path(path)
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Auto-detect type from filename or data
    if dataset_type == 'auto':
        # Try to infer from filename
        name_lower = path.stem.lower()
        for ds_name in DATASET_TYPES.keys():
            if ds_name in name_lower:
                dataset_type = ds_name
                break
        
        # Try to infer from data
        if dataset_type == 'auto' and 'name' in data:
            name = data['name'].lower()
            if 'factual' in name or 'qa' in name:
                dataset_type = 'ds1'
            elif 'retrieval' in name or 'qrel' in name:
                dataset_type = 'ds2'
            elif 'unanswerable' in name or 'trap' in name:
                dataset_type = 'ds3'
            elif 'multihop' in name or 'multi' in name:
                dataset_type = 'ds4'
            elif 'structured' in name or 'extraction' in name:
                dataset_type = 'ds5'
    
    # Load with appropriate schema
    if dataset_type in DATASET_TYPES:
        dataset_class = DATASET_TYPES[dataset_type]
        return dataset_class(**data)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Use one of {list(DATASET_TYPES.keys())}")


def save_dataset(
    dataset: Any,
    path: Union[str, Path],
    indent: int = 2
):
    """
    Save a dataset to JSON file.
    
    Args:
        dataset: Dataset object (DS1Dataset, DS2Dataset, etc.)
        path: Output path
        indent: JSON indentation
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(
            dataset.model_dump(),
            f,
            indent=indent,
            ensure_ascii=False
        )


def load_all_datasets(
    datasets_dir: Union[str, Path]
) -> BenchmarkDatasets:
    """
    Load all datasets from a directory.
    
    Expected structure:
    datasets_dir/
        ds1_*.json
        ds2_*.json
        ds3_*.json
        ds4_*.json
        ds5_*.json
    
    Args:
        datasets_dir: Directory containing dataset files
        
    Returns:
        BenchmarkDatasets container with all loaded datasets
    """
    datasets_dir = Path(datasets_dir)
    
    result = BenchmarkDatasets()
    
    # Try to load each dataset type
    for ds_name, ds_class in DATASET_TYPES.items():
        # Find file matching pattern
        pattern = f"{ds_name}_*.json"
        files = list(datasets_dir.glob(pattern))
        
        if files:
            # Load the first matching file
            dataset = load_dataset(files[0], ds_name)
            setattr(result, ds_name, dataset)
    
    return result


def save_all_datasets(
    datasets: BenchmarkDatasets,
    output_dir: Union[str, Path],
    indent: int = 2
):
    """
    Save all datasets to a directory.
    
    Args:
        datasets: BenchmarkDatasets container
        output_dir: Output directory
        indent: JSON indentation
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for ds_name in ['ds1', 'ds2', 'ds3', 'ds4', 'ds5']:
        dataset = getattr(datasets, ds_name, None)
        if dataset:
            filename = f"{ds_name}_{dataset.name.lower().replace(' ', '_')}.json"
            save_dataset(dataset, output_dir / filename, indent)

