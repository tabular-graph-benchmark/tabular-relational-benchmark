import os
import glob
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Dataset loader that only scans and loads raw dataset files."""

    def __init__(self, base_dir):
        """Initialize the dataset loader.

        Args:
            base_dir: Base directory containing datasets.
        """

        self.base_dir = Path(base_dir)
        self.dataset_info = {}  # stores dataset metadata
        self._scan_datasets()

    def _scan_datasets(self):
        """Scan all available datasets under the expected folder structure."""

        for size in ["small_datasets", "large_datasets"]:
            size_dir = self.base_dir / size
            if not size_dir.exists():
                continue

            for task_type in ["binclass", "multiclass", "regression"]:
                task_dir = size_dir / task_type
                if not task_dir.exists():
                    continue

                for feature_type in ["numerical", "categorical", "balanced"]:
                    feature_dir = task_dir / feature_type
                    if not feature_dir.exists():
                        continue

                    # Collect dataset subfolders under this category.
                    # Note: exclude 'covtype' to match upstream behavior.
                    dataset_dirs = [
                        d for d in feature_dir.iterdir() if d.is_dir() and d.name != "covtype"
                    ]

                    for dataset_dir in dataset_dirs:
                        dataset_name = dataset_dir.name
                        self.dataset_info[dataset_name] = {
                            "size": size,
                            "task_type": task_type,
                            "feature_type": feature_type,
                            "path": str(dataset_dir),
                        }

        logger.info("Found %s dataset(s)", len(self.dataset_info))

    def get_dataset_categories(self):
        """Return dataset category counts.

        Returns:
            dict: A nested dict containing dataset counts per category.
        """

        categories = {
            "small_datasets": {
                "binclass": {"numerical": 0, "categorical": 0, "balanced": 0},
                "multiclass": {"numerical": 0, "categorical": 0, "balanced": 0},
                "regression": {"numerical": 0, "categorical": 0, "balanced": 0},
            },
            "large_datasets": {
                "binclass": {"numerical": 0, "categorical": 0, "balanced": 0},
                "multiclass": {"numerical": 0, "categorical": 0, "balanced": 0},
                "regression": {"numerical": 0, "categorical": 0, "balanced": 0},
            },
        }

        for info in self.dataset_info.values():
            size = info["size"]
            task_type = info["task_type"]
            feature_type = info["feature_type"]
            categories[size][task_type][feature_type] += 1

        return categories

    def get_datasets_by_category(self, size=None, task_type=None, feature_type=None):
        """Filter datasets by category.

        Args:
            size: Dataset size bucket ('small_datasets' or 'large_datasets').
            task_type: Task type ('binclass', 'multiclass', or 'regression').
            feature_type: Feature type ('numerical', 'categorical', or 'balanced').

        Returns:
            list: Dataset names matching the provided filters.
        """

        filtered_datasets = []

        for name, info in self.dataset_info.items():
            if size and info["size"] != size:
                continue
            if task_type and info["task_type"] != task_type:
                continue
            if feature_type and info["feature_type"] != feature_type:
                continue

            filtered_datasets.append(name)

        return filtered_datasets

    def load_dataset(self, dataset_name):
        """Load the raw dataset file for a given dataset name.

        Args:
            dataset_name: Dataset name.

        Returns:
            dict: Dataset info and the raw pandas DataFrame.
        """

        if dataset_name not in self.dataset_info:
            raise ValueError(f"Dataset not found: {dataset_name}")

        info = self.dataset_info[dataset_name]
        dataset_path = Path(info["path"])

        # Try to locate a data file.
        data_files = list(dataset_path.glob("*.csv"))
        if not data_files:
            data_files = list(dataset_path.glob("*.CSV"))
        if not data_files:
            data_files = list(dataset_path.glob("*.data"))
        if not data_files:
            data_files = list(dataset_path.glob("*.arff"))

        if not data_files:
            raise ValueError(f"No data file found under {dataset_path}")

        data_file = data_files[0]

        try:
            if data_file.suffix.lower() == ".csv":
                df = pd.read_csv(data_file)
            elif data_file.suffix.lower() == ".arff":
                from scipy.io import arff

                data, meta = arff.loadarff(data_file)
                df = pd.DataFrame(data)
            else:
                # Attempt multiple separators.
                for sep in [",", "\t", " ", ";"]:
                    try:
                        df = pd.read_csv(data_file, sep=sep)
                        if df.shape[0] > 0 and df.shape[1] > 0:
                            break
                    except Exception:
                        continue
        except Exception as e:
            raise ValueError(f"Error loading dataset {dataset_name}: {str(e)}")

        return {
            "name": dataset_name,
            "info": info,
            "df": df,
            "file_path": str(data_file),
        }

    def list_all_datasets(self):
        """List all discovered dataset names."""

        return list(self.dataset_info.keys())

    def get_dataset_info(self, dataset_name):
        """Get metadata for a dataset.

        Args:
            dataset_name: Dataset name.

        Returns:
            dict: Dataset metadata.
        """

        if dataset_name not in self.dataset_info:
            raise ValueError(f"Dataset not found: {dataset_name}")

        return self.dataset_info[dataset_name]


def example_usage():
    """Minimal example usage for manual testing."""

    loader = DatasetLoader("./data")

    categories = loader.get_dataset_categories()
    print("Dataset category stats:", categories)

    datasets = loader.get_datasets_by_category("small_datasets", "binclass", "numerical")
    print(f"Found {len(datasets)} small binclass numerical dataset(s)")

    if datasets:
        dataset = loader.load_dataset(datasets[0])
        print(f"Loaded dataset: {dataset['name']}")
        print(f"Dataset shape: {dataset['df'].shape}")
        print(f"Data file path: {dataset['file_path']}")
