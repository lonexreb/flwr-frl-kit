from __future__ import annotations
from typing import Any, Callable, Literal
import inspect
from torch.utils.data import DataLoader
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (
    IidPartitioner,
    DirichletPartitioner,
    PathologicalPartitioner,
    ExponentialPartitioner,
    LinearPartitioner,
    SquarePartitioner,
    NaturalIdPartitioner,
)


class FederatedDataLoader:
    """Factory class to create federated and centralized dataloaders with flexible configuration.

    Note:
        Centralized dataset methods (create_centralized_loader, load_centralized_dataset)
        are provided for testing and evaluation purposes only. Real federated learning
        scenarios should use create_federated_loader with appropriate partitioning.
    """

    _PARTITIONER_REGISTRY = {
        "iid": IidPartitioner,
        "dirichlet": DirichletPartitioner,
        "pathological": PathologicalPartitioner,
        "exponential": ExponentialPartitioner,
        "linear": LinearPartitioner,
        "square": SquarePartitioner,
        "natural_id": NaturalIdPartitioner,
    }

    _fds_cache: dict[str, FederatedDataset] = {}

    @classmethod
    def create_federated_loader(
        cls,
        dataset_name: str,
        partition_id: int,
        num_partitions: int,
        partitioner_type: str = "iid",
        partitioner_kwargs: dict[str, Any] | None = None,
        split: str = "train",
        batch_size: int = 32,
        shuffle: bool = True,
        transform: Callable | None = None,
        train_test_split: float | None = None,
        train_test_seed: int = 42,
        return_both_splits: bool = False,
        dataloader_kwargs: dict[str, Any] | None = None,
    ) -> DataLoader | tuple[DataLoader, DataLoader]:
        """Create federated dataloader(s) from a partitioned dataset.

        Args:
            dataset_name: HuggingFace dataset name (e.g., "uoft-cs/cifar10")
            partition_id: ID of the partition to load
            num_partitions: Total number of partitions
            partitioner_type: Type of partitioner ("iid", "dirichlet", "pathological", etc.)
            partitioner_kwargs: Additional kwargs for the partitioner
            split: Dataset split to use ("train", "test", etc.)
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle data
            transform: Optional transform function to apply to batches
            train_test_split: If set, split partition into train/test (e.g., 0.2 for 80/20)
            train_test_seed: Random seed for train/test split
            return_both_splits: If True and train_test_split is set, return (train_loader, test_loader)
            dataloader_kwargs: Additional kwargs for DataLoader

        Returns:
            DataLoader or tuple of (train_loader, test_loader)

        Raises:
            ValueError: If partitioner_type is not supported
            TypeError: If provided kwargs don't match requirements
            RuntimeError: If dataloader creation fails
        """
        if partitioner_kwargs is None:
            partitioner_kwargs = {}
        if dataloader_kwargs is None:
            dataloader_kwargs = {}

        # Validate partitioner type
        if partitioner_type not in cls._PARTITIONER_REGISTRY:
            available = ", ".join(cls._PARTITIONER_REGISTRY.keys())
            raise ValueError(
                f"Unknown partitioner type '{partitioner_type}'. Available: {available}"
            )

        # Create cache key
        cache_key = f"{dataset_name}_{split}_{partitioner_type}_{num_partitions}"

        try:
            # Get or create FederatedDataset
            if cache_key not in cls._fds_cache:
                partitioner_class = cls._PARTITIONER_REGISTRY[partitioner_type]

                # Create partitioner with error handling
                try:
                    partitioner = partitioner_class(
                        num_partitions=num_partitions, **partitioner_kwargs
                    )
                except TypeError as e:
                    sig = inspect.signature(partitioner_class.__init__)
                    params = [
                        p.name
                        for p in sig.parameters.values()
                        if p.name not in ("self", "num_partitions", "args", "kwargs")
                    ]
                    raise TypeError(
                        f"Failed to initialize {partitioner_type} partitioner. "
                        f"Error: {str(e)}. "
                        f"Expected parameters (besides num_partitions): {', '.join(params)}. "
                        f"Provided: {', '.join(partitioner_kwargs.keys())}"
                    ) from e

                cls._fds_cache[cache_key] = FederatedDataset(
                    dataset=dataset_name,
                    partitioners={split: partitioner},
                )

            # Load partition
            fds = cls._fds_cache[cache_key]
            partition = fds.load_partition(partition_id, split=split)

            # Apply train/test split if requested
            if train_test_split is not None:
                partition = partition.train_test_split(
                    test_size=train_test_split, seed=train_test_seed
                )

            # Apply transform if provided
            if transform is not None:
                if train_test_split is not None:
                    partition = partition.with_transform(transform)
                else:
                    partition = partition.with_transform(transform)

            # Create DataLoader(s)
            if train_test_split is not None and return_both_splits:
                train_data = partition["train"]
                test_data = partition["test"]

                train_loader = DataLoader(
                    train_data,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    **dataloader_kwargs,
                )
                test_loader = DataLoader(
                    test_data,
                    batch_size=batch_size,
                    shuffle=False,
                    **dataloader_kwargs,
                )
                return train_loader, test_loader
            elif train_test_split is not None:
                # Return only train split by default
                data = partition["train"]
            else:
                data = partition

            return DataLoader(
                data,
                batch_size=batch_size,
                shuffle=shuffle,
                **dataloader_kwargs,
            )

        except Exception as e:
            if isinstance(e, (ValueError, TypeError)):
                raise
            raise RuntimeError(
                f"Failed to create federated dataloader for {dataset_name}: {str(e)}"
            ) from e

    @classmethod
    def load_centralized_dataset(
        cls,
        dataset_name: str,
        split: str = "test",
        transform: Callable | None = None,
        format: str = "torch",
    ):
        """Load a complete centralized dataset (for testing purposes only).

        Warning:
            This method loads the entire dataset without partitioning and should
            ONLY be used for testing and evaluation. Real federated learning should
            use create_federated_loader with appropriate partitioning.

        Args:
            dataset_name: HuggingFace dataset name (e.g., "uoft-cs/cifar10")
            split: Dataset split to use ("train", "test", etc.)
            transform: Optional transform function to apply to batches
            format: Dataset format ("torch", "numpy", etc.)

        Returns:
            Complete dataset (not partitioned)

        Raises:
            RuntimeError: If dataset loading fails
        """
        try:
            dataset = load_dataset(dataset_name, split=split)

            if transform is not None:
                dataset = dataset.with_format(format).with_transform(transform)
            else:
                dataset = dataset.with_format(format)

            return dataset

        except Exception as e:
            raise RuntimeError(
                f"Failed to load centralized dataset {dataset_name}: {str(e)}"
            ) from e

    @classmethod
    def create_centralized_loader(
        cls,
        dataset_name: str,
        split: str = "test",
        batch_size: int = 128,
        shuffle: bool = False,
        transform: Callable | None = None,
        dataloader_kwargs: dict[str, Any] | None = None,
    ) -> DataLoader:
        """Create centralized dataloader from a complete dataset split (for testing purposes only).

        Warning:
            This method loads the entire dataset without partitioning and should
            ONLY be used for testing and evaluation. Real federated learning should
            use create_federated_loader with appropriate partitioning.

        Args:
            dataset_name: HuggingFace dataset name (e.g., "uoft-cs/cifar10")
            split: Dataset split to use ("train", "test", etc.)
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle data
            transform: Optional transform function to apply to batches
            dataloader_kwargs: Additional kwargs for DataLoader

        Returns:
            DataLoader for the centralized dataset

        Raises:
            RuntimeError: If dataloader creation fails
        """
        if dataloader_kwargs is None:
            dataloader_kwargs = {}

        try:
            dataset = load_dataset(dataset_name, split=split)

            if transform is not None:
                dataset = dataset.with_format("torch").with_transform(transform)
            else:
                dataset = dataset.with_format("torch")

            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                **dataloader_kwargs,
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to create centralized dataloader for {dataset_name}: {str(e)}"
            ) from e

    @classmethod
    def available_partitioners(cls) -> list[str]:
        """Return list of available partitioner types."""
        return list(cls._PARTITIONER_REGISTRY.keys())

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the FederatedDataset cache."""
        cls._fds_cache.clear()
