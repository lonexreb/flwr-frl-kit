from __future__ import annotations
from typing import Any, Literal
import inspect
import numpy as np
from torchrl.envs import (
    GymEnv,
    DMControlEnv,
    BraxEnv,
    JumanjiEnv,
    ParallelEnv,
    EnvBase,
    TransformedEnv,
)
from torchrl.envs.transforms import Compose as EnvCompose


class FederatedRLEnvLoader:
    """Factory class to create TorchRL environments with federated partitioning logic.

    In federated RL, different clients can work with:
    - Different tasks/environments
    - Different seeds/initial conditions
    - Different environment configurations
    - Different difficulty levels

    This class provides partitioning logic to distribute environment configurations
    across federated clients while maintaining reproducibility.
    """

    _ENV_REGISTRY = {
        "gym": GymEnv,
        "dm_control": DMControlEnv,
        "brax": BraxEnv,
        "jumanji": JumanjiEnv,
    }

    @classmethod
    def create_partitioned_env(
        cls,
        env_type: str,
        env_name: str,
        partition_id: int,
        num_partitions: int,
        partition_strategy: Literal["seed", "task_list", "config_list"] = "seed",
        task_list: list[str] | None = None,
        config_list: list[dict[str, Any]] | None = None,
        base_seed: int = 42,
        num_parallel: int = 1,
        transforms: list | None = None,
        env_kwargs: dict[str, Any] | None = None,
    ) -> EnvBase:
        """Create a partitioned RL environment for federated learning.

        Args:
            env_type: Type of environment ("gym", "dm_control", "brax", "jumanji")
            env_name: Name of the environment (e.g., "CartPole-v1", "walker-walk")
            partition_id: ID of the partition (client ID)
            num_partitions: Total number of partitions (clients)
            partition_strategy: How to partition environments:
                - "seed": Each client gets same env with different seed
                - "task_list": Each client gets different task from task_list
                - "config_list": Each client gets different config from config_list
            task_list: List of task names for "task_list" strategy
            config_list: List of config dicts for "config_list" strategy
            base_seed: Base random seed for reproducibility
            num_parallel: Number of parallel environments per client
            transforms: List of TorchRL transforms to apply
            env_kwargs: Additional kwargs for environment creation

        Returns:
            Configured TorchRL environment (optionally parallel and transformed)

        Raises:
            ValueError: If env_type or partition_strategy is invalid
            TypeError: If provided kwargs don't match requirements
            RuntimeError: If environment creation fails

        Examples:
            >>> # Different seeds per client
            >>> env = FederatedRLEnvLoader.create_partitioned_env(
            ...     env_type="gym",
            ...     env_name="CartPole-v1",
            ...     partition_id=0,
            ...     num_partitions=10,
            ...     partition_strategy="seed",
            ...     base_seed=42
            ... )

            >>> # Different tasks per client
            >>> env = FederatedRLEnvLoader.create_partitioned_env(
            ...     env_type="gym",
            ...     env_name="",  # Not used with task_list
            ...     partition_id=0,
            ...     num_partitions=3,
            ...     partition_strategy="task_list",
            ...     task_list=["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
            ... )
        """
        if env_kwargs is None:
            env_kwargs = {}

        # Validate env_kwargs for gymnasium environments
        if env_type == "gym":
            invalid_kwargs = {"seed", "frameskip"}
            found_invalid = invalid_kwargs.intersection(env_kwargs.keys())
            if found_invalid:
                raise ValueError(
                    f"Invalid kwargs for Gymnasium environment: {found_invalid}. "
                    f"'seed' should be passed to env.reset(seed=...), not __init__(). "
                    f"'frameskip' is not a valid CartPole kwarg."
                )

        # Validate env type
        if env_type not in cls._ENV_REGISTRY:
            available = ", ".join(cls._ENV_REGISTRY.keys())
            raise ValueError(
                f"Unknown environment type '{env_type}'. Available: {available}"
            )

        env_class = cls._ENV_REGISTRY[env_type]

        try:
            # Determine environment configuration based on partition strategy
            if partition_strategy == "seed":
                # Each client gets same env with different seed
                partition_seed = base_seed + partition_id
                final_env_name = env_name
                final_kwargs = {**env_kwargs}

            elif partition_strategy == "task_list":
                # Round-robin assignment of tasks to clients
                if task_list is None or len(task_list) == 0:
                    raise ValueError("task_list must be provided for 'task_list' strategy")

                task_idx = partition_id % len(task_list)
                final_env_name = task_list[task_idx]
                partition_seed = base_seed + partition_id
                final_kwargs = {**env_kwargs}

            elif partition_strategy == "config_list":
                # Round-robin assignment of configs to clients
                if config_list is None or len(config_list) == 0:
                    raise ValueError("config_list must be provided for 'config_list' strategy")

                config_idx = partition_id % len(config_list)
                partition_config = config_list[config_idx]
                final_env_name = env_name
                partition_seed = base_seed + partition_id
                final_kwargs = {**env_kwargs, **partition_config}

            else:
                raise ValueError(
                    f"Unknown partition_strategy '{partition_strategy}'. "
                    f"Available: 'seed', 'task_list', 'config_list'"
                )

            # Create base environment
            try:
                # Handle different env types that may have different init signatures
                if env_type == "gym":
                    base_env = env_class(env_name=final_env_name, **final_kwargs)
                elif env_type == "dm_control":
                    # DMControl uses domain_name and task_name
                    if "domain_name" not in final_kwargs:
                        parts = final_env_name.split("-")
                        if len(parts) >= 2:
                            final_kwargs["domain_name"] = parts[0]
                            final_kwargs["task_name"] = parts[1]
                        else:
                            final_kwargs["domain_name"] = final_env_name
                            final_kwargs["task_name"] = final_kwargs.get("task_name", "stand")
                    base_env = env_class(**final_kwargs)
                else:
                    base_env = env_class(env_name=final_env_name, **final_kwargs)

            except TypeError as e:
                sig = inspect.signature(env_class.__init__)
                params = [
                    p.name for p in sig.parameters.values()
                    if p.name not in ("self", "args", "kwargs")
                ]
                raise TypeError(
                    f"Failed to initialize {env_type} environment. "
                    f"Error: {str(e)}. "
                    f"Expected parameters: {', '.join(params)}. "
                    f"Provided: {', '.join(final_kwargs.keys())}"
                ) from e

            # Create parallel environments if requested
            if num_parallel > 1:
                env = ParallelEnv(
                    num_workers=num_parallel,
                    create_env_fn=lambda: env_class(
                        env_name=final_env_name if env_type != "dm_control" else None,
                        **final_kwargs
                    ) if env_type != "dm_control" else env_class(**final_kwargs),
                )
            else:
                env = base_env

            # Apply transforms if provided
            if transforms is not None and len(transforms) > 0:
                transform_compose = EnvCompose(*transforms)
                env = TransformedEnv(env, transform_compose)

            return env

        except Exception as e:
            if isinstance(e, (ValueError, TypeError)):
                raise
            raise RuntimeError(
                f"Failed to create partitioned RL environment: {str(e)}"
            ) from e

    @classmethod
    def create_env_batch(
        cls,
        env_type: str,
        env_name: str,
        num_partitions: int,
        partition_strategy: Literal["seed", "task_list", "config_list"] = "seed",
        task_list: list[str] | None = None,
        config_list: list[dict[str, Any]] | None = None,
        base_seed: int = 42,
        transforms: list | None = None,
        env_kwargs: dict[str, Any] | None = None,
    ) -> list[EnvBase]:
        """Create a batch of partitioned environments for all clients (for testing purposes).

        Warning:
            This method creates environments for ALL partitions and should primarily
            be used for centralized testing. In real federated scenarios, each client
            calls create_partitioned_env with their own partition_id.

        Args:
            env_type: Type of environment
            env_name: Name of the environment
            num_partitions: Total number of partitions to create
            partition_strategy: How to partition environments
            task_list: List of task names for "task_list" strategy
            config_list: List of config dicts for "config_list" strategy
            base_seed: Base random seed
            transforms: List of TorchRL transforms
            env_kwargs: Additional environment kwargs

        Returns:
            List of configured environments, one per partition

        Raises:
            ValueError, TypeError, RuntimeError: Same as create_partitioned_env
        """
        envs = []
        for partition_id in range(num_partitions):
            env = cls.create_partitioned_env(
                env_type=env_type,
                env_name=env_name,
                partition_id=partition_id,
                num_partitions=num_partitions,
                partition_strategy=partition_strategy,
                task_list=task_list,
                config_list=config_list,
                base_seed=base_seed,
                num_parallel=1,
                transforms=transforms,
                env_kwargs=env_kwargs,
            )
            envs.append(env)
        return envs

    @classmethod
    def available_env_types(cls) -> list[str]:
        """Return list of available environment types."""
        return list(cls._ENV_REGISTRY.keys())
