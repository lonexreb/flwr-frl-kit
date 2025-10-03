from __future__ import annotations
from typing import Any
import inspect
import torch.nn as nn
import torchrl.modules as tm


class TorchRLNetBuilder:
    """Unified class to initialize any pre-configured network from TorchRL."""

    _NETWORK_REGISTRY: dict[str, type[nn.Module]] = {}

    @classmethod
    def _build_registry(cls) -> None:
        """Build the network registry from torchrl.modules.__all__."""
        if cls._NETWORK_REGISTRY:
            return  # Already built

        for name in tm.__all__:
            obj = getattr(tm, name, None)
            if obj and inspect.isclass(obj):
                try:
                    if issubclass(obj, nn.Module):
                        cls._NETWORK_REGISTRY[name] = obj
                except TypeError:
                    pass

    def __init__(self, network_type: str, **kwargs: Any):
        """Initialize a TorchRL network.

        Args:
            network_type: Type of network to create (e.g., "MLP", "ConvNet", "ActorValueOperator").
                Use available_networks() to see all available network types.
            **kwargs: Network-specific configuration parameters

        Raises:
            ValueError: If network_type is not supported
            TypeError: If provided kwargs don't match network requirements
            RuntimeError: If network initialization fails
        """
        self._build_registry()

        if network_type not in self._NETWORK_REGISTRY:
            available = ", ".join(sorted(self._NETWORK_REGISTRY.keys()))
            raise ValueError(
                f"Unknown network type '{network_type}'. Available: {available}"
            )

        network_class = self._NETWORK_REGISTRY[network_type]

        try:
            self.network = network_class(**kwargs)
        except TypeError as e:
            # Get the function signature for better error messages
            sig = inspect.signature(network_class.__init__)
            params = [
                p.name for p in sig.parameters.values()
                if p.name not in ('self', 'args', 'kwargs')
            ]
            raise TypeError(
                f"Failed to initialize {network_type} with provided arguments. "
                f"Error: {str(e)}. "
                f"Expected parameters: {', '.join(params)}. "
                f"Provided: {', '.join(kwargs.keys())}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize {network_type}: {str(e)}"
            ) from e

    def get_network(self) -> nn.Module:
        """Return the initialized network."""
        return self.network

    @classmethod
    def available_networks(cls) -> list[str]:
        """Return list of available network types."""
        cls._build_registry()
        return sorted(cls._NETWORK_REGISTRY.keys())
