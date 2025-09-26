#!/usr/bin/env python3
"""Simple test client to diagnose connection issues."""

import logging
import flwr as fl
import numpy as np

logging.basicConfig(level=logging.DEBUG)

class SimpleTestClient(fl.client.NumPyClient):
    """Minimal test client."""

    def get_parameters(self, config):
        print("get_parameters called")
        return [np.array([1.0, 2.0, 3.0])], {}

    def fit(self, parameters, config):
        print(f"fit called with {len(parameters)} parameters")
        return [np.array([1.1, 2.1, 3.1])], 3, {"loss": 0.5}

    def evaluate(self, parameters, config):
        print(f"evaluate called with {len(parameters)} parameters")
        return 0.3, 3, {"accuracy": 0.9}

if __name__ == "__main__":
    print("Starting simple test client...")
    fl.client.start_client(
        server_address="localhost:8080",
        client=SimpleTestClient(),
    )