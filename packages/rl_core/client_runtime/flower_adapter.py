from __future__ import annotations
import flwr as fl

class FlowerClientAdapter(fl.client.NumPyClient):
    def __init__(self, rl_client, round_train_steps: int, eval_episodes: int):
        self.rl = rl_client
        self.round_train_steps = round_train_steps
        self.eval_episodes = eval_episodes

    # server -> client: global weights
    def set_parameters(self, parameters):
        # parameters is a list of ndarrays; we passed a single blob
        (blob,) = parameters
        self.rl.set_weights({"model": blob.tobytes()})

    # client -> server
    def get_parameters(self, config):
        w = self.rl.get_weights()["model"]
        import numpy as np
        return [np.frombuffer(w, dtype=np.uint8)], {}

    def fit(self, parameters, config):
        if parameters:
            self.set_parameters(parameters)
        metrics = self.rl.train_for(int(config.get("steps", self.round_train_steps)))
        # return new params + num_examples + metrics
        params, _ = self.get_parameters(config)
        return params, int(metrics["steps"]), metrics

    def evaluate(self, parameters, config):
        if parameters:
            self.set_parameters(parameters)
        out = self.rl.evaluate(int(config.get("episodes", self.eval_episodes)))
        # loss is optional; send 0.0 to satisfy API
        return 0.0, int(out["episodes"]), out
