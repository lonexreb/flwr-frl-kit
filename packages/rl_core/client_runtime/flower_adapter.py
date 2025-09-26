import numpy as np
import flwr as fl

class FlowerClientAdapter(fl.client.NumPyClient):
    """
    Thin bridge from your RLClient to Flower.
    - Parameters are shipped as a single uint8 ndarray (serialized PyTorch state_dict bytes).
    - Metrics are passed through unchanged.
    """

    def __init__(self, rl_client, round_train_steps: int = 2048, eval_episodes: int = 5, dummy_weight_update: bool = False):
        self.rl = rl_client
        self.round_train_steps = round_train_steps
        self.eval_episodes = eval_episodes
        self.dummy_weight_update = dummy_weight_update

    # ----- Parameter exchange: one blob packed into uint8 ndarray -----
    def get_parameters(self, config):
        if self.dummy_weight_update:
            # Return dummy weights for testing
            dummy_blob = b"dummy_weights"
            return [np.frombuffer(dummy_blob, dtype=np.uint8)], {}
        blob = self.rl.get_weights()["model"]  # bytes from RLClient
        return [np.frombuffer(blob, dtype=np.uint8)], {}

    def set_parameters(self, parameters):
        if self.dummy_weight_update:
            # Skip setting parameters in dummy mode
            return
        if parameters:
            (arr,) = parameters
            self.rl.set_weights({"model": arr.tobytes()})

    # ----- Flower hooks -----
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        steps = int(config.get("steps", self.round_train_steps))
        if self.dummy_weight_update:
            # Return dummy metrics for testing
            metrics = {
                "steps": steps,
                "entropy": 0.5,
                "kl": 0.01,
                "loss": 1.0,
                "policy_loss": 0.5,
                "value_loss": 0.5
            }
        else:
            metrics = self.rl.train_for(steps)  # must include: steps, entropy, kl, loss, policy_loss, value_loss
        newparams, _ = self.get_parameters(config)
        # (parameters, num_examples, metrics)
        return newparams, int(metrics["steps"]), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        episodes = int(config.get("episodes", self.eval_episodes))
        if self.dummy_weight_update:
            # Return dummy evaluation metrics for testing
            out = {
                "avg_return": 100.0,
                "std_return": 10.0,
                "episodes": episodes
            }
        else:
            out = self.rl.evaluate(episodes)  # must return: avg_return, std_return, episodes
        # (loss, num_examples, metrics) — loss unused
        return 0.0, int(out["episodes"]), out
