import numpy as np
import flwr as fl

class FlowerClientAdapter(fl.client.NumPyClient):
    """
    Thin bridge from your RLClient to Flower.
    - Parameters are shipped as a single uint8 ndarray (serialized PyTorch state_dict bytes).
    - Metrics are passed through unchanged.
    """

    def __init__(self, rl_client, round_train_steps: int = 2048, eval_episodes: int = 5):
        self.rl = rl_client
        self.round_train_steps = round_train_steps
        self.eval_episodes = eval_episodes

    # ----- Parameter exchange: one blob packed into uint8 ndarray -----
    def get_parameters(self, config):
        blob = self.rl.get_weights()["model"]  # bytes from RLClient
        return [np.frombuffer(blob, dtype=np.uint8)], {}

    def set_parameters(self, parameters):
        if parameters:
            (arr,) = parameters
            self.rl.set_weights({"model": arr.tobytes()})

    # ----- Flower hooks -----
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        steps = int(config.get("steps", self.round_train_steps))
        metrics = self.rl.train_for(steps)  # must include: steps, entropy, kl, loss, policy_loss, value_loss
        newparams, _ = self.get_parameters(config)
        # (parameters, num_examples, metrics)
        return newparams, int(metrics["steps"]), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        episodes = int(config.get("episodes", self.eval_episodes))
        out = self.rl.evaluate(episodes)  # must return: avg_return, std_return, episodes
        # (loss, num_examples, metrics) — loss unused
        return 0.0, int(out["episodes"]), out
