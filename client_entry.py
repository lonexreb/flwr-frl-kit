import os
import flwr as fl
from packages.rl_core.client_runtime.a2c_client import A2CClient, A2CConfig
from packages.rl_core.client_runtime.flower_adapter import FlowerClientAdapter

# ---- env knobs (override per container) ----
SERVER = os.getenv("FLOWER_SERVER", "127.0.0.1:8080")
ENV_ID = os.getenv("ENV_ID", "CartPole-v1")
SEED = int(os.getenv("SEED", "17"))
ROUND_STEPS = int(os.getenv("ROUND_STEPS", "2048"))
EVAL_EPISODES = int(os.getenv("EVAL_EPISODES", "5"))

# ---- build RL client ----
rl = A2CClient(A2CConfig(env_id=ENV_ID, seed=SEED, rollout_len=128))
client = FlowerClientAdapter(
    rl_client=rl,
    round_train_steps=ROUND_STEPS,
    eval_episodes=EVAL_EPISODES,
)

# ---- start Flower NumPy client ----
fl.client.start_numpy_client(server_address=SERVER, client=client)
