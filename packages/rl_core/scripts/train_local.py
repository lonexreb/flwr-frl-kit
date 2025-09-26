from __future__ import annotations
import time
from rich.console import Console
from rich.table import Table
from rl_core.client_runtime.a2c_client import A2CClient, A2CConfig

def main():
    console = Console()
    cfg = A2CConfig(env_id="CartPole-v1", seed=17, rollout_len=128)
    client = A2CClient(cfg)

    total = 20000
    tick = 2048
    done = 0
    t0 = time.time()
    table = Table("Total Steps", "AvgRet(eval@5)", "Entropy", "KL", "sec/2k")
    while done < total:
        logs = client.train_for(tick)
        eval_out = client.evaluate(episodes=5)
        dt = time.time() - t0
        t0 = time.time()
        table.add_row(
            str(done + logs["steps"]),
            f"{eval_out['avg_return']:.1f}",
            f"{logs['entropy']:.3f}",
            f"{logs['kl']:.4f}",
            f"{dt:.2f}"
        )
        done += logs["steps"]
    console.print(table)

if __name__ == "__main__":
    main()
