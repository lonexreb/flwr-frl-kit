from __future__ import annotations
import torch as t

@t.no_grad()
def compute_gae(rewards, dones, values, next_value, gamma=0.99, lam=0.95):
    T = len(rewards)
    adv = t.zeros(T, device=values.device)
    last_gae = 0.0
    for t_ in reversed(range(T)):
        nonterminal = 1.0 - dones[t_]
        delta = rewards[t_] + gamma * next_value * nonterminal - values[t_]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t_] = last_gae
        next_value = values[t_]
    returns = values + adv
    return adv, returns
