"""Evaluate each role on the same individually seeded episodes."""

import numpy as np
import torch
from stable_baselines3.common.utils import obs_as_tensor, set_random_seed


@torch.no_grad()
def evaluate(model, env, episodes=100, seed=10000):
    extractor = model.policy.features_extractor
    was_training = model.policy.training
    model.policy.set_training_mode(False)
    results = {}
    try:
        for role in ("teacher", "student"):
            returns = []
            # One active lane gives exactly the requested set of episode seeds,
            # independent of episode length and the number of training workers.
            for episode in range(episodes):
                set_random_seed(seed + episode)
                env.seed(seed + episode)
                obs = env.reset()
                total = 0.0
                while True:
                    tensor = obs_as_tensor(obs, model.device)
                    features = getattr(extractor, f"forward_{role}")(tensor)
                    features = extractor._add_noise(features)
                    actions = (
                        model.policy.get_actions_from_features(True, features)
                        .cpu()
                        .numpy()
                    )
                    if hasattr(env.action_space, "low"):
                        actions = np.clip(
                            actions, env.action_space.low, env.action_space.high
                        )
                    obs, reward, done, _ = env.step(actions)
                    total += float(reward[0])
                    if done[0]:
                        returns.append(total)
                        break
            results[role] = {
                "return": float(np.mean(returns)),
                "return_std": float(np.std(returns)),
            }
    finally:
        model.policy.set_training_mode(was_training)
    return results
