"""Evaluate saved teacher and student policies on the same episode seeds."""

import argparse
import json
import os
from pathlib import Path
import sys

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from rail.envs.factory import make_vector_env
from rail.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    cfg = yaml.safe_load((args.run / "config.yaml").read_text())
    env = make_vector_env(cfg, 1, args.seed)
    try:
        if cfg["normalize_env"]:
            env = VecNormalize.load(args.run / "vecnormalize.pkl", env)
            env.training = False
            env.norm_reward = False
        model = PPO.load(args.run / "model.zip", env=env, device=args.device)
        result = evaluate(model, env, args.episodes, args.seed)
        metric = "return"
        result["imitation_gap"] = result["teacher"][metric] - result["student"][metric]
        result.update(episodes=args.episodes, seed=args.seed, metric=metric)
        path = args.run / f"evaluation-{args.seed}-{args.episodes}.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
