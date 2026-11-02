"""Train RAIL. Run with --help for overrides and a short smoke test."""

import argparse
import json
import os
from pathlib import Path
import platform
import sys

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from rail.callback import RAILCallback
from rail.envs.factory import training_envs
from rail.models import CombinedExtractors
from rail.policy import CustomPPOPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs/quarter_cheetah.yaml",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument(
        "--output", type=Path, default=Path("runs/quarter-cheetah-seed0")
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=60,
        help="target episodes per role during training (rounded down per worker)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="two tiny rollouts; does not reproduce paper results",
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.eval_episodes < 1:
        parser.error("--eval-episodes must be positive")
    if args.smoke:
        cfg.update(
            n_envs=1, n_eval_envs=1, max_steps=32, max_episode_steps=16, eval_interval=2
        )
        cfg["ppo"].update(n_steps=16, batch_size=16, n_epochs=1)
        cfg["methods"]["rail"].update(batch_size=16, n_align_epochs=1)
        args.eval_episodes = 1
    # Refuse to mix runs or overwrite existing checkpoints.
    args.output.mkdir(parents=True, exist_ok=False)
    cfg["seed"] = args.seed
    (args.output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    import importlib.metadata as metadata

    versions = {
        dist.metadata["Name"]: dist.version for dist in metadata.distributions()
    }
    (args.output / "versions.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "packages": versions,
                "requested_device": args.device,
                "smoke": args.smoke,
            },
            indent=2,
        )
    )
    # Small networks are much faster with a single CPU thread during smoke tests.
    if args.smoke:
        torch.set_num_threads(1)
    env, eval_env = training_envs(cfg, args.seed)
    try:
        model = PPO(
            CustomPPOPolicy,
            env,
            seed=args.seed,
            device=args.device,
            verbose=1,
            policy_kwargs=dict(
                features_extractor_class=CombinedExtractors,
                features_extractor_kwargs=dict(
                    emb_d=cfg["methods"]["rail"]["embd"],
                    normalize=True,
                    detach_forward=True,
                    augment=cfg["methods"]["rail"]["augment"],
                ),
            ),
            **cfg["ppo"],
        )
        callback = RAILCallback(eval_env, cfg, args.output, args.eval_episodes)
        model.learn(total_timesteps=cfg["max_steps"], callback=callback)
        model.save(args.output / "model")
        if isinstance(env, VecNormalize):
            env.save(args.output / "vecnormalize.pkl")
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
