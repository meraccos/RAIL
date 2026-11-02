"""QuarterCheetah environment construction and normalization."""

from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from rail.envs.quarter_cheetah import QuarterCheetah
from rail.wrappers import DictFrameStack


def make_env(config):
    env = TimeLimit(QuarterCheetah(**config["env_vars"]), config["max_episode_steps"])
    if config["n_stack_student"] > 1:
        env = DictFrameStack(env, "student_obs", config["n_stack_student"])
    return env


def make_vector_env(config, n_envs, seed):
    vec_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    return make_vec_env(
        lambda: make_env(config), n_envs=n_envs, seed=seed, vec_env_cls=vec_cls
    )


def training_envs(config, seed):
    train = make_vector_env(config, config["n_envs"], seed)
    evaluate = make_vector_env(config, config["n_eval_envs"], seed + 10000)
    if config["normalize_env"]:
        train = VecNormalize(train, norm_obs=True, norm_reward=True)
        evaluate = VecNormalize(
            evaluate, norm_obs=True, norm_reward=False, training=False
        )
    return train, evaluate
