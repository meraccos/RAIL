import os

if __import__("sys").platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv
from gymnasium import spaces
import numpy as np


class QuarterCheetah(HalfCheetahEnv):
    def __init__(self, render_student_obs=True, **kwargs):
        super().__init__(**kwargs)
        self.render_student_obs = render_student_obs
        observation_spaces = {
            "teacher_obs": spaces.Box(
                low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
            )
        }
        if render_student_obs:
            observation_spaces["student_obs"] = spaces.Box(
                low=-1.0, high=1.0, shape=(3, 84, 84), dtype=np.float32
            )
        self.observation_space = spaces.Dict(observation_spaces)

    def render(self):
        img = super().render()
        return img

    def render_student(self):
        img = super().render()
        img = img.copy()
        img[:, :42, :] = 0
        return img

    def _get_imitation_obs(self, obs_t):
        obs = {"teacher_obs": obs_t.astype(np.float32)}
        if self.render_student_obs:
            obs_s = self.render_student().transpose(2, 0, 1) / 255.0 * 2 - 1
            obs["student_obs"] = obs_s.astype(np.float32)
        return obs

    def reset(self, **kwargs):
        obs_t, info = super().reset(**kwargs)
        obs = self._get_imitation_obs(obs_t)
        return (obs, info)

    def step(self, action):
        obs_t, reward, terminated, truncated, info = super().step(action)
        obs = self._get_imitation_obs(obs_t)
        info["is_success"] = False
        return (obs, reward, terminated, truncated, info)
