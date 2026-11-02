from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DictFrameStack(gym.Wrapper):
    def __init__(self, env, obs_key: str, n_stack: int, grayscale: bool = False):
        super().__init__(env)
        self.obs_key = obs_key
        self.n_stack = n_stack
        self.grayscale = grayscale
        assert isinstance(env.observation_space, spaces.Dict)
        assert obs_key in env.observation_space.spaces
        base_space = env.observation_space.spaces[obs_key]
        low = base_space.low
        high = base_space.high
        if len(base_space.shape) == 3:
            (c, h, w) = base_space.shape
            out_c = 1 if grayscale else c
            if grayscale:
                low = low[:1]
                high = high[:1]
            stacked_shape = (out_c * n_stack, h, w)
        elif len(base_space.shape) == 1:
            if grayscale:
                raise ValueError(
                    "grayscale=True is only supported for image observations."
                )
            stacked_shape = (base_space.shape[0] * n_stack,)
        else:
            raise ValueError(
                f"DictFrameStack only supports 1D vectors and CHW images, got {base_space.shape}."
            )
        stacked_space = spaces.Box(
            low=np.repeat(low, n_stack, axis=0),
            high=np.repeat(high, n_stack, axis=0),
            shape=stacked_shape,
            dtype=base_space.dtype,
        )
        self.observation_space = spaces.Dict(
            {**env.observation_space.spaces, obs_key: stacked_space}
        )
        self._frames = deque(maxlen=n_stack)

    def _to_gray(self, x):
        return x.mean(axis=0, keepdims=True)

    def _process(self, frame):
        return self._to_gray(frame) if self.grayscale else frame

    def _stack(self):
        return np.concatenate(list(self._frames), axis=0)

    def reset(self, **kwargs):
        (obs, info) = self.env.reset(**kwargs)
        frame = self._process(obs[self.obs_key])
        self._frames.clear()
        for _ in range(self.n_stack):
            self._frames.append(frame)
        obs[self.obs_key] = self._stack()
        return (obs, info)

    def step(self, action):
        (obs, reward, terminated, truncated, info) = self.env.step(action)
        self._frames.append(self._process(obs[self.obs_key]))
        obs[self.obs_key] = self._stack()
        return (obs, reward, terminated, truncated, info)
