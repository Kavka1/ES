from typing import Dict, List, Tuple, Type
import numpy as np
import gym


class Env_wrapper(object):
    def __init__(self, env_config: Dict) -> None:
        super().__init__()

        self.env_name = env_config['env_name']
        self.max_episode_length = env_config['max_episode_length']

        self.env = gym.make(self.env_name)
        self.env._max_episode_steps = self.max_episode_length

    def reset(self) -> np.array:
        return self.env.reset()
    
    def step(self, action: np.array) -> Tuple[np.array, float, bool, Dict]:
        return self.env.step(action)

    def render(self) -> None:
        self.env.render()