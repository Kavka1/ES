from typing import Type, Dict, List, Tuple
import numpy as np
from Repo.ES_base.utils import ShareNoiseTable
from model import Policy
import ray
import gym


class Worker(object):
    def __init__(
        self,
        env_name: str,
        model_params: Dict,
        noise_id: np.float64,
        seed_noise_table: int,
        delta_std: float,
        num_rollouts: int,
        num_evaluation: int
    ) -> None:
        super().__init__()

        self.env = gym.make(env_name)
        self.policy = Policy(model_params)
        self.NoiseTable = ShareNoiseTable(ray.get(noise_id), seed_noise_table)

        self.delta_std = delta_std

        self.num_rollouts = num_rollouts
        self.num_evaluation = num_evaluation

    def update(self, origin_policy: np.array) -> None:
        delta_sp, rollout_rewards, rollout_steps = [], [], 0

        for i_rollout in range(self.num_rollouts):
            sp, delta = self.NoiseTable.sample_delta(origin_policy.size)
            delta = (delta * self.delta_std).reshape(origin_policy.shape)

            self.policy.update_params(origin_policy + delta)
            reward_pos, timesteps_pos = self.rollouts()

            self.policy.update_params(origin_policy - delta)
            reward_neg, timesteps_neg = self.rollouts()

            delta_sp.append(sp)
            rollout_rewards.append([reward_pos, reward_neg])
            rollout_steps += timesteps_pos + timesteps_neg

        return {'delta_sp': delta_sp, 'rewards': rollout_rewards, 'rollout_steps': rollout_steps}

    def rollouts(self) -> Tuple[float, int]:
        total_reward = 0
        timesteps = 0
        for i in range(self.num_evaluation):
            done = False
            obs = self.env.reset()
            while not done:
                action = self.policy.forward(obs)   # Todo: check the action bound of env
                obs, r, done, _ = self.env.step(action)
                total_reward += r
                timesteps += 1
        return total_reward, timesteps