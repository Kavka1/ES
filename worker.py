from typing import Type, Dict, List, Tuple
import numpy as np
from env import Env_wrapper
from noise import ShareNoiseTable
from model import Policy
import ray

from obs_filter import MeanStdFilter


@ray.remote
class Worker(object):
    def __init__(
        self,
        env_config: Dict,
        model_config: Dict,
        noise_table: np.float64,
        delta_sample_seed: int,
        delta_std: float,
        num_rollouts: int,
        num_evaluation: int
    ) -> None:
        super().__init__()

        self.env = Env_wrapper(env_config)
        self.policy = Policy(model_config)
        self.NoiseTable = ShareNoiseTable(noise_table, delta_sample_seed)

        self.obs_filter = MeanStdFilter(shape = self.env.observation_space.shape)

        self.delta_std = delta_std
        self.num_rollouts = num_rollouts
        self.num_evaluation = num_evaluation

    def do_rollouts(self, origin_policy: np.array) -> None:
        delta_sp, rollout_rewards, obs_buffer, rollout_steps = [], [], [], 0

        for i_rollout in range(self.num_rollouts):
            sp, delta = self.NoiseTable.sample_delta(origin_policy.size)
            delta = (delta * self.delta_std).reshape(origin_policy.shape)

            self.policy.update_params(origin_policy + delta)
            reward_pos, timesteps_pos, obs_local_pos = self.rollouts()

            self.policy.update_params(origin_policy - delta)
            reward_neg, timesteps_neg, obs_local_neg = self.rollouts()

            delta_sp.append(sp)
            rollout_rewards.append([reward_pos, reward_neg])
            obs_buffer += [obs_local_pos, obs_local_neg]
            rollout_steps += timesteps_pos + timesteps_neg

        self.obs_filter.push_batch(np.concatenate(obs_buffer, axis=0))
        filter_stats = [self.obs_filter.mean, self.obs_filter.square_sum, self.obs_filter.count]

        return {'delta_sp': delta_sp, 'rewards': rollout_rewards, 'rollout_steps': rollout_steps, 'filter_stats': filter_stats}

    def rollouts(self) -> Tuple[float, int]:
        total_reward = 0
        timesteps = 0
        obs_buffer = []
        for i in range(self.num_evaluation):
            done = False
            obs = self.env.reset()
            while not done:
                action = self.policy.forward(self.obs_filter(obs))
                obs, r, done, _ = self.env.step(action)
                total_reward += r
                timesteps += 1
                obs_buffer.append(obs)
        return total_reward, timesteps, np.stack(obs_buffer, axis=0)