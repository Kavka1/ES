from typing import *
import numpy as np
from numpy.lib.function_base import gradient
from numpy.lib.npyio import save
import ray
from worker import Worker

from ES.common.optimizer import Adam
from ES.common.env import Env_wrapper
from ES.common.obs_filter import MeanStdFilter
from ES.common.noise import create_shared_noise, ShareNoiseTable
from ES.common.utils import check_path, batch_weighted_sum
from ES.common.model import Policy


class Master(object):
    def __init__(self, config: Dict) -> None:
        super().__init__()

        self.num_workers = config['num_workers']
        self.num_rollouts = config['num_rollouts']
        self.num_evaluation = config['num_evaluation']

        self.noise_table_size = config['noise_table_size']
        self.noise_table_seed = config['noise_table_seed']
        self.delta_sample_seed = config['noise_sample_seed']
        self.delta_std = config['delta_std']

        self.estimator_type = config['estimator_type']
        self.fitness_type = config['fitness_type']

        self.policy = Policy(config['model_config'])
        self.env = Env_wrapper(config['env_config'])

        self.init_gradient_estimator()

        self.init_noise_table()
        self.init_obsfilter(self.env.observation_space.shape)
        self.init_workers(config['model_config'], config['env_config'])
        self.init_optimizer(config['optimizer_config'])

        self.best_score = 0
    
    def init_gradient_estimator(self) -> None:
        if self.estimator_type == 'vanilla':
            self.gradient_estimation = self.gradient_estimation_vanilla
        elif self.estimator_type == 'antithetic':
            self.gradient_estimation = self.gradient_estimation_anti
        elif self.estimator_type == 'finite_difference':
            self.gradient_estimation = self.gradient_estimation_FD
        else:
            raise ValueError(f"The gradient estimator type {self.estimator_type} illegal.")

    def init_noise_table(self) -> None:
        self.noise_table = create_shared_noise.remote(self.noise_table_seed, self.noise_table_size)
        self.deltas = ShareNoiseTable(ray.get(self.noise_table), self.delta_sample_seed)

    def init_workers(self, model_config: Dict, env_config: Dict) -> None:
        self.workers = [
            Worker.remote(
                env_config = env_config,
                model_config = model_config,
                noise_table = self.noise_table,
                delta_sample_seed = self.delta_sample_seed,
                delta_std = self.delta_std,
                num_rollouts = self.num_rollouts,
                num_evaluation = self.num_evaluation,
                estimation_type = self.estimator_type
            ) for _ in range(self.num_workers)
        ]

    def init_optimizer(self, optimizer_config: Dict) -> None:
        self.optimizer = Adam(optimizer_config)
        self.batch_size = optimizer_config['batch_size']

    def init_obsfilter(self, shape: Tuple) -> None:
        self.obs_filter = MeanStdFilter(shape)

    def fitness_transform(self, rewards: np.array) -> np.array:
        if self.fitness_type == 'value':
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
            return rewards
        elif self.fitness_type == 'rank':
            rank_index = np.argsort(-rewards)
            ranks = np.zeros(shape=len(rewards))
            for rank, index in enumerate(rank_index):
                ranks[index] = rank + 1
            rewards = np.array([max(0, np.log(len(ranks)/2 + 1) - np.log(k)) for k in ranks])
            rewards = rewards / np.sum(rewards) - 1 / len(rewards)
            return rewards
        else:
            raise TypeError(f'The fitness type {self.fitness_type} illegal.')

    def gradient_estimation_vanilla(self, rewards: np.array, delta_sp: np.array) -> np.array:
        rewards = self.fitness_transform(rewards)
        deltas = [self.deltas.get_noise(sp, self.policy.total_size) for sp in delta_sp]

        gradient, count = batch_weighted_sum(rewards=rewards, deltas=deltas, batch_size=self.batch_size)
        gradient /= (len(delta_sp) * self.delta_std)
        gradient /= (np.linalg.norm(gradient) / (self.policy.total_size + 1e-8))

        return gradient

    def gradient_estimation_anti(self, rewards: np.array, delta_sp: np.array) -> np.array:
        rewards = rewards[:, 0] - rewards[:, 1]     # Authentic Gradient Estimation: Pos reward - Neg reward
        rewards = self.fitness_transform(rewards)
        deltas = [self.deltas.get_noise(sp, self.policy.total_size) for sp in delta_sp]

        gradient, count = batch_weighted_sum(rewards=rewards, deltas=deltas, batch_size=self.batch_size)
        gradient /= (len(delta_sp) * self.delta_std)
        gradient /= (np.linalg.norm(gradient) / (self.policy.total_size + 1e-8))

        return gradient

    def gradient_estimation_FD(self, rewards: np.array, delta_sp: np.array) -> np.array:
        rewards = rewards[:, 0] - rewards[:, 1]     # Authentic Gradient Estimation: Pos reward - baseline reward
        rewards = self.fitness_transform(rewards)
        deltas = [self.deltas.get_noise(sp, self.policy.total_size) for sp in delta_sp]

        gradient, count = batch_weighted_sum(rewards=rewards, deltas=deltas, batch_size=self.batch_size)
        gradient /= (len(delta_sp) * self.delta_std)
        gradient /= (np.linalg.norm(gradient) / (self.policy.total_size + 1e-8))

        return gradient

    def apply_gradient(self, gradient: np.array) -> None:
        update = self.optimizer.update(gradient)
        self.policy.increment_update(update)

    def synchronous_filters(self, mean: List[np.array], square_sum: List[np.array], count: int) -> None:
        mean = sum(mean) / self.num_workers
        square_sum = sum(square_sum) / self.num_workers        
        self.obs_filter.update(mean, square_sum, count)

    def evaluation(self) -> float:
        total_r = 0
        for i_evaluation in range(self.num_evaluation):
            done = False
            obs = self.env.reset()
            while not done:
                a = self.policy.forward(self.obs_filter(obs))
                obs, r, done, _ = self.env.step(a)
                total_r += r
        return total_r / self.num_evaluation

    def save_policy(self, save_path: str, remark: str) -> None:
        check_path(save_path)
        np.save(save_path + remark, self.policy.get_params())
        print(f"------Policy parameters saved to {save_path}------")

    def save_filter(self, save_path: str, remark: str) -> None:
        check_path(save_path)
        filter_params = np.array([self.obs_filter.mean, self.obs_filter.square_sum, self.obs_filter.count])
        np.save(save_path + remark, filter_params)
        print(f'------Obs filter parameters saved to {save_path}------')