from typing import *
from gym.core import Env
import numpy as np
from numpy.lib.npyio import save
import ray
from ray.worker import remote
from model import Policy
from utils import ShareNoiseTable, check_path, create_shared_noise
from worker import Worker
from optimizer import Adam
from utils import batch_weighted_sum
from env import Env_wrapper


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

        self.policy = Policy(config['model_config'])
        self.env = Env_wrapper(config['env_config'])
        self.init_noise_table()
        self.init_workers(config['model_config'], config['env_config'])
        self.init_optimizer(config['optimizer_config'])

        self.best_score = 0
    
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
            ) for _ in range(self.num_workers)
        ]

    def init_optimizer(self, optimizer_config: Dict) -> None:
        self.optimizer = Adam(optimizer_config)
        self.batch_size = optimizer_config['batch_size']

    def gradient_estimation(self, rewards: np.array, delta_sp: np.array) -> np.array:
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
        rewards = rewards[:, 0] - rewards[:, 1]     # Authentic Gradient Estimation: Pos reward - Neg reward
        deltas = [self.deltas.get_noise(sp, self.policy.total_size) for sp in delta_sp]

        gradient, count = batch_weighted_sum(rewards=rewards, deltas=deltas, batch_size=self.batch_size)
        gradient /= len(delta_sp)
        gradient /= (np.linalg.norm(gradient) / (self.policy.total_size + 1e-8))

        return gradient

    def apply_gradient(self, gradient: np.array) -> None:
        update = self.optimizer.update(gradient)
        self.policy.increment_update(update)

    def evaluation(self) -> float:
        total_r = 0
        for i_evaluation in range(self.num_evaluation):
            done = False
            obs = self.env.reset()
            while not done:
                a = self.policy.forward(obs)    # Todo: scale
                obs, r, done, _ = self.env.step(a)
                total_r += r
        return total_r / self.num_evaluation

    def save_policy(self, save_path: str, remark: str) -> None:
        check_path(save_path)
        np.save(save_path + remark, self.policy.get_params())
        print(f"------Policy parameters saved to {save_path}------")