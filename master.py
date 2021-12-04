from _typeshed import Self
from typing import *
import numpy as np
import gym
import ray
from model import Policy
from utils import ShareNoiseTable, create_shared_noise
from worker import Worker


class Master(object):
    def __init__(self, args: Dict) -> None:
        super().__init__()

        self.model_params = args['model_params']
        self.num_workers = args['num_workers']
        self.noise_table_size = args['noise_table_size']
        self.env_name = args['env_name']
        self.num_rollouts = args['num_rollouts']
        self.num_evaluation = args['num_evaluation']

        self.noise_table_seed = args['noise_table_seed']
        self.delta_sample_seed = args['noise_sample_seed']

        self.policy = Policy(args['model_params'])

        self.init_noise_table()
        self.init_workers()
    
    def init_noise_table(self) -> None:
        self.noise_table = create_shared_noise(self.noise_table_seed, self.noise_table_size).remote()
        self.deltas = ShareNoiseTable(ray.get(self.noise_table), self.delta_sample_seed)

    def init_workers(self) -> None:
        self.workers = [Worker.remote(
                        env_name = self.env_name,
                        model_params = self.model_params,
                        noise_table = self.noise_table,
                        delta_sample_seed = self.delta_sample_seed,
                        num_rollouts = self.num_rollouts,
                        num_evaluation = self.num_evaluation,) for _ in range(self.num_workers)]