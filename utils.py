import sys
from typing import *
import gym
import numpy as np
import ray
import os
import datetime

import yaml


def check_path(path: str) -> None:
    if os.path.exists(path) is not True:
        os.makedirs(path)


def create_experiment_dir(config: Dict) -> None:
    exp_name = f"{config['env_config']['env_name']}_{datetime.datetime.now().strftime('%m-%d_%H-%M')}"
    result_path = config['result_path'] + exp_name + '/'
    check_path(result_path)
    config.update({'result_path': result_path})

    with open(result_path + 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, indent=2)
    
    return config

@ray.remote
def create_shared_noise(seed: int, count: int = 2500000):
    noise = np.random.RandomState(seed).randn(count).astype(np.float64)
    return noise


class ShareNoiseTable(object):
    def __init__(self, noise: np.float64, seed: int) -> None:
        super().__init__()
        self.noise = noise
        self.random_generator = np.random.RandomState(seed)

    def get_noise(self, start_point: int, size: int) -> np.array:
        return self.noise[start_point : start_point + size]

    def sample_start_point(self, size: int) -> int:
        sp = self.random_generator.randint(0, len(self.noise) - size + 1)
        return sp
    
    def sample_delta(self, size: int) -> Tuple[int, np.array]:
        sp = self.sample_start_point(size)
        noise = self.get_noise(sp, size)
        return sp, noise


def batch_yield(items: List, batch_size: int) -> Tuple:
    assert batch_size > 1
    batch = []
    for x in items:
        batch.append(x)
        if len(batch) == batch_size:
            yield tuple(batch)
            del batch[:]
    if batch:
        yield tuple(batch)


def batch_weighted_sum(rewards: np.array, deltas: List[np.array], batch_size: int) -> Tuple[np.array, int]:
    total = 0
    num_items_summed = 0
    for batch_r, batch_delta in zip(batch_yield(rewards, batch_size), batch_yield(deltas, batch_size)):
        assert len(batch_r) == len(batch_delta) <= batch_size
        total += np.dot(np.array(batch_r, dtype=np.float64), np.array(batch_delta, dtype=np.float64))
        num_items_summed += len(batch_delta)
    return total, num_items_summed


def supplement_config(config: Dict):
    env_config = config['env_config']
    model_config = config['model_config']
    
    env = gym.make(env_config['env_name'])
    model_config.update({
        'in_dim': env.observation_space.shape[0],
        'out_dim': env.action_space.shape[0],
        'action_low': float(env.action_space.low[0]),
        'action_high': float(env.action_space.high[0])
    })

    config.update({
        'model_config': model_config
    })

    return config