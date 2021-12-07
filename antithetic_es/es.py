from typing import *
import numpy as np
import ray
from master import Master


def vanilla_es(master: Master) -> int:
    pass


def antithetic_es(master: Master) -> int:
    timesteps = 0
    delta_sps = []
    rewards = []
    filter_mean, filter_s, filter_count = [], [], 0

    policy_remote = ray.put(master.policy.get_params())
    rollout_remote = [worker.do_rollouts_anti.remote(policy_remote) for worker in master.workers]
    results = ray.get(rollout_remote)
    
    for result in results:
        timesteps += result['rollout_steps']
        delta_sps += result['delta_sp']
        rewards += result['rewards']
        filter_mean.append(result['filter_stats'][0])
        filter_s.append(result['filter_stats'][1])
        filter_count += result['filter_stats'][2]

    delta_sps = np.array(delta_sps)
    rewards = np.array(rewards, dtype=np.float64)  # [num_rollout, 2]

    gradient = master.gradient_estimation_anti(rewards, delta_sps)
    master.apply_gradient(gradient)
    master.synchronous_filters(filter_mean, filter_s, filter_count)

    return timesteps


def finite_different_es(master: Master) -> int:
    pass