from typing import *
import numpy as np
from numpy.core.numeric import roll
import ray
import gym
from ray import worker
from master import Master


def ES_loop(master: Master) -> int:
    timesteps = 0
    delta_sps, rewards = [], []

    policy_remote = ray.put(master.policy.get_params())
    rollout_remote = [worker.do_rollouts.remote(policy_remote) for worker in master.workers]
    results = ray.get(rollout_remote)
    for result in results:
        timesteps += result['rollout_steps']
        delta_sps += result['delta_sp']
        rewards += result['rewards']

    delta_sps = np.array(delta_sps)
    rewards = np.array(rewards, dtype=np.float64)  # [num_rollout, 2]

    gradient = master.gradient_estimation(rewards, delta_sps)
    master.apply_gradient(gradient)

    return timesteps