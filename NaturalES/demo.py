import os
from typing import List, Type, Dict
import argparse
import numpy as np
import yaml

from ES.common.obs_filter import MeanStdFilter
from ES.common.model import Policy
from ES.common.env import Env_wrapper


def load_exp(result_path: str) -> Dict:
    assert os.path.exists(result_path)
    with open(result_path + 'config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f.read())
    return config


def create_arguments() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_path', type=str, default='results/Walker2d-v2_12-06_14-46')
    parser.add_argument('--num_rollout', type=int, default=50)
    args = parser.parse_args()
    args = vars(args)

    return args


def demo() -> None:
    args = create_arguments()
    result_path = f"/home/xukang/GitRepo/ES/antithetic_es/results/{args['exp_path'].split('/')[-1]}/"

    config = load_exp(result_path)

    policy = Policy(config['model_config'])
    env = Env_wrapper(config['env_config'])
    obs_filter = MeanStdFilter(shape = env.observation_space.shape)
    
    policy.load_params(result_path + 'models/params_best.npy')
    obs_filter.load_params(result_path + 'filters/params_best.npy')
    
    for i_rolllout in range(args['num_rollout']):
        episode_reward, episode_step = 0, 0
        done = False
        obs = env.reset()
        while not done:
            env.render()
            a = policy.forward(obs_filter(obs))
            obs, r, done, _ = env.step(a)
            episode_reward += r
            episode_step += 1

        print(f"i_rollout: {i_rolllout}  episode_steps: {episode_step}  episode_reward: {episode_reward}")


if __name__ == '__main__':
    demo()