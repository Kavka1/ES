from typing import Dict, List, Tuple, Type
import numpy as np
from master import Master
from es import ES_loop
import os
import yaml
from torch.utils.tensorboard import SummaryWriter
from utils import create_experiment_dir, supplement_config


def load_config() -> Dict:
    file_path = os.path.abspath('.') + '/config.yaml'
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read())
    return config


def train() -> None:
    config = load_config()
    config = supplement_config(config)
    config = create_experiment_dir(config)

    logger = SummaryWriter(config['result_path'] + '/logs')

    master = Master(config)
    total_steps = 0
    for iter in range(config['max_iteration']):
        steps = ES_loop(master)
        total_steps += steps
        eval_r = master.evaluation()

        if eval_r > master.best_score:
            master.best_score = eval_r
            master.save_policy(config['result_path'] + 'models/', remark = 'params_best')
            master.save_filter(config['result_path'] + 'filters/', remark = 'params_best')
        if iter % config['save_model_interval'] == 0:
            master.save_policy(config['result_path'] + 'models/', remark = f'params_{iter}')
            master.save_filter(config['result_path'] + 'filters/', remark = f'params_{iter}')

        print(f"Iteration: {iter}  total_steps: {total_steps}  evaluation_r: {eval_r}")

        logger.add_scalar('Metric/evaluation_score_timestep', eval_r, total_steps)
        logger.add_scalar('Metric/evaluation_score_iter', eval_r, iter)


if __name__ == '__main__':
    train()