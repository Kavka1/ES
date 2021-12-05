from typing import Dict, List, Tuple, Type
import numpy as np
from master import Master
from es import ES_loop
import os
import yaml
import pandas as pd
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
    iteration_log, step_log, score_log = [], [], []

    for iter in range(config['max_iteration']):
        steps = ES_loop(master)
        eval_r = master.evaluation()
        total_steps += steps

        print(f"Iteration: {iter}  total_steps: {total_steps}  evaluation_r: {eval_r}")
        
        if eval_r > master.best_score:
            master.best_score = eval_r
            master.save_policy(config['result_path'] + 'models/', remark = 'params_best')
            master.save_filter(config['result_path'] + 'filters/', remark = 'params_best')
        if iter % config['save_model_interval'] == 0:
            master.save_policy(config['result_path'] + 'models/', remark = f'params_{iter}')
            master.save_filter(config['result_path'] + 'filters/', remark = f'params_{iter}')

        iteration_log.append(iter)
        step_log.append(total_steps)
        score_log.append(eval_r)
        df = pd.DataFrame({'Iteration': iteration_log, 'Timesteps': step_log, 'Evaluation_score': score_log})
        df.to_csv(config['result_path'] + 'stats.csv', index=False)

        logger.add_scalar('Metric/evaluation_score_timestep', eval_r, total_steps)
        logger.add_scalar('Metric/evaluation_score_iter', eval_r, iter)


if __name__ == '__main__':
    train()