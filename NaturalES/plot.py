from os import path
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.npyio import load
import pandas as pd
import sys


RESULT_PATH = '/home/xukang/GitRepo/ES/NaturalES/results/'


def smooth(data: np.array, weight: np.array) -> np.array:
    smooth_data = []
    last = data[0]
    for point in data:
        point = point * (1 - weight) + last * weight
        smooth_data.append(point)
        last = point
    return np.array(smooth_data)


def smooth_and_fill_between(data: np.array, label: str, smooth_weight: float, width: float, alpha: float) -> None:
    y = smooth(data, smooth_weight)
    y_std = y.std()
    plt.plot(y, label=label)
    plt.fill_between(np.array(list(range(len(data)))), y - width * y_std, y + width * y_std, alpha=alpha)


def load_stats(path: str) -> np.array:
    with open(path, 'r') as f:
        df = pd.read_csv(f)
    data = np.array(df)
    data = data[:, -1]
    return data


def plot_NES_variants() -> None:
    path_vanilla = RESULT_PATH + 'HalfCheetah-v2_estimator-vanilla_fitness-value_12-08_10-08/stats.csv'
    path_anti = RESULT_PATH + 'HalfCheetah-v2_12-05_20-58/stats.csv'
    path_fd = RESULT_PATH + 'HalfCheetah-v2_estimator-antithetic_fitness-rank_12-08_01-17/stats.csv'

    data_vanilla = load_stats(path_vanilla)[:750]
    data_anti = load_stats(path_anti)[:750]
    data_fd = load_stats(path_fd)[:750]

    smooth_and_fill_between(data_vanilla, label='Vanilla', smooth_weight=0, width=0, alpha=0.2)
    smooth_and_fill_between(data_anti, label='Antithetic', smooth_weight=0, width=0, alpha=0.2)
    smooth_and_fill_between(data_fd, label='Finite Difference', smooth_weight=0, width=0, alpha=0.2)

    plt.legend()
    plt.grid()
    plt.xlabel('Generation')
    plt.ylabel('Episode reward')
    plt.show()


def plot_NES_rank() -> None:
    path_value = RESULT_PATH + 'HalfCheetah-v2_12-05_20-58/stats.csv'
    path_rank = RESULT_PATH + 'HalfCheetah-v2_estimator-antithetic_fitness-rank_12-08_01-17/stats.csv'
    
    data_v = load_stats(path_value)[:750]
    data_r = load_stats(path_rank)[:750]

    smooth_and_fill_between(data_v, label='True value', smooth_weight=0, width=0., alpha=0.2)
    smooth_and_fill_between(data_r, label='Rank normalization', smooth_weight=0., width=0., alpha=0.2)
    plt.legend(loc='lower right')
    plt.grid()
    plt.xlabel('Generation')
    plt.ylabel('Episode reward')
    plt.show()


def plot_NES_workers() -> None:
    pass




if __name__ == '__main__':
    plot_NES_variants()
    #plot_NES_rank()