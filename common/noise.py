from typing import *
import numpy as np
import ray


@ray.remote
def create_shared_noise(seed: int, count: int = 2500000) -> np.array:
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