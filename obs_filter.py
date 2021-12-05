from typing import Dict, List, Tuple, Type, Union
import numpy as np


class RunningStats(object):
    def __init__(self, shape: Union[int, Tuple]) -> None:
        super().__init__()
        
        self.shape = shape
        self._mean = np.zeros(shape = shape, dtype = np.float64)
        self._square_sum = np.zeros(shape=shape, dtype = np.float64)
        self._count = 0

    def push(self, x):
        n = self._count
        self._count += 1
        if self._count == 1:
            self._mean[...] = x
        else:
            delta = x - self._mean
            self._mean[...] += delta / self._count
            self._square_sum[...] += delta**2 * n / self._count
 
    @property
    def var(self) -> np.array:
        return self._square_sum / (self._count - 1) if self._count > 1 else np.square(self._mean)

    @property
    def std(self) -> np.array:
        return np.sqrt(self.var)


class MeanStdFilter(object):
    def __init__(self, shape: Union[int, Tuple]) -> None:
        super().__init__()

        self.shape = shape
        self.rs = RunningStats(shape)

    def __call__(self, x: np.array) -> np.array:
        assert x.shape == self.shape, (f"Filter.__call__: x.shape-{x.shape} != filter.shape-{self.shape}")
        return (x - self.rs._mean) / (self.rs.std + 1e-6)

    def push_batch(self, x_batch: np.array) -> None:
        assert len(x_batch.shape) == 2 and x_batch[0].shape == self.shape
        for x in x_batch:
            self.rs.push(x)

    def update(self, mean: np.array, square_sum: np.array, count: int) -> None:
        assert mean.shape == square_sum.shape == self.shape, (f"Filter.update: mean_shape {mean.shape} std_shape: {square_sum.shape} filter_shape: {self.shape}")
        self.rs._mean[...] = mean
        self.rs._square_sum[...] = square_sum
        self.rs._count = count

    @property
    def mean(self) -> np.array:
        return self.rs._mean
    
    @property
    def square_sum(self) -> np.array:
        return self.rs._square_sum

    @property
    def count(self) -> int:
        return self.rs._count