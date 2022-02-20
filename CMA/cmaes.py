from optparse import Option
from tkinter import Menu
from typing import Any, Dict, List, Tuple, Optional, cast
import math
from matplotlib.pyplot import axis
import numpy as np
from ES.CMA.utils import _is_valid_bounds, _compress_symmetric, _decompress_symmetric
from pytest import param


_EPS = 1e-8
_MEAN_MAX = 1e32
_SIGMA_MAX = 1e32


class CMA:
    def __init__(
        self,
        mean: np.ndarray,
        sigma: float,
        bounds: Optional[np.ndarray] = None,
        n_max_resampling: int = 100,
        seed: Optional[int] = None,
        population_size: Optional[int] = None,
        cov: Optional[np.ndarray] = None
    ) -> None:
        assert sigma > 0, 'Overall std sigma must be non-zero positive'
        assert np.all(np.abs(mean) < _MEAN_MAX), f'Abs of all elements of mean vector must be less than {_MEAN_MAX}'

        n_dim = len(mean)
        if population_size is None:
            population_size = 4 + math.floor(3 * math.log(n_dim))
        assert population_size > 0, 'population size must be non-zero positive'

        mu = population_size // 2   # Elites number

        weight_prime = np.array([
            math.log((population_size + 1) / 2) - math.log(i + 1) for i in range(population_size)
        ])

        mu_eff = (np.sum(weight_prime[:mu]) ** 2) / np.sum(weight_prime[:mu] ** 2)  # mu_effective for measurement 
        mu_eff_minus = (np.sum(weight_prime[mu:]) ** 2) / np.sum(weight_prime[mu:] ** 2)    # mu_effective for negative weights for measurement

        # learning rate for mean
        c_m = 1

        # learning rate for rank-one update
        alpha_cov = 2
        c_1 = alpha_cov / ((n_dim + 1.3) ** 2 + mu_eff)
        ## learning rate for cumulation for the rank-one updte
        c_c = (4 + mu_eff / n_dim) / (n_dim + mu_eff + 5)
        assert c_c <= 1, "Invalid learning rate for cumulation for the rank-one update"
        
        # learning rate for rank-mu update
        c_mu = min(
            1 - c_1 - 1e-8,
            alpha_cov * (mu_eff - 2 + 1 / mu_eff) / ((n_dim + 2) ** 2 + alpha_cov * mu_eff / 2)
        )
        assert c_1 <= 1 - c_mu, "Invalid learning rate for rank-one update"
        assert c_mu <= 1 - c_1, "Invalid learning rate for rank-mu update"
        
        # learning rate for sigma (step size control)
        c_sigma = (mu_eff + 2) / (n_dim + mu_eff + 5)   # lr
        d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (n_dim + 1)) - 1) + c_sigma   # damping
        assert c_sigma < 1, "Invalid learning control for cumulation for step-size control"

        min_alpha = min(
            1 + c_1 / c_mu,
            1 + (2 * mu_eff_minus) / (mu_eff + 2),
            (1 - c_1 - c_mu) / (n_dim * c_mu)
        )

        positive_sum = np.sum(weight_prime[weight_prime>0])
        negative_sum = np.sum(np.abs(weight_prime[weight_prime<0]))
        weights = np.where(
            weight_prime > 0,
            1 / positive_sum * weight_prime,
            min_alpha / negative_sum * weight_prime
        )

        self._n_dim = n_dim
        self._popsize = population_size
        self._mu = mu
        self._mu_eff = mu_eff
        
        self._c_m = c_m
        self._c_1 = c_1
        self._c_c = c_c
        self._c_mu = c_mu
        self._c_sigma = c_sigma
        self._d_sigma = d_sigma

        self._weights = weights

        # E||N(0, I)||
        self._chi_n = math.sqrt(self._n_dim) * (1.0 - (1.0 / (4.0 * self._n_dim)) + 1.0 / (21.0 * (self._n_dim ** 2)))
        
        # evolution path
        self._p_sigma = np.zeros(n_dim)
        self._p_c = np.zeros(n_dim)

        self._mean = mean
        if cov is None:
            self._C = np.eye(n_dim)
        else:
            self._C = cov
            assert cov.shape == (n_dim, n_dim), "Invalid shape of the covariance matrix"

        self._sigma = sigma
        self._D: Optional[np.ndarray] = None  # diagonal matrix with eigenvalues of C
        self._B: Optional[np.ndarray] = None  # matrix composed by eigenvectors of C

        assert bounds is None or _is_valid_bounds(bounds, mean), "Invalid bounds"
        self._bounds = bounds
        
        self._n_max_resampling = n_max_resampling

        self._g = 0 # generation
        self._rng = np.random.RandomState(seed)

        # Termination criteria
        self._tolx = 1e-12 * sigma
        self._tolxup = 1e4
        self._tolfun = 1e-12
        self._tolconditioncov = 1e14

        self._funhist_term = 10 + math.ceil(20 * n_dim / population_size)
        self._funhist_values = np.empty(self._funhist_term * 2)

    def __getstate__(self) -> Dict[str, Any]:
        attrs = {}
        for name in self.__dict__:
            if name == '_rng':
                continue
            if name == '_C':
                sym1d = _compress_symmetric(self._C)
                attrs['_c_1d'] = sym1d
                continue
            attrs[name] = getattr(self, name)
        return attrs

    def __setstate__(self, state: Dict[str, Any]) -> None:
        state['_C'] = _decompress_symmetric(state['_c_1d'])
        del state['_c_1d']
        self.__dict__.update(state)
        setattr(self, '_rng', np.random.RandomState())

    @property
    def dim(self) -> int:
        return self._n_dim
    
    @property
    def population_size(self) -> int:
        return self._popsize

    @property
    def generation(self) -> int:
        return self._g

    def set_bounds(self, bounds: Optional[np.ndarray]) -> None:
        assert bounds is None or _is_valid_bounds(bounds, self._mean), "Invalid bounds"
        self._bounds = bounds

    def _eigen_decomposition(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._B is not None and self._D is not None:
            return self._B, self._D
        self._C = (self._C + self._C.T) / 2
        D2, B = np.linalg.eigh(self._C)
        D = np.sqrt(np.where(D2 < 0, _EPS, D2))
        self._C = np.dot(np.dot(B, np.diag(D ** 2)), B.T)
        self._B, self._D = B, D
        return B, D

    def _sample_solution(self) -> np.ndarray:
        B, D = self._eigen_decomposition()
        z = self._rng.randn(self._n_dim)
        y = cast(np.ndarray, B.dot(np.diag(D))).dot(z)
        x = self._mean + self._sigma * y
        return x

    def _is_feasible(self, param: np.ndarray) -> bool:
        if self._bounds is None:
            return True
        return cast(bool, np.all(param >= self._bounds[: ,0]) and np.all(param <= self._bounds[:, 1]))

    def _repair_infeasible_params(self, param: np.ndarray) -> np.ndarray:
        if self._bounds is None:
            return param
        param = np.where(param < self._bounds[:, 0], self._bounds[:, 0], param)
        param = np.where(param > self._bounds[:, 1], self._bounds[:, 1], param)
        return param

    def ask(self) -> np.ndarray:
        """sample a parameter

        Returns:
            np.array: sample from the dist
        """
        for i in range(self._n_max_resampling):
            x = self._sample_solution()
            if self._is_feasible(x):
                return x
        x = self._sample_solution()
        x = self._repair_infeasible_params(x)
        return x

    def tell(self, solutions: List[Tuple[np.ndarray, float]]) -> None:
        """Update based on the samples and corresponding values

        Args:
            solutions (List[Tuple[np.array, float]]): [(param1, value1), (param2, value2)]
        """
        assert len(solutions) == self._popsize, "Must tell popsize-length solutions"
        for s in solutions:
            assert np.all(np.abs(s[0]) < _MEAN_MAX), f"Abs of all param values must be less than {_MEAN_MAX} to avoid overflow errors"
        
        self._g += 1

        solutions.sort(key= lambda s: s[1])

        # Stores 'best' and 'worst' values of the
        # last 'self._funhist_term' generations.
        funhist_idx = 2 * (self.generation % self._funhist_term)
        self._funhist_values[funhist_idx] = solutions[0][1]
        self._funhist_values[funhist_idx + 1] = solutions[-1][1]

        # Sample new population of search_points for k=1, ..., popsize
        B, D = self._eigen_decomposition()
        self._B, self._D = None, None

        x_k = np.array([s[0] for s in solutions])   # ~ N(m, sigma^2 * C)
        y_k = (x_k - self._mean) / self._sigma  # ~ N(0, C)

        # Selection
        y_w = np.sum(y_k[: self._mu].T * self._weights[: self._mu], axis=1)
        # Mean update
        self._mean += self._c_m * self._sigma * y_w

        # Step-size control
        C_2 = cast(np.ndarray, cast(np.ndarray, B.dot(np.diag(1 / D))).dot(B.T))    # C^(-1/2) = B D^(-1) B^T
        self._p_sigma = (1 - self._c_sigma) * self._p_sigma + math.sqrt(self._c_sigma * (2 - self._c_sigma) * self._mu_eff) * C_2.dot(y_w)
        norm_p_sigma = np.linalg.norm(self._p_sigma)
        self._sigma *= np.exp((self._c_sigma / self._d_sigma) * (norm_p_sigma / self._chi_n - 1))
        self._sigma = min(self._sigma, _SIGMA_MAX)

        # Covariance matrix update
        h_sigma_cond_left = norm_p_sigma / math.sqrt(1 - (1 - self._c_sigma) ** (2 * (self._g + 1)))
        h_sigma_cond_right = (1.4 + 2 / (self._n_dim + 1)) * self._chi_n
        h_sigma = 1.0 if h_sigma_cond_left < h_sigma_cond_right else 0.0

        self._p_c = (1 - self._c_c) * self._p_c + h_sigma * math.sqrt(self._c_c * (2 - self._c_c) * self._mu_eff) * y_w
        w_io = self._weights * np.where(self._weights >= 0, 1, self._n_dim / (np.linalg.norm(C_2.dot(y_k.T), axis=0) ** 2 + _EPS))

        delta_h_sigma = (1 - h_sigma) * self._c_c * (2 - self._c_c)
        assert delta_h_sigma <= 1

        rank_one = np.outer(self._p_c, self._p_c)
        rank_mu = np.sum(np.array([w * np.outer(y, y) for w, y in zip(w_io, y_k)]), axis=0)
        self._C = (
            (1 + self._c_1 * delta_h_sigma - self._c_1 - self._c_mu * np.sum(self._weights)) * self._C
            + self._c_1 * rank_one
            + self._c_mu * rank_mu
        )

    def should_stop(self) -> bool:
        B, D = self._eigen_decomposition()
        dC = np.diag(self._C)

        # Stop if the range of function values of the recent generation is below tolfun.
        if (
            self.generation > self._funhist_term
            and np.max(self._funhist_values) - np.min(self._funhist_values)
            < self._tolfun
        ):
            return True

        # Stop if the std of the normal distribution is smaller than tolx
        # in all coordinates and pc is smaller than tolx in all components.
        if np.all(self._sigma * dC < self._tolx) and np.all(self._sigma * self._p_c < self._tolx):
            return True

        # Stop if detecting divergent behavior.
        if self._sigma * np.max(D) > self._tolxup:
            return True

        # No effect coordinates: stop if adding 0.2-standard deviations
        # in any single coordinate does not change m.
        if np.any(self._mean == self._mean + (0.2 * self._sigma * np.sqrt(dC))):
            return True

        # No effect axis: stop if adding 0.1-standard deviation vector in
        # any principal axis direction of C does not change m. "pycma" check
        # axis one by one at each generation.
        i = self.generation % self.dim
        if np.all(self._mean == self._mean + (0.1 * self._sigma * D[i] * B[:, i])):
            return True

        # Stop if the condition number of the covariance matrix exceeds 1e14.
        condition_cov = np.max(D) / np.min(D)
        if condition_cov > self._tolconditioncov:
            return True

        return False