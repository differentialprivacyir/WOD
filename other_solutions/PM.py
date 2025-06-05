import numpy as np
from typing import Sequence

class PM_Class:
    def __init__(self, dimensions, eps, seed):
        """
        Parameters:
            dimensions (int): number of dimensions
            eps (float): privacy budget
            seed (float): random seed
        """
        self.d = dimensions
        self.k = max(1, min(self.d, int(np.floor(eps / 2.5))))  # Eq. (12)
        self.eps = eps
        self.rng = np.random.default_rng(seed=seed)

    def perturb_tuple_PM(self, t: Sequence[float]) -> np.ndarray:
        """
        Implements Algorithm 4 for an input tuple t in [-1,1]^d.

        Parameters
        ----------
        t      : 1-D numpy array of length d (each entry in [-1,1])
        eps    : total privacy budget ε
        rng    : numpy Generator for reproducibility

        Returns
        -------
        t_star : perturbed tuple satisfying ε-local DP
        """
        d = len(t)
        chosen = self.rng.choice(d, size=self.k, replace=False)

        t_star = np.zeros(d, dtype=float)
        sub_eps = self.eps / self.k                             # each selected coord.

        for j in chosen:
            noisy = self.PM_randomize(t[j], sub_eps)
            # t_star[j] = (d / self.k) * noisy              # line 6 of Algorithm 4
            t_star[j] = noisy  # I dont understand d/k and remove it for better utility.

        return t_star

    def PM_randomize(self, t, eps):
        """
        Algorithm 2: Piecewise Mechanism for One-Dimensional Numeric Data.
        
        Parameters:
            t (float): input number in [-1, 1]
            eps (float): privacy budget

        Returns:
            float: The perturbed value t
        """
        x = self.rng.uniform(0, 1)
        p = np.exp(eps/2) / (np.exp(eps/2) + 1)

        C = (np.exp(eps/2) + 1) / (np.exp(eps/2) - 1)
        l_t = ((C + 1)/2) * t - ((C - 1)/2)
        r_t = l_t + C - 1
        if x < p:
            return self.rng.uniform(l_t, r_t)
        else:
            return self.random_uniform_union_range(-C, l_t, r_t, C)

    def random_uniform_union_range(self, l2, l1, r2, r1):
        """
        Choose randomly uniform in range (l2, l1) U (r1, r2)
        note that: l2 < l1 < r1 < r2
        """
        left_length = abs(l1 - l2)
        right_length = abs(r2 - r1)

        x = self.rng.uniform(0, 1)
        p_left = left_length / (left_length + right_length)
        if x < p_left:
            return self.rng.uniform(l2, l1)
        else:
            return self.rng.uniform(r1, r2)
