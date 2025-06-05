import numpy as np
from math import exp
from scipy.special import comb
from typing import Sequence

class Duchi_Class:
    def __init__(self, dimensions, eps, seed):
        """
        Parameters:
            dimensions (int): number of dimensions
            eps (float): privacy budget ε > 0
            seed (float): random seed
        """
        self.d = dimensions
        self.eps = eps
        self.rng = np.random.default_rng(seed=seed)

        if self.eps <= 0:
            raise ValueError("Epsilon must be positive.")

    def Duchi_multidim_ldp(self, x: Sequence[float]):
        """
        One-shot ε-LDP perturbation of a d-dimensional numeric vector
        using Duchi et al.'s mechanism (Algorithm 3).

        Parameters
        ----------
        x   : sequence of floats (values must lie in [-1,1])

        Returns
        -------
        numpy.ndarray of shape (d,) with entries in {-B, +B}
        """
        x = np.asarray(x, dtype=float)
        if not np.all(np.abs(x) <= 1):
            raise ValueError("All coordinates of x must be in [-1, 1].")

        B = _compute_B(self.d, self.eps)

        # Step 1 – sample helper vector v ∈ {-1,1}ᵈ
        probs = 0.5 + 0.5 * x               # P[v_j = +1]
        v = np.where(self.rng.random(self.d) < probs, 1.0, -1.0)

        # Step 3 – Bernoulli for choosing T⁺ / T⁻
        u = self.rng.random() < (exp(self.eps) / (exp(self.eps) + 1))

        # Draw t* uniformly from T⁺ or T⁻ by rejection sampling
        sign_target = 1 if u else -1
        while True:
            t_star = self.rng.choice([-B, B], size=self.d)
            if np.dot(t_star / B, v) * sign_target >= 0:  # dot ≥ 0 for T⁺, ≤ 0 for T⁻
                return t_star


    def Duchi_randomize(self, t: float) -> float:
        """
        One-dimensional Duchi et al. mechanism (Algorithm 1).

        Parameters
        ----------
        t : float
            Real number in [-1, 1] to be privatized.

        Returns
        -------
        float
            Privatized value in {-(e^ε + 1)/(e^ε - 1),  (e^ε + 1)/(e^ε - 1)}.
        """
        if not (-1.0 <= t <= 1.0):
            raise ValueError("Input must lie in [-1, 1].")

        # Convenience variables
        e_eps = np.exp(self.eps)
        s = (e_eps + 1) / (e_eps - 1)        # magnitude of the released value
        p = (e_eps - 1) / (2 * (e_eps + 1)) * t + 0.5  # Bernoulli success prob.

        # Draw Bernoulli(u) and release ±s
        u = self.rng.uniform(0, 1) < p
        return  s if u else -s

def _compute_Cd(d: int) -> float:
    """
    Compute C_d as in Eq. (9) of the paper.
    """
    if d % 2:   # d is odd
        return (2 ** (d - 1)) / comb(d - 1, (d - 1) // 2)
    else:       # d is even
        return (2 ** (d - 1) + 0.5 * comb(d, d // 2)) / comb(d - 1, d // 2)

def _compute_B(d: int, eps: float) -> float:
    """
    Compute B as in Eq. (10).
    """
    # C_d = _compute_Cd(d)
    C_d = 1  # Complicated formula exist at Duchi paper and I decide to set 1 for better utility
    return ((exp(eps) + 1) / (exp(eps) - 1)) * C_d
