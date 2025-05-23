import numpy as np
import random

def perturbation_average_PDP(m, epsilon, b, delta):
    """
    Perturbation Mechanism for Average Value as described in Algorithm 1.
    
    Parameters:
        m (float): The average value m ∈ [-1, 1]
        epsilon (float): The privacy budget ε
        b (float): Upper bound for the perturbation range [-b, b]
        delta (float): A small threshold used in perturbation
    
    Returns:
        float: The perturbed value m̃
    """

    # q = 2 / ((np.exp(epsilon) - 1) * delta * (2*b - delta))
  
    x = np.random.uniform(0, 1)
    
    # probability q + q(exp(epsilon)) != 1
    # so we must normalize probabilites.
    if x <= np.exp(epsilon) / (np.exp(epsilon) + 1):
        # Sample m̃ uniformly from L(δ, m)
        # L(δ, m) is the region centered on m with length δ
        m_tilde = np.random.uniform(m - delta / 2, m + delta / 2)
    else:
        m_tilde = np.random.uniform(m - b, m + b)
    
    
    if m_tilde > 1:
        m_tilde = 1
    elif m_tilde < -1:
        m_tilde = -1
    return float(m_tilde)


def perturbation_eigenvector_GPM(eigenvector, epsilon):
    """
    Perturbation Mechanism for Eigenvector as described in Algorithm 2.
    
    Parameters:
        eigenvector (list): The eigenvector of values has elements ∈ [-1, 1]
        epsilon (float): The privacy budget ε
    
    Returns:
        list: The perturbed values of eigenvector
    """

    new_eigenvector = np.copy(eigenvector)
    d = len(eigenvector)
    vector = np.zeros(d)

    # set Bernoulli variable
    x = np.random.uniform(0, 1)
    p_A = np.exp(epsilon) / (np.exp(epsilon) + 1)
    X = 0
    if x < p_A:
        X = 1
    
    # set k
    k = 1
    if X == 1: # choose even number
        k = random.randrange(0, int((d+1)/4), 2)
    else:      # choose odd number
        k = random.randrange(1, int((d+1)/4), 2)

    # set k bits of vector to 1
    indices = random.sample(range(d), k)
    vector[indices] = 1

    for index, v in enumerate(vector):
        low_value = 0
        high_value = 0

        if v == 0 or np.abs(eigenvector[index]) < 0.5:
            # low_value = (eigenvector[index] * np.exp(epsilon) - 1) / (np.exp(epsilon) - 1)
            # high_value = (eigenvector[index] * np.exp(epsilon) + 1) / (np.exp(epsilon) - 1)
            low_value = eigenvector[index]
            high_value = eigenvector[index]
        else:
            low_value_left = -1 * ((np.exp(epsilon) + 1) / (np.exp(epsilon) - 1))
            high_value_left = (eigenvector[index] * np.exp(epsilon) - 1) / (np.exp(epsilon) - 1)

            low_value_right =  (eigenvector[index] * np.exp(epsilon) + 1) / (np.exp(epsilon) - 1)
            high_value_right =  (np.exp(epsilon) + 1) / (np.exp(epsilon) - 1)

            # left_len = high_value_left - low_value_left
            # right_len = high_value_right - low_value_right
            # probability_left = left_len / (left_len + right_len)

            if eigenvector[index] <= high_value_left:
                high_value = high_value_left
                low_value = low_value_left
            else:
                high_value = high_value_right
                low_value = low_value_right

        new_eigenvector[index] = np.random.uniform(low_value, high_value)

    return new_eigenvector
