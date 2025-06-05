import numpy as np

def PM_randomize(t, epsilon):
    """
    Algorithm 2: Piecewise Mechanism for One-Dimensional Numeric Data.
    
    Parameters:
        t (float): input number in [-1, 1]
        epsilon (float): privacy budget
    
    Returns:
        float: The perturbed value t
    """
    x = np.random.uniform(0, 1)
    p = np.exp(epsilon/2) / (np.exp(epsilon/2) + 1)

    C = (np.exp(epsilon/2) + 1) / (np.exp(epsilon/2) - 1)
    l_t = ((C + 1)/2) * t - ((C - 1)/2)
    r_t = l_t + C - 1
    if x < p:
        return np.random.uniform(l_t, r_t)
    else:
        return random_uniform_union_range(-C, l_t, r_t, C)

def random_uniform_union_range(l2, l1, r2, r1):
    """
    Choose randomly uniform in range (l2, l1) U (r1, r2)
    note that: l2 < l1 < r1 < r2
    """
    left_length = abs(l1 - l2)
    right_length = abs(r2 - r1)

    x = np.random.uniform(0, 1)
    p_left = left_length / (left_length + right_length)
    if x < p_left:
        return np.random.uniform(l2, l1)
    else:
        return np.random.uniform(r1, r2)
