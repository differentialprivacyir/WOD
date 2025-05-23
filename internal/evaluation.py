import numpy as np
from sklearn.metrics import mean_squared_error

from itertools import zip_longest
from tabulate import tabulate   # pip install tabulate

def findMSE(original_data, perturbed_data):
    """
    Mean squared error regression loss.

    Parameters:
        original_data : array-like of shape (n_samples,) or (n_samples, n_outputs) Ground truth (correct) target values.
        perturbed_data : array-like of shape (n_samples,) or (n_samples, n_outputs) Estimated target values.
    
    Returns:
        loss : float or ndarray of floats
    
    Examples:
        >>> from sklearn.metrics import mean_squared_error
        >>> y_true = [3, -0.5, 2, 7]
        >>> y_pred = [2.5, 0.0, 2, 8]
        >>> mean_squared_error(y_true, y_pred)
        0.375
        >>> y_true = [3, -0.5, 2, 7]
        >>> y_pred = [2.5, 0.0, 2, 8]
        >>> mean_squared_error(y_true, y_pred, squared=False)
        0.612...
        >>> y_true = [[0.5, 1],[-1, 1],[7, -6]]
        >>> y_pred = [[0, 2],[-1, 2],[8, -5]]
        >>> mean_squared_error(y_true, y_pred)
        0.708...
        >>> mean_squared_error(y_true, y_pred, squared=False)
        0.822...  
    """
    return mean_squared_error(original_data, perturbed_data)

def variation_distance(p, q):
    """
    Total variation distance between two discrete distributions p and q.

    Both p and q are converted to NumPy arrays, normalised to sum to 1
    (in case they aren't already), and must have the same shape.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")

    # normalise to ensure they are probability distributions
    if p.sum() != 0:
        p = p / p.sum()
    if q.sum() != 0:
        q = q / q.sum()


    return 0.5 * np.sum(np.abs(p - q))


def average_variation_distance(list_a, list_b):
    """
    Average the variation distance over two equally-long lists of arrays.
    """
    if len(list_a) != len(list_b):
        raise ValueError("Both lists must have the same number of elements")

    distances = [variation_distance(a_i, b_i) for a_i, b_i in zip(list_a, list_b)]
    return distances, np.mean(distances)


def print_table(list1, list2, title1, title2):
    rows = list(zip_longest(list1, list2, fillvalue=""))   # keeps rows even if lengths differ
    print(tabulate(rows, headers=[title1, title2], tablefmt="github"))

def test_AVD():
    # --- Numerical example ------------------------------------------------------
    a = [
        np.array([0.2, 0.8]),          # distribution 1
        np.array([0.1, 0.4, 0.5]),     # distribution 2
        np.array([0.3, 0.7])           # distribution 3
    ]

    b = [
        np.array([0.2, 0.7]),
        np.array([0.09, 0.4, 0.6]),
        np.array([0.25, 0.75])
    ]

    pairwise_distances, avg_distance = average_variation_distance(a, b)
    print('avg_distance is', avg_distance)
