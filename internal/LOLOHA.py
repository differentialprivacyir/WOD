import numpy as np
import xxhash
from sys import maxsize


def reduce_domain_row(evolution_row, g, user_hash_function):
    return [(xxhash.xxh32(str(value), seed=user_hash_function).intdigest() % g) for value in evolution_row]

def compute_optimal_domain_size(eps_perm, alpha, optimal=True):
    # BiLOLOHA parameter
    g = 2
    if not optimal:
        g = int(max(np.rint((np.sqrt(np.exp(4*eps_perm) - 14*np.exp(2*eps_perm) - 12*np.exp(2*eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+3)) + 1) - np.exp(2*eps_perm) + 6*np.exp(eps_perm) - 6*np.exp(eps_perm*alpha) + 1) / (6*(np.exp(eps_perm) - np.exp(eps_perm*alpha)))), 2))
        print('Optimal domain size is', g)

    return g

def reduce_domain_dataset(evolution_dataset, g):
    print('Reducing domain ...')
    user_hash_functions = []
    hashed_evolution_dataset = []

    for row in evolution_dataset:
        # Random 'hash function', i.e., a seed to use xxhash package
        user_seed = np.random.randint(0, maxsize, dtype=np.int64)
        hashed_evolution_dataset.append(reduce_domain_row(row, g, user_seed))
        user_hash_functions.append(user_seed) 
    
    return hashed_evolution_dataset, user_hash_functions

def perturbation_GRR(hashed_evolution_dataset, domain_size_g, eps_perm, eps_1):
    print('Perturbation with GRR ...')
    # GRR parameters for round 1
    p1_llh = np.exp(eps_perm) / (np.exp(eps_perm) + domain_size_g - 1)
    q1_llh = (1 - p1_llh) / (domain_size_g-1)
    
    # GRR parameters for round 2
    p2_llh = (q1_llh - np.exp(eps_1) * p1_llh) / ((-p1_llh * np.exp(eps_1)) + domain_size_g*q1_llh*np.exp(eps_1) - q1_llh*np.exp(eps_1) - p1_llh*(domain_size_g-1)+q1_llh)
    q2_llh = (1 - p2_llh) / (domain_size_g-1)


    sanitization_datasest = []
    for input_sequence in hashed_evolution_dataset:
        # Cache for memoized values
        lst_memoized = {val:None for val in range(domain_size_g)}
        sanitization_sequence = []
        
        for input_data in input_sequence:
            if lst_memoized[input_data] is None: # If hashed value not memoized
                # Memoization
                first_sanitization = GRR_Client(input_data, domain_size_g, p1_llh)
                lst_memoized[input_data] = first_sanitization
            else: # Use already memoized hashed value
                first_sanitization = lst_memoized[input_data]

            second_sanitization = GRR_Client(first_sanitization, domain_size_g, p2_llh)
            sanitization_sequence.append(second_sanitization)
        
        sanitization_datasest.append(sanitization_sequence)

    return sanitization_datasest


def GRR_Client(input_data, k, p):
    """
    Generalized Randomized Response (GRR) protocol
    """
    
    # Mapping domain size k to the range [0, ..., k-1]
    domain = np.arange(k) 

    # GRR perturbation function
    rnd = np.random.random()
    if rnd <= p:
        return input_data

    else:
        return np.random.choice(domain[domain != input_data])


def LOLOHA_Aggregator(reports, user_hash_functions, k, eps_perm, eps_1, alpha, optimal=True):    
    """
    Estimate frequency of reports of all users at one period of time
    
    Parameters:
        reports: list (reports of all users at one period of time)
        user_hash_functions: list (random seeds with size number of users)
        k: int (domain size of original evolution data)
        eps_perm: float
        eps_1: float
        alpha: float
        optimal: bool
 
    Returns:
        list: frequency estimation
    """
    # Number of reports
    n = len(reports)
    
    # BiLOLOHA parameter
    g = 2
    
    if optimal:
    
        # Optimal LH (OLOLOHA) parameter
        g = max(np.rint((np.sqrt(np.exp(4*eps_perm) - 14*np.exp(2*eps_perm) - 12*np.exp(2*eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+3)) + 1) - np.exp(2*eps_perm) + 6*np.exp(eps_perm) - 6*np.exp(eps_perm*alpha) + 1) / (6*(np.exp(eps_perm) - np.exp(eps_perm*alpha)))), 2)

    # GRR parameters for round 1
    p1 = np.exp(eps_perm) / (np.exp(eps_perm) + g - 1)
    q1 = (1 - p1) / (g-1)

    # GRR parameters for round 2
    p2 = (q1 - np.exp(eps_1) * p1) / ((-p1 * np.exp(eps_1)) + g*q1*np.exp(eps_1) - q1*np.exp(eps_1) - p1*(g-1)+q1)
    q2 = (1 - p2) / (g-1)
    
    # Count how many times each value has been reported
    q1 = 1 / g #updating q1 in the server        
    count_report = np.zeros(k)
    for usr_val, usr_seed in zip(reports, user_hash_functions):
        for v in range(k):
            if usr_val == (xxhash.xxh32(str(v), seed=usr_seed).intdigest() % g):
                count_report[v] += 1
    
    # Ensure non-negativity of estimated frequency
    est_freq = ((count_report - n * q1 * (p2 - q2) - n * q2) / (n * (p1 - q1) * (p2 - q2))).clip(0)

    # Re-normalized estimated frequency
    norm_est_freq = np.nan_to_num(est_freq / sum(est_freq))

    return norm_est_freq
