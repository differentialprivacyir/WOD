import numpy as np
import xxhash


def reduce_domain(evolution_dataset, eps_perm, rnd_seed, alpha, optimal=True):
    print('Reducing domain ...')
    # BiLOLOHA parameter
    g = 2
    
    if optimal:
        # Optimal LH (OLOLOHA) parameter
        g = int(max(np.rint((np.sqrt(np.exp(4*eps_perm) - 14*np.exp(2*eps_perm) - 12*np.exp(2*eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+3)) + 1) - np.exp(2*eps_perm) + 6*np.exp(eps_perm) - 6*np.exp(eps_perm*alpha) + 1) / (6*(np.exp(eps_perm) - np.exp(eps_perm*alpha)))), 2))
        print('Optimal domain size is', g)

    hashed_dataset = []
    for data in evolution_dataset:
        hashed_dataset.append([(xxhash.xxh32(str(value), seed=rnd_seed).intdigest() % g) for value in data])
    
    return hashed_dataset, g

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
