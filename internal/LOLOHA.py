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
    
    return hashed_dataset
