import numpy as np
from typing import Sequence

class BitFlipPM_Class:
    def __init__(self, b:int, d:int, eps_perm:float, seed:int):
        """
        Parameters:
            b (int): number of buckets
            d (int): number of bits each user sample/report
            eps_perm (float): privacy budget ε > 0
            seed (float): random seed
        """
        self.rng = np.random.default_rng(seed=seed)
        self.b = b
        self.d = d
        self.eps_perm = eps_perm

    # Competitor: dBitFlipPM [3]
    def dBitFlipPM_Client(self, input_sequence: Sequence[float], k:int):
        
        # calculate bulk number of user's value
        bulk_size = k / self.b
        
        # bucketized sequence
        bucket_sequence = [int(input_data / bulk_size) for input_data in input_sequence]
        
        # Select random bits and permanently memoize them
        j = self.rng.choice(range(0, self.b), self.d, replace=False)
        
        # UE matrix of b buckets
        UE_b = np.eye(self.b)
        
        # mapping {0, 1}^d possibilities of input data
        mapping_d = np.unique([UE_b[val][j] for val in bucket_sequence], axis=0)
        
        # Privacy budget consumption min(d+1, b)
        final_budget = len(mapping_d)
            
        # Cache for memoized values
        lst_memoized = {str(val): None for val in mapping_d}
        
        # List of sanitized reports throughout \tau data collections 
        sanitized_reports = []
        for bucketized_data in bucket_sequence:
            
            pattern = str(UE_b[bucketized_data][j])
            if lst_memoized[pattern] is None: # Memoize value
            
                first_sanitization = self.dBit(bucketized_data, self.b, self.d, j, self.eps_perm)
                lst_memoized[pattern] = first_sanitization
            
            else: # Use already memoized value
                first_sanitization = lst_memoized[pattern]
            
            sanitized_reports.append(first_sanitization)

        # Number of memoized responses
        nb_changes = len(np.unique([val for val in lst_memoized.values() if val is not None], axis = 0))
        
        # Boolean value to indicate if number of memoized responses equal number of bucket value changes
        detect_change = len(np.unique(bucket_sequence)) == nb_changes

        return sanitized_reports, final_budget, detect_change

    def dBitFlipPM_Aggregator(self, reports):
        
        # Estimated frequency of each bucket
        est_freq = []
        for v in range(self.b):
            h = 0
            for bi in reports:
                if bi[v] >= 0: # only the sampled bits
                    h += (bi[v] * (np.exp(self.eps_perm / 2) + 1) - 1) / (np.exp(self.eps_perm / 2) - 1)
            est_freq.append(h * self.b / (len(reports) * self.d ))
        
        # Ensure non-negativity of estimated frequency
        est_freq = np.array(est_freq).clip(0)
        
        # Re-normalized estimated frequency
        norm_est = np.nan_to_num(est_freq / sum(est_freq))
        return norm_est
    
    def dBit(self, bucketized_data, b, d, j, eps_perm):
        # SUE parameters
        p1 = np.exp(eps_perm / 2) / (np.exp(eps_perm / 2) + 1)
        q1 = 1 - p1
        
        # Unary encoding
        permanent_sanitization = np.ones(b) * - 1 # set to -1 non-sampled bits

        # Permanent Memoization
        idx_j = 0
        for i in range(b):
            if i in j: # only the sampled bits
                rand = self.rng.random()
                if bucketized_data == j[idx_j]:
                    permanent_sanitization[j[idx_j]] = int(rand <= p1)
                else:
                    permanent_sanitization[j[idx_j]] = int(rand <= q1)

                idx_j+=1
        return permanent_sanitization
