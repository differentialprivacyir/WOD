import numpy as np
from typing import Sequence

class Rappor_Class:
    def __init__(self, eps_perm, eps_1, seed):
        """
        Parameters:
            dimensions (int): number of dimensions
            eps (float): privacy budget ε > 0
            seed (float): random seed
        """
        self.rng = np.random.default_rng(seed=seed)
        
        # SUE parameters for round 1
        self.p1 = np.exp(eps_perm / 2) / (np.exp(eps_perm / 2) + 1)
        self.q1 = 1 - self.p1

        # SUE parameters for round 2
        self.p2 = - (np.sqrt((4 * np.exp(7 * eps_perm / 2) - 4 * np.exp(5 * eps_perm / 2) - 4 * np.exp(
            3 * eps_perm / 2) + 4 * np.exp(eps_perm / 2) + np.exp(4 * eps_perm) + 4 * np.exp(3 * eps_perm) - 10 * np.exp(
            2 * eps_perm) + 4 * np.exp(eps_perm) + 1) * np.exp(eps_1)) * (np.exp(eps_1) - 1) * (
                            np.exp(eps_perm) - 1) ** 2 - (
                            np.exp(eps_1) - np.exp(2 * eps_perm) + 2 * np.exp(eps_perm) - 2 * np.exp(
                        eps_1 + eps_perm) + np.exp(eps_1 + 2 * eps_perm) - 1) * (
                            np.exp(3 * eps_perm / 2) - np.exp(eps_perm / 2) + np.exp(eps_perm) - np.exp(
                        eps_1 + eps_perm / 2) - np.exp(eps_1 + eps_perm) + np.exp(eps_1 + 3 * eps_perm / 2) + np.exp(
                        eps_1 + 2 * eps_perm) - 1)) / ((np.exp(eps_1) - 1) * (np.exp(eps_perm) - 1) ** 2 * (
                    np.exp(eps_1) - np.exp(2 * eps_perm) + 2 * np.exp(eps_perm) - 2 * np.exp(eps_1 + eps_perm) + np.exp(
                eps_1 + 2 * eps_perm) - 1))
        self.q2 = 1 - self.p2

    def RAPPOR_Client(self, input_sequence: Sequence[float], k: int):
        # The analytical analysis of how to calculate parameters (p1, q2, p2, q2) is from [2]

        # Cache for memoized values
        lst_memoized = {val:None for val in range(k)}
        
        # List of sanitized reports throughout \tau data collections
        sanitized_reports = []
        for input_data in input_sequence:
            
            # Unary encoding
            input_ue_data = np.zeros(k)
            input_ue_data[input_data] = 1
            
            if lst_memoized[input_data] is None: # If hashed value not memoized

                # Memoization
                first_sanitization = self.UE_Client(input_ue_data, k, self.p1, self.q1)
                lst_memoized[input_data] = first_sanitization

            else: # Use already memoized hashed value
                first_sanitization = lst_memoized[input_data]
            
            sanitized_reports.append(self.UE_Client(first_sanitization, k, self.p2, self.q2))
        
        # Number of data value changes, i.e, of privacy budget consumption
        final_budget = sum([val is not None for val in lst_memoized.values()])

        return sanitized_reports, final_budget

    def UE_Client(self, input_ue_data, k, p, q):
        """
        Unary Encoding (UE) protocol
        """
        
        # Initializing a zero-vector
        sanitized_vec = np.zeros(k)

        # UE perturbation function
        for ind in range(k):
            if input_ue_data[ind] != 1:
                rnd = self.rng.random()
                if rnd <= q:
                    sanitized_vec[ind] = 1
            else:
                rnd = self.rng.random()
                if rnd <= p:
                    sanitized_vec[ind] = 1
        return sanitized_vec

    def RAPPOR_Aggregator(self, ue_reports):
        # Number of reports
        n = len(ue_reports)

        # Ensure non-negativity of estimated frequency
        est_freq = ((sum(ue_reports) - n * self.q1 * (self.p2 - self.q2) - n * self.q2) / (n * (self.p1 - self.q1) * (self.p2 - self.q2))).clip(0)
        
        # Re-normalized estimated frequencies
        norm_est_freq = np.nan_to_num(est_freq / sum(est_freq))

        return norm_est_freq
