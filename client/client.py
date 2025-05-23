from internal.haar_transform import *
from internal.normalizer import *
from internal.perturbations import *

class Client:
    def __init__(self, epsilon, random_seed, b, delta):
        self.epsilon = epsilon
        self.b = b
        self.delta = delta
        np.random.seed(random_seed)
        random.seed(random_seed)
    
    def send_perturbed_avg_eigenvector(self, data, evolution_data):
        """
        This function sends all properties of one client to server.
        properties are collection of perturbed data with size tau.

        Parameters:
            data (list): input row
            evolution_data (list): input row

        Returns:
            list of (list, float): [new_avg, new_eignevector]
        """

        dict_memoized = {}
        haar_transform_obj= HaarTransform()
        results = []

        for e in evolution_data:
            user_data = data.copy()
            user_data.append(e)  # combine data with evolution
            key = self.array_to_string(user_data)

            if key not in dict_memoized: # If data does not memoized
                # haar transform
                avg, eigenvector = haar_transform_obj.transnform(user_data)

                # perturbations
                new_eigenvector = perturbation_eigenvector_GPM(eigenvector, self.epsilon)
                new_avg = perturbation_average_PDP(avg, self.epsilon, self.b, self.delta)

                dict_memoized[key] = (new_avg, new_eigenvector)
                results.append((new_avg, new_eigenvector))
            else:
                results.append(dict_memoized[key])

        return results

    def array_to_string(self, arr):
        return ','.join([str(val) for val in arr])

