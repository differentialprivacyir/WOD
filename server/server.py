from internal.haar_transform import *
from internal.normalizer import *
from internal.evaluation import *

class Server:
    def __init__(self, domains):
        self.dimensions_length = len(domains)
        self.domains = domains

    def received_avg_eigenvector(self, perturbed_data):
        """
        This function receive all properties of one client.
        perturbed_data are list of (avg, eigenvector) with size tau.

        Parameters:
            perturbed_data (list): list of (avg, eigenvector)

        Returns:
            list: list of rows (tau * (d+1))
        """

        # inverse haar transform
        haar_transform_obj= HaarTransform()

        dict_memoized = {}
        retrieval_data = []
        for avg, eigenvector in perturbed_data:
            key = self.array_to_string([*eigenvector, avg])

            if key not in dict_memoized: # If data does not memoized
                re = haar_transform_obj.inverse_transform(avg, eigenvector, self.dimensions_length)
                retrieval_data.append(re)
                dict_memoized[key] = re
            else:
                retrieval_data.append(dict_memoized[key])

        return retrieval_data

    def array_to_string(self, arr):
        return ','.join([str(val) for val in arr])
