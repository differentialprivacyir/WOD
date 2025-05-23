from internal.haar_transform import *
from internal.normalizer import *
from internal.evaluation import *

class Server:
    def __init__(self, domains):
        self.dimensions_length = len(domains)
        self.domains = domains

    def received_avg_eigenvector(self, avg, eigenvector):
        # inverse haar transform
        haar_transform_obj= HaarTransform()
        retrieval_data = haar_transform_obj.inverse_transform(avg, eigenvector, self.dimensions_length)

        return retrieval_data
