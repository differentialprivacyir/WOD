import math
import numpy as np
from collections import deque

class HaarTransform:
    def transnform(self, row):
        row = self.convert_data_to_next_power_of_2(row)
        # row = np.array(row)
        avg = np.mean(row)
        eigenvector = self.find_eigenvector_bfs(row)
        return avg, eigenvector

    def inverse_transform(self, avg, eigenvector, length):
        adopted_length = self.next_power_of_2(length)
        row = []
        for i in range(adopted_length):
            index = i + adopted_length

            row.append(avg)
            while(index > 1):
                g = (-1) ** ((index % 2) + 2)
                index = int(index / 2)
                row[i] += g * eigenvector[index-1]
        return row[:length]

    def find_eigenvector_bfs(self, row):
        eigenvector = []
        queue = deque([row])
        while queue:
            node = queue.popleft()
            if len(node) == 1:
                continue

            left_node, right_node = self.split_row(node)
            eigenvector.append(self.calculate_root_eigenvector(left_node, right_node))
            queue.append(left_node)
            queue.append(right_node)
        return eigenvector

    def split_row(self, data):
        middle_index = int(len(data) / 2)
        left_data = data[:middle_index]
        right_data = data[middle_index:]
        return left_data, right_data

    def calculate_root_eigenvector(self, left_data, right_data):
        ml = np.mean(left_data)
        mr = np.mean(right_data)
        return (ml-mr)/2

    def next_power_of_2(self, n):
        return 2 ** math.ceil(math.log2(n))

    def convert_data_to_next_power_of_2(self, data):
        size = len(data)
        next_power_of_2 = 2 ** math.ceil(math.log2(size))
        diff = next_power_of_2 - size
        return np.pad(data, (0, diff), mode='constant')
    

def test_haar_transform():
    haar_transform_obj= HaarTransform()
    data = [9,7,3,5,8,4,5,7,1]
    print('input data:', data)
    avg, eigenvector = haar_transform_obj.transnform(data)
    retrieval_data = haar_transform_obj.inverse_transform(avg, eigenvector, len(data))
    print('retrieval data:', retrieval_data)
