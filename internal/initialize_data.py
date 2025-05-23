import pandas as pd
import numpy as np

def read_dataset(filename, get_evolution, limited_number = None):
    print('Reading dataset ...')
    data = []

    file = open(filename,'r')
    file.readline() # just skip headline

    while True:
        line = file.readline()
        if not line:
            break
        line = line.strip()
        line_splited = line.split(",")
        data.append([int(x) for x in line_splited])

    file.close()

    if limited_number is None:
        limited_number = len(data)

    if (not get_evolution):
        return data[:limited_number], []
    
    ###
    # append each row of sync.csv to end of each row of dataset as an array
    print('Append evolution data ...')
    df = pd.read_csv('dataset/Syn.csv')
    df.columns = range(df.shape[1])
    evolution_dataset = []
    for index, _ in enumerate(data):
        evolution_row = np.array(df.iloc[index%df.shape[0]])
        evolution_dataset.append(list(evolution_row))

    return data[:limited_number], evolution_dataset[:limited_number]

def attributes_domain(filename):
    print('Reading domains ...')
    domains = []
    
    file = open(filename,'r')
    file.readline() # just skip headline

    while True:
        line = file.readline()
        if not line:
            break
        line = line.strip()
        line_splited = line.split(" ")
        domains.append([int(x) for x in line_splited[3:]])

    file.close()
    return domains
