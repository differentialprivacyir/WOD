import pandas as pd
import numpy as np

def read_dataset(filename, dataFrame = None, limited_number = 0):
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

    if limited_number == 0:
        limited_number = len(data)
    
    if dataFrame is None:
        return data[:limited_number], None

    ###
    # append each row of sync.csv to end of each row of dataset as an array
    print('Append evolution data ...')
    evolution_dataset = []
    for index, _ in enumerate(data):
        evolution_row = np.array(dataFrame.iloc[index%dataFrame.shape[0]])
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

def read_evolution_dataset(filename):
    df = pd.read_csv(filename)
    df.columns = range(df.shape[1])
    
    return df
