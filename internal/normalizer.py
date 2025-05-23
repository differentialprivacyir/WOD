import numpy as np

def normalize(x, domain):
    max_domain = np.max(domain)
    min_domain = np.min(domain)
    return ((2*(x-min_domain)) / (max_domain-min_domain)) - 1


def denormalize(y, domain):
    max_domain = np.max(domain)
    min_domain = np.min(domain)
    return (((max_domain - min_domain) * (y + 1)) / 2) + min_domain


def normalize_dataset(dataset, domains):
    """
    Normailzing dataset in order to minMAX algorithm
    
    Parameters:
        dataset (list of rows)
        domains (list of rows)
    
    Returns:
        list of rows: normalized dataset
    """

    print("Normalizing dataset to [-1,1]")
    normalized_data = []
    for row in dataset:
        normalized_row = [normalize(element, domains[index]) for index, element in enumerate(row)]
        normalized_data.append(normalized_row)

    return normalized_data

def denormalize_dataset(normalized_dataset, domains):
    """
    Normailzing dataset in order to minMAX algorithm
    
    Parameters:
        normalized_dataset (list of rows)
        domains (list of rows)
    
    Returns:
        list of rows: original dataset
    """

    print("Denormaizing dataset ...")
    original_dataset = []
    for row in normalized_dataset:
        original_row = [denormalize(element, domains[index]) for index, element in enumerate(row)]
        original_dataset.append(original_row)

    return original_dataset
