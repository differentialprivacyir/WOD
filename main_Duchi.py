# ------------ Test Duchi Mechanism ------------------------------ #

from internal.initialize_data import *
from other_solutions.Duchi import *
from internal.normalizer import *
from internal.evaluation import *
import os
import progressbar  # pip install progressbar

EPSILON = os.environ.get('EPSILON', 1)
RADNOM_SEED = os.environ.get('RADNOM_SEED', 10)
DATASET_NUMBER = os.environ.get('DATASET_NUMBER', 2)
LIMITED_NUMBER = os.environ.get('LIM', 0)
EVOLUTION_DOMAIN_SIZE = 360  # in order to Syn.csv
ALPHA = 0.4
epsiolon1 = ALPHA * EPSILON

def main():
    ## Initialize dataset
    df = read_evolution_dataset('dataset/Syn.csv')
    dataset, _ = read_dataset(f'dataset/Data{DATASET_NUMBER}-coarse.dat', dataFrame=df, limited_number=int(LIMITED_NUMBER))
    domains = attributes_domain(f'dataset/Data{DATASET_NUMBER}-coarse.domain')
    number_of_users = len(dataset)
    duchi_obj = Duchi_Class(len(dataset[0]), EPSILON, RADNOM_SEED)

    print('dataset[0] is',dataset[0])
    print('number of users is', number_of_users)

    ## Normalize Dataset
    # normalize to [-1,1]
    normalized_dataset = normalize_dataset(dataset, domains)

    ## Perturbation
    print('Perturbation with Duchi Mechanism ...')
    retrieval_dataset = []
    for row in normalized_dataset:
        retrieval_dataset.append(duchi_obj.Duchi_multidim_ldp(row))

    print_table(normalized_dataset[0], retrieval_dataset[0],
            'Normalized Dataset', 'Perturbed Dataset')

    ## Evaluation
    # denormalizing
    denormalized = denormalize_dataset(retrieval_dataset, domains)

    print('MSE is', findMSE(normalized_dataset, retrieval_dataset))
    _, avg = average_variation_distance(dataset, denormalized)
    print('Average Variation Distance is', avg)


if __name__ == "__main__":
    main()
