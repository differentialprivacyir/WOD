from internal.initialize_data import *
from internal.LOLOHA import *
from client.client import *
from server.server import *
import os
import progressbar  # pip install progressbar

EPSILON = os.environ.get('EPSILON', 1)
RADNOM_SEED = os.environ.get('RADNOM_SEED', 10)
DATASET_NUMBER = os.environ.get('DATASET_NUMBER', 2)
B = 0.005
DELTA = 0.001
LIMITED_NUMBER = os.environ.get('LIM', 0)
EVOLUTION_DOMAIN_SIZE = 360  # in order to Syn.csv
ALPHA = 0.4
epsiolon1 = ALPHA * EPSILON

def main():
    ## Initialize dataset
    df = read_evolution_dataset('dataset/Syn.csv')
    dataset, evolution_dataset = read_dataset(f'dataset/Data{DATASET_NUMBER}-coarse.dat', df, limited_number=int(LIMITED_NUMBER))
    domains = attributes_domain(f'dataset/Data{DATASET_NUMBER}-coarse.domain')
    tau = len(evolution_dataset[0])
    number_of_users = len(dataset)

    print('dataset[0] is',dataset[0])
    print('evolution_dataset[0][:10] is',evolution_dataset[0][:10])
    print('tau is', tau)
    print('number of users is', number_of_users)

    ## Real frequency for each data collection $t \in [\tau]$
    dic_real_freq = compute_frequency(evolution_dataset, tau, EVOLUTION_DOMAIN_SIZE)

    ## Reduce domain size by hashing
    g = compute_optimal_domain_size(EPSILON, ALPHA)
    hashed_evolution_dataset, user_hash_functions = reduce_domain_dataset(evolution_dataset, g)

    # Revise the domains (append evolution domain)
    domains.append(list(range(g)))

    client_obj = Client(EPSILON, RADNOM_SEED, B, DELTA)
    server_obj = Server(domains)

    print_table(evolution_dataset[0][:10], hashed_evolution_dataset[0][:10], 'evolution_dataset', 'hashed_evolution_dataset')

    ## Perturbation with GRR
    perturbed_evolution_dataset = perturbation_GRR(hashed_evolution_dataset, g, EPSILON, 0.2)
    print_table(hashed_evolution_dataset[0][:10], perturbed_evolution_dataset[0][:10], 'hashed_evolution_dataset', 'perturbed_evolution_dataset')

    ## Normalize Dataset
    # normalize to [-1,1]
    normalized_dataset = normalize_dataset(dataset, domains)
    normalized_evolution_dataset = normalize_dataset(perturbed_evolution_dataset, [list(range(g)) for _ in range(tau)])

    print_table(perturbed_evolution_dataset[0][:10], normalized_evolution_dataset[0][:10], 'perturbed_evolution_dataset', 'normalized_evolution_dataset')

    ## Wheel of Differential
    print('Wheel of Differential ...')
    retrieval_dataset = []
    retrieval_evolutional_dataset = []

    for data, data_e in zip(normalized_dataset, normalized_evolution_dataset):
        perturbed_data = client_obj.send_perturbed_avg_eigenvector(data, data_e)
        retrieval_data = server_obj.received_avg_eigenvector(perturbed_data)
        retrieval_dataset.append(retrieval_data[0][:-1])
        retrieval_evolutional_dataset.append(get_coloumn_dataset(retrieval_data, -1))

    ## Evaluation
    print_table([*normalized_dataset[0], normalized_evolution_dataset[0][0]], [*retrieval_dataset[0], retrieval_evolutional_dataset[0][0]],
            'normalized data', 'retrival data')

    print('domain size of retrieval data is',len(retrieval_dataset[0]))

    # denormalizing
    denormalized = denormalize_dataset(retrieval_dataset, domains)
    denormalized_evolution_dataset = denormalize_dataset(retrieval_evolutional_dataset, [list(range(g)) for _ in range(tau)])
    rounded_evolution_dataset = round_dataset(denormalized_evolution_dataset)

    print_table(dataset[0], denormalized[0], 'original', 'retrieved')
    print_table(hashed_evolution_dataset[0][:10], rounded_evolution_dataset[0][:10], 'original evolution', 'retrieved evolution')

    print('MSE is', findMSE(normalized_dataset, retrieval_dataset))
    _, avg = average_variation_distance(dataset, denormalized)
    print('Average Variation Distance is', avg)

    ## Evaluate Frequency Estimation
    prog = progressbar.ProgressBar(maxval=tau)
    prog.start()

    dic_estimate_freq = []
    for t in range(tau):
        dic_estimate_freq.append(LOLOHA_Aggregator(get_coloumn_dataset(rounded_evolution_dataset, t), user_hash_functions, EVOLUTION_DOMAIN_SIZE, EPSILON, epsiolon1, ALPHA))
        prog.update(t) 

    prog.finish()

    print_table(dic_real_freq[0][:10], dic_estimate_freq[0][:10], 'real frequency', 'estimate frequency')

    print('MSE of frequency is', findMSE(dic_real_freq, dic_estimate_freq))


if __name__ == "__main__":
    main()
