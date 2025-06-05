from internal.initialize_data import *
from internal.normalizer import *
from internal.evaluation import *
from other_solutions.dBitFlipPM import *
import os
import progressbar  # pip install progressbar

EPSILON = os.environ.get('EPSILON', 1)
RADNOM_SEED = os.environ.get('RADNOM_SEED', 10)
DATASET_NUMBER = os.environ.get('DATASET_NUMBER', 2)
LIMITED_NUMBER = os.environ.get('LIM', 10000)
EVOLUTION_DOMAIN_SIZE = 360  # in order to Syn.csv
ALPHA = 0.4
epsiolon1 = ALPHA * EPSILON

def main():
    ## Initialize dataset
    df = read_evolution_dataset('dataset/Syn.csv')
    _, evolution_dataset = read_dataset(f'dataset/Data{DATASET_NUMBER}-coarse.dat', dataFrame=df, limited_number=int(LIMITED_NUMBER))
    tau = len(evolution_dataset[0])
    n = len(evolution_dataset)
    b = EVOLUTION_DOMAIN_SIZE  # number of buckets
    d = EVOLUTION_DOMAIN_SIZE  # number of bits each user sample/report

    print('number of users is', n)
    print('evolution_dataset[0][:10] is',evolution_dataset[0][:10])
    print('tau is', tau)
    print('number of buckets is', b)

    ## Real frequency for each data collection $t \in [\tau]$
    dic_real_freq = compute_frequency(evolution_dataset, tau, EVOLUTION_DOMAIN_SIZE)

    ## Perturbation with Rappor
    dBitFlipPM_obj = BitFlipPM_Class(b, d, EPSILON, RADNOM_SEED)
    perturbed_evolution_dataset = []

    prog = progressbar.ProgressBar(maxval=n)
    prog.start()
    for index, user_row in enumerate(evolution_dataset):
        perturbed_row, _, _ = dBitFlipPM_obj.dBitFlipPM_Client(user_row, EVOLUTION_DOMAIN_SIZE)
        perturbed_evolution_dataset.append(perturbed_row)
        prog.update(index)
    prog.finish()

    ## Evaluate Frequency Estimation
    dic_estimate_freq = []
    for t in range(tau):
        dic_estimate_freq.append(dBitFlipPM_obj.dBitFlipPM_Aggregator(get_coloumn_dataset(perturbed_evolution_dataset, t)))

    print_table(dic_real_freq[0][:10], dic_estimate_freq[0][:10], 'real frequency', 'estimate frequency')

    print('MSE of frequency is', findMSE(dic_real_freq, dic_estimate_freq))


if __name__ == "__main__":
    main()
