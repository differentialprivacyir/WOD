from internal.initialize_data import *
from internal.normalizer import *
from internal.evaluation import *
from other_solutions.Rappor import *
import os
import progressbar  # pip install progressbar

EPSILON = float(os.environ.get('EPSILON', 1))
RADNOM_SEED = int(os.environ.get('RADNOM_SEED', 10))
DATASET_NUMBER = int(os.environ.get('DATASET_NUMBER', 2))
LIMITED_NUMBER = int(os.environ.get('LIM', 0))
LIMITED_DIMENSIONS = int(os.environ.get('LIM_DIM', 0))
LIMITED_TAU = int(os.environ.get('LIM_tAU', 0))
SILENCE = bool(os.environ.get('SILENCE', True))

EVOLUTION_DOMAIN_SIZE = 360  # in order to Syn.csv
ALPHA = 0.4
epsiolon1 = ALPHA * EPSILON

def main():
    ## Initialize dataset
    df = read_evolution_dataset('dataset/Syn.csv')
    _, evolution_dataset = read_dataset(f'dataset/Data{DATASET_NUMBER}-coarse.dat', dataFrame=df,
                                        limited_number=LIMITED_NUMBER,
                                        limited_tau=LIMITED_TAU)
    tau = len(evolution_dataset[0])
    n = len(evolution_dataset)

    print('algorithm running is Rappor')
    print('number of users is', n)
    print('evolution_dataset[0][:10] is',evolution_dataset[0][:10])
    print('tau is', tau)
    print('epsilon is', EPSILON)
    print('datset number is', DATASET_NUMBER)

    ## Real frequency for each data collection $t \in [\tau]$
    dic_real_freq = compute_frequency(evolution_dataset, tau, EVOLUTION_DOMAIN_SIZE)

    ## Perturbation with Rappor
    rappor_obj = Rappor_Class(EPSILON, epsiolon1, RADNOM_SEED)
    perturbed_evolution_dataset = []

    prog = progressbar.ProgressBar(maxval=n)
    prog.start()
    for index, user_row in enumerate(evolution_dataset):
        perturbed_row, _ = rappor_obj.RAPPOR_Client(user_row, EVOLUTION_DOMAIN_SIZE)
        perturbed_evolution_dataset.append(perturbed_row)
        prog.update(index)
    prog.finish()

    ## Evaluate Frequency Estimation
    dic_estimate_freq = []
    for t in range(tau):
        dic_estimate_freq.append(rappor_obj.RAPPOR_Aggregator(get_coloumn_dataset(perturbed_evolution_dataset, t)))

    print_table(dic_real_freq[0][:10], dic_estimate_freq[0][:10],
                'real frequency', 'estimate frequency',
                silence=SILENCE)

    print('MSE of frequency is', findMSE(dic_real_freq, dic_estimate_freq))


if __name__ == "__main__":
    main()
