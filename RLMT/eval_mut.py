import argparse
import os.path
import warnings
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from tqdm import tqdm

import settings
import utils as utils
import statsmodels as sm

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument("-e", "--environment_name", help="Environment's name", required=False, default="CartPole-v1")
    my_parser.add_argument("-a", "--algorithm", help="DRL algorithm to use", required=False, default="ppo")
    my_parser.add_argument("-m", "--mutation",
                           help="Code string encoding the mutations and their magnitudes (if applicable). For instance: ILF. Default is 'Healthy' (no mutation)",
                           required=True)
    args = my_parser.parse_args()

    # Params
    model = args.algorithm
    mut_name = args.mutation
    environment = args.environment_name

    N = 2000
    n = 20

    print('Environement: {}, Model {}, Mutation {}'.format(environment, model, mut_name))
    path = os.path.join('..', 'csv_res', environment, model.upper())

    # Data loading
    dat = pd.read_csv(os.path.join(path, 'results_Healthy.csv'), sep=';')
    dat_mut = pd.read_csv(os.path.join(path, 'results_{}.csv'.format(mut_name)), sep=';')


    # Source: https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
    def atoi(text):
        return int(text) if text.isdigit() else text


    def natural_keys(text):
        import re
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [atoi(c) for c in re.split(r'(\d+)$', text)]


    assert mut_name != 'Healthy', "Mutation can not be Healthy agent"


    def anom_res(s, s2):
        dist_orig, dist = [], []
        bin_edges = np.histogram_bin_edges(
            np.concatenate((list(dat[s].to_numpy()[0]), list(dat_mut[s2].to_numpy()[0]))), bins='auto')

        for i in range(N):
            pop_unknown = np.random.choice(s, size=n // 2, replace=False)
            pop_sound = list(set(s) - set(pop_unknown))

            acc_choice2 = dat[pop_unknown].to_numpy()[0]
            acc_choice = dat[pop_sound].to_numpy()[0]

            hist, _ = np.histogram(acc_choice, density=True, bins=bin_edges)
            hist_mut, _ = np.histogram(acc_choice2, density=True, bins=bin_edges)

            dist_orig.append(utils.hellinger_distance(hist, hist_mut))

        for i in range(N):
            pop_unknown = np.random.choice(s2, size=n // 2, replace=False)
            pop_sound = np.random.choice(s, size=n // 2, replace=False)

            acc_choice2 = dat_mut[pop_unknown].to_numpy()[0]
            acc_choice = dat[pop_sound].to_numpy()[0]

            hist, _ = np.histogram(acc_choice, density=True, bins=bin_edges)
            hist_mut, _ = np.histogram(acc_choice2, density=True, bins=bin_edges)

            dist.append(utils.hellinger_distance(hist, hist_mut))

        if dist != dist_orig:
            try:
                p_value = utils.p_value_glm(dist_orig, dist)
                effect_size = utils.cohen_d(dist_orig, dist)
                power = utils.power(dist_orig, dist)

            # When predictors can be perfectly distinguished based on one variable
            # Means there is no doubt what is what, so killed
            except sm.tools.sm_exceptions.PerfectSeparationError:
                return 'Killed'

            if power < 0.8:
                return 'Inconclusive'
            else:
                if p_value < 0.05 and effect_size >= 0.5:
                    return 'Killed'
                else:
                    return 'Not Killed'
        else:
            return 'Not Killed'


    def dc_res(s, s2):
        acc_choice = list(dat[s].to_numpy()[0])
        acc_choice2 = list(dat_mut[s2].to_numpy()[0])

        if acc_choice != acc_choice2:
            p_value = utils.p_value_glm(acc_choice, acc_choice2)
            effect_size = utils.cohen_d(acc_choice, acc_choice2)
            power = utils.power(acc_choice, acc_choice2)

            if power < 0.8:
                return 'Inconclusive'
            else:
                if p_value < 0.05 and effect_size >= 0.5:
                    return 'Killed'
                else:
                    return 'Not Killed'
        else:
            return 'Not Killed'


    p = dc_res(dat.columns[1:21], dat_mut.columns[1:21])
    p2 = anom_res(dat.columns[1:21], dat_mut.columns[1:21])

    p3 = 0
    for (a, b) in zip(dat.to_numpy()[0][1:21], dat_mut.to_numpy()[0][1:21]):
        if b / a < 0.9:
            p3 += 1

    print(f"Average Mutated/Healthy Ratio Test : {p3 / 20}")
    print(f"Reward Distribution Test : {p}")
    print(f"Distance Distribution Test: {p2}")
    print("\n")
