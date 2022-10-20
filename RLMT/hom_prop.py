from dataclasses import dataclass
import argparse

import torch as th

th.set_num_threads(1)

from gym.envs.registration import register
import gym

gym.logger.set_level(40)

import utils as utils
import numpy as np
import statsmodels as sm

from agent import Agent
import settings

from typing import List, Tuple
import os
import csv
import re

import multiprocessing as mp
import ast

MAX_CPU = None


@dataclass
class EnvConfig:
    init_val: List[float]
    limits: List[Tuple[float]]


def correct_reward_list(rew_dict):
    """
    Get agents in correct order
    """

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

    my_list = list(rew_dict.keys())
    my_list.sort(key=natural_keys)

    col = []
    for value in [rew_dict[k] for k in my_list if k in rew_dict]:
        col.append(value[0])

    return col


def mt_dtr(s, s2):
    """
    'dtr' test method, i.e. statistical test of the distribution of the intra/inter hellinger distance
    """
    dist_orig, dist = [], []
    bin_edges = np.histogram_bin_edges(np.concatenate((s, s2)), bins='auto')

    for _ in range(2000):
        ind = np.arange(len(s))
        ind_unknown = np.random.choice(ind, size=len(s) // 2, replace=False)
        ind_choice = list(set(ind) - set(ind_unknown))

        acc_choice2 = np.array(s)[ind_unknown]
        acc_choice = np.array(s)[ind_choice]

        hist, _ = np.histogram(acc_choice, density=True, bins=bin_edges)
        hist_mut, _ = np.histogram(acc_choice2, density=True, bins=bin_edges)

        dist_orig.append(utils.hellinger_distance(hist, hist_mut))

    for _ in range(2000):
        acc_choice2 = np.random.choice(s2, size=len(s) // 2, replace=False)
        acc_choice = np.random.choice(s, size=len(s) // 2, replace=False)

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
            return True

        if power < 0.8:
            return False
        else:
            if p_value < 0.05 and effect_size >= 0.5:
                return True
            else:
                return False
    else:
        return False


def mt_rew(s, s2):
    """
    'r' test method, i.e. statistical test of the distribution of rewards
    """
    if s != s2:
        p_value = utils.p_value_glm(s, s2)
        effect_size = utils.cohen_d(s, s2)
        power = utils.power(s, s2)

        if power < 0.8:
            return False
        else:
            if p_value < 0.05 and effect_size >= 0.5:
                return True
            else:
                return False
    else:
        return False


def hom_check_property(dict_mut: dict, list_hom: list, current_mut_name: str):
    """
    Checking the type of HOM we have.

    @param dict_mut: Dictionary of whether a given test environment kills the our FOM or not. Extracted from .csv file
    @param list_hom: Test environments killed by the HOM
    @param current_mut_name: Name of the HOM to extract which FOM is concerned
    @return:
    """
    foms = current_mut_name.split('-')
    fom_1 = dict_mut[foms[0]]
    fom_2 = dict_mut[foms[1]]

    # Subsuming if the number of test killing the HOM is inferior to the union of tests killing the FOM AND number of tests killing the HOM is not 0 
    # NOTE: We also add the conditions that the HOM might be subsuming if the number of test killing the HOM is equal to the union of tests killing the FOM
    # AND the intersection of FOM is equal to its Union. In that case, it's possible that follwoing Jia et al. definition, we have a HOM that kills exactly
    # the same test as the FOM, so is included in the intersection. Since Jia et al. definition states that a HOM is strongly subsuming if "if a test case kills a 
    # strongly subsuming HOM, it guarantees that its constituent FOMs are killed as well", the inclusion seems not to be strict, hence the case
    if (np.sum(list_hom) < np.sum(fom_1 + fom_2) or (np.sum(list_hom) == np.sum(fom_1 + fom_2)) and np.array_equal(
            fom_1 + fom_2, fom_1 * fom_2)) and np.sum(list_hom) > 0:
        # Testing if the intersection between tests killing the HOM and the union of tests killing the FOM is empty or not
        if np.sum((fom_1 + fom_2) * list_hom) == 0:
            return 'Weakly Subsuming and Decoupled'
        else:
            # Testing if the tests subset killing the HOM is included in the intersection of tests killing the FOM
            if sum([x == True and y == False for (x, y) in zip(list_hom, (fom_1 * fom_2))]) > 0:
                return 'Weakly Subsuming and Coupled'
            else:
                return 'Strongly Subsuming and Coupled'
    # Not subsuming, discarded
    else:
        return 'Non-subsuming'


if __name__ == '__main__':

    my_parser = argparse.ArgumentParser()
    my_parser.add_argument("-e", "--environment_name", help="Environment's name", required=False, default="CartPole-v1")
    my_parser.add_argument("-a", "--algorithm", help="DRL algorithm to use", required=False, default="ppo")
    my_parser.add_argument("-t", "--test_mode", help="Test mode. 'r' for reward based or 'dtr' for distance to reward",
                           required=False, default="r")
    my_parser.add_argument("-n", "--ncpus",
                           help="Number of cpus to run agents on. GPU is not handle for now. Default is number of cpus available - 1",
                           required=False, default=mp.cpu_count() - 1)
    args = my_parser.parse_args()

    assert args.environment_name in settings.CUSTOM_ENV_SETTINGS, "Unknown environment name {}".format(
        args.environment_name)

    MAX_CPU = args.ncpus

    file_name = "mut_{}_{}".format(args.algorithm,
                                   args.environment_name) if args.test_mode == 'r' else "mut_{}_{}_dtr".format(
        args.algorithm, args.environment_name)

    # Loading test environments parameters + which test environment killed which FOM
    with open(os.path.join('fom_test_env_killed', '{}.csv'.format(file_name)), mode='r') as infile:
        reader = csv.reader(infile, delimiter=";")
        csv_dict = {}
        for ind, rows in enumerate(reader):
            if ind != 0:
                csv_dict.update({rows[0]: np.array([True if x == 'True' else False for x in rows[1:-1]])})
            else:
                csv_dict.update({"Env": rows[1:-1]})
    # Test environments are as stings in the .csv so convert them to list of float
    test_env = [ast.literal_eval(x) for x in csv_dict['Env']]

    # Healthy Agents creation
    path = os.path.join('..', "experiments", 'Healthy_Agents', args.environment_name, args.algorithm.upper(), "logs")
    mut_mode = 'Healthy'
    h_agents = [
        Agent(
            args.algorithm, {"n_episodes": 10}, args.environment_name, i, False, os.path.join(
                path, f"{mut_mode}_{args.algorithm}_{args.environment_name}_{i}", "best_model"
            )
        )
        for i in range(20)
    ]
    # Getting the HOM list
    considered_mutation = [f for f in os.listdir(
        os.path.join('..', 'experiments', 'Mutated_Agents', 'HighOrderMutation', args.environment_name, args.algorithm))
                           if os.path.isdir(
            os.path.join('..', 'experiments', 'Mutated_Agents', 'HighOrderMutation', args.environment_name,
                         args.algorithm, f))]

    file_name = 'hom_{}_{}{}.csv'.format(args.algorithm, args.environment_name,
                                         '_dtr' if args.test_mode == 'dtr' else '')
    with open(file_name, 'w') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=';')
        fields = ['Mutation'] + ['{}'.format(p) for p in test_env] + ['Total'] + ['Properties']
        csv_writer.writerow(fields)

    for k, mut in enumerate(considered_mutation):
        print("Evaluating mutations... {}/{}".format(k + 1, len(considered_mutation)))
        # Load mutated agents
        path = os.path.join('..', "experiments", 'Mutated_Agents', 'HighOrderMutation', args.environment_name,
                            args.algorithm.upper(), mut, "logs")
        m_agents = [Agent(
            args.algorithm, {"n_episodes": 10}, args.environment_name, i, False, os.path.join(
                path, f"{mut}_{args.algorithm}_{args.environment_name}_{i}", "best_model"
            )
        )
            for i in range(20)
        ]

        list_mut = []

        for ind, p in enumerate(test_env):
            print("\nTesting {}/{} environment, parameters {}".format(ind + 1, len(test_env), p))

            # Registering custom env
            register(
                # unique identifier for the env `name-version`
                id="Custom{}".format(args.environment_name),
                # path to the class for creating the env
                # Note: entry_point also accept a class as input (and not only a string)
                entry_point="custom_test_env:Custom{}".format(args.environment_name.split('-')[0]),
                # Max number of steps per episode, using a `TimeLimitWrapper`
                max_episode_steps=settings.CUSTOM_ENV_SETTINGS[args.environment_name]['max_episode_steps'],
                reward_threshold=settings.CUSTOM_ENV_SETTINGS[args.environment_name]['reward_threshold'],
                kwargs={'params': p}
            )

            for i in range(len(h_agents)):
                h_agents[i].environment = "Custom{}".format(args.environment_name)
                h_agents[i].init_agent(test=True)

            for i in range(len(m_agents)):
                m_agents[i].environment = "Custom{}".format(args.environment_name)
                m_agents[i].init_agent(test=True)

            manager = mp.Manager()
            return_dict = manager.dict()

            with mp.Pool(processes=MAX_CPU):
                processes = [
                    mp.Process(target=h_agents[i].test, args=(return_dict,))
                    for i in range(len(h_agents))
                ]
                for process in processes:
                    process.start()
                for process in processes:
                    process.join()

            acc_choice = correct_reward_list(return_dict)

            manager = mp.Manager()
            m_dict_temp = manager.dict()

            with mp.Pool(processes=MAX_CPU):
                processes = [
                    mp.Process(target=m_agents[i].test, args=(m_dict_temp,))
                    for i in range(len(m_agents))
                ]

                for process in processes:
                    process.start()

                for process in processes:
                    process.join()

            acc_choice2 = correct_reward_list(m_dict_temp)

            if args.test_mode == 'r':
                list_mut.append(mt_rew(acc_choice, acc_choice2))
            else:
                list_mut.append(mt_dtr(acc_choice, acc_choice2))

        # Writing results
        file_name = 'hom_{}_{}{}.csv'.format(args.algorithm, args.environment_name,
                                             '_dtr' if args.test_mode == 'dtr' else '')
        with open(file_name, 'a') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=';')
            row = [mut] + [l for l in list_mut] + [sum(list_mut)] + [hom_check_property(csv_dict, list_mut, mut)]
            csv_writer.writerow(row)
