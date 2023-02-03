from dataclasses import dataclass
import argparse

import torch as th

th.set_num_threads(1)

from gym.envs.registration import register
import gym

gym.logger.set_level(40)

import utils
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
    Get agents in correct order since we are multiprocessing
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

        # bin_edges = np.histogram_bin_edges(np.concatenate((acc_choice, acc_choice2)), bins='auto')
        hist, _ = np.histogram(acc_choice, density=True, bins=bin_edges)
        hist_mut, _ = np.histogram(acc_choice2, density=True, bins=bin_edges)

        dist_orig.append(utils.hellinger_distance(hist, hist_mut))

    for _ in range(2000):
        acc_choice2 = np.random.choice(s2, size=len(s) // 2, replace=False)
        acc_choice = np.random.choice(s, size=len(s) // 2, replace=False)

        # bin_edges = np.histogram_bin_edges(np.concatenate((acc_choice, acc_choice2)), bins='auto')
        hist, _ = np.histogram(acc_choice, density=True, bins=bin_edges)
        hist_mut, _ = np.histogram(acc_choice2, density=True, bins=bin_edges)

        dist.append(utils.hellinger_distance(hist, hist_mut))

    if dist != dist_orig:
        try:
            p_value = utils.p_value_glm(dist_orig, dist)
            effect_size = utils.cohen_d(dist_orig, dist)
            power = utils.power(dist_orig, dist)
            # print(p_value, effect_size, power)

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


def test_val(agents: List[Agent], comp_list: List[float], current_val: List[float], environment_name: str,
             test_mode: str):
    """
    Function to test healthy agents both on initial env and candidate env using distribution wise test.

    @param agents: Healthy agents list
    @param comp_list: Reward list on initial environment extracted from .csv
    @param current_val: Parameters of the candidate custom environment
    @param environment_name: Name of the environment (e.g. CartPole-v1)
    @param test_mode: Either 'r' or 'dtr'
    @return: Boolean of the test result
    """

    # Registering custom env
    register(
        # unique identifier for the env `name-version`
        id="Custom{}".format(environment_name),
        # path to the class for creating the env
        # Note: entry_point also accept a class as input (and not only a string)
        entry_point="custom_test_env:Custom{}".format(environment_name.split('-')[0]),
        # Max number of steps per episode, using a `TimeLimitWrapper`
        max_episode_steps=settings.CUSTOM_ENV_SETTINGS[environment_name]['max_episode_steps'],
        reward_threshold=settings.CUSTOM_ENV_SETTINGS[environment_name]['reward_threshold'],
        kwargs={'params': current_val}
    )

    for i in range(len(agents)):
        agents[i].environment = "Custom{}".format(environment_name)
        agents[i].init_env(test=True)

    manager = mp.Manager()
    return_dict = manager.dict()

    with mp.Pool(processes=MAX_CPU):
        processes = [
            mp.Process(target=agents[i].test, args=(return_dict,))
            for i in range(len(agents))
        ]

        for process in processes:
            process.start()

        for process in processes:
            process.join()

    return mt_dtr(comp_list, correct_reward_list(return_dict)) if test_mode == 'dtr' else mt_rew(comp_list,
                                                                                                 correct_reward_list(
                                                                                                     return_dict))


def determine_boundaries(init_env: EnvConfig, agents: List[Agent], comp_list: List[float], precision: List[float],
                         environment_name: str, test_mode: str):
    """
    Function to generate test environments based on the healthy agents rewards on both initial and candidate test environments.
    Uses binary search.

    @param init_env: Initial environment parameters
    @param agents: Healthy agents list
    @param comp_list: Reward list on initial environment extracted from .csv
    @param precision: Precision for the binary search stop criteria
    @param environment_name: Name of the environment (e.g. CartPole-v1)
    @param test_mode: Either 'r' or 'dtr'
    @return: List of test environments as a list of list of float parameters
    """
    # Checking no conflicts on limits
    for ind in range(len(init_env.init_val)):
        assert init_env.limits[ind][1] >= init_env.init_val[ind] >= init_env.limits[ind][
            0], 'Initial value of parameter n°{} is outside of search limits'.format(ind)

    boundaries = []
    print("Searching boundaries on axis...")
    for ind in range(len(init_env.init_val)):

        print("Checking parameter n°{}".format(ind))
        print("Initial value is {}\n".format(init_env.init_val[ind]))
        b_up = [ele for ele in init_env.init_val]
        b_low = [ele for ele in init_env.init_val]

        print("Upper bound...")
        # Checking best value between initial parameter and upper limit.
        # If initial == upper, then boundary will be initial
        # Else binary search algorithm
        if init_env.init_val[ind] != init_env.limits[ind][1]:
            curr_low = np.array(init_env.init_val)
            curr_up = np.array(
                [init_env.init_val[k] if k != ind else init_env.limits[ind][1] for k in range(len(init_env.init_val))])

            # If upper limit is acceptable, than stop
            if not test_val(agents, comp_list, curr_up, environment_name, test_mode):
                b_up[ind] = init_env.limits[ind][1]
            # Otherwise, search
            else:
                while abs((curr_up - curr_low)[ind]) > precision[ind]:
                    m = (curr_up + curr_low) / 2
                    print("\nUpdating current value to {}".format(m[ind]))
                    print("Test configuration: ", m)
                    if not test_val(agents, comp_list, m, environment_name, test_mode):
                        curr_low = m
                    else:
                        curr_up = m
                # At last iteration, the low is OK
                b_up[ind] = curr_low[ind]

        print("Found: {}".format(b_up[ind]))

        print("\nLower bound...")
        # Same but for lower bound
        if init_env.init_val[ind] != init_env.limits[ind][0]:
            curr_low = np.array(init_env.init_val)
            curr_up = np.array(
                [init_env.init_val[k] if k != ind else init_env.limits[ind][0] for k in range(len(init_env.init_val))])
            # If lower limit is acceptable, than stop
            if not test_val(agents, comp_list, curr_up, environment_name, test_mode):
                b_low[ind] = init_env.limits[ind][0]
            # Otherwise, search
            else:
                while abs((curr_up - curr_low)[ind]) > precision[ind]:
                    m = (curr_up + curr_low) / 2
                    print("\nUpdating current value to {}".format(m[ind]))
                    print("Test configuration: ", m)
                    if not test_val(agents, comp_list, m, environment_name, test_mode):
                        curr_low = m
                    else:
                        curr_up = m
                # At last iteration, the low is OK
                b_low[ind] = curr_low[ind]

        print("Found: {}".format(b_low[ind]))

        print("Bound axis values are [{}, {}]\n".format(b_low[ind], b_up[ind]))
        boundaries.append([b_low, b_up])

    print("Searching mid points boundaries...")
    mid_bounds = []
    for i in range(len(boundaries) - 1):
        for j in range(i + 1, len(boundaries)):
            for p in range(2):
                for l in range(2):
                    if boundaries[i][p] != init_env.init_val and boundaries[j][l] != init_env.init_val:
                        curr_low = np.array(init_env.init_val)
                        curr_up = np.array([boundaries[i][p][k] if k != j else boundaries[j][l][k] for k in
                                            range(len(init_env.init_val))])
                        print("\nSearching middle points between {} and {}".format(curr_low, curr_up))
                        if not test_val(agents, comp_list, curr_up, environment_name, test_mode):
                            mid_bounds.append(list(curr_up))
                        else:
                            while np.linalg.norm(curr_up - curr_low) > np.linalg.norm(precision):
                                m = (curr_up + curr_low) / 2
                                print("\nUpdating current points to {}".format(m))
                                print("Test configuration: ", m)
                                if not test_val(agents, comp_list, m, environment_name, test_mode):
                                    curr_low = m
                                else:
                                    curr_up = m
                            mid_bounds.append(list(curr_low))

    mid_bounds = [mid_bounds]
    bounds = [list(x) for x in set(tuple(x) for x in [item for sublist in boundaries + mid_bounds for item in sublist])]

    if init_env.init_val not in bounds:
        bounds.append(init_env.init_val)

    return bounds


if __name__ == '__main__':

    my_parser = argparse.ArgumentParser()
    my_parser.add_argument("-e", "--environment_name", help="Environment's name", required=False, default="CartPole-v1")
    my_parser.add_argument("-a", "--algorithm", help="DRL algorithm to use", required=False, default="ppo")
    my_parser.add_argument('-i', '--init_val', type=float, nargs='+',
                           help='Initial environment parameters values to consider. Ex: "-i 1.0 0.1', required=True)
    my_parser.add_argument('-l', '--limits', type=float, nargs='+',
                           help='Upper/Lower bound for each parameters for the search. Synthax is "-l param1_lower param1_upper param2_lower .... Ex for two parameters: "-l 1.0 20.0 0.1 20.0',
                           required=True)
    my_parser.add_argument("-b", "--bounds", type=str,
                           help="Optional. If bounds are provided as a string of a list of list, then search is bypassed.",
                           required=False, default=None)
    my_parser.add_argument("-t", "--test_mode", help="Test mode. 'r' for reward based or 'dtr' for distance to reward",
                           required=False, default="r")
    my_parser.add_argument("-n", "--ncpus",
                           help="Number of cpus to run agents on. GPU is not handle for now. Default is number of cpus available - 1",
                           required=False, default=mp.cpu_count() - 1)
    args = my_parser.parse_args()

    assert 2 * len(args.init_val) == len(
        args.limits), "'--limits' argument should have 2 * number of parameters in '--init_val' argument"
    assert args.environment_name in settings.CUSTOM_ENV_SETTINGS, "Unknown environment name {}".format(
        args.environment_name)

    lim = [args.limits[i:i + 2] for i in range(0, len(args.limits), 2)]
    init_env = EnvConfig(init_val=args.init_val, limits=lim)
    MAX_CPU = args.ncpus

    if args.bounds is None:
        # Path to healthy agents weights
        path = os.path.join('..', "experiments", 'Healthy_Agents', args.environment_name, args.algorithm.upper(),
                            "logs")
        mut_mode = 'Healthy'

        # Healthy agents initialization for the search
        agents = [
            Agent(
                args.algorithm, {"n_episodes": 10}, args.environment_name, i, False, os.path.join(
                    path, f"{mut_mode}_{args.algorithm}_{args.environment_name}_{i}", "best_model"
                )
            )
            for i in range(20)
        ]

        for agent in agents:
            agent.init_agent(test=True)

        with open(
                os.path.join('..', 'csv_res', args.environment_name, args.algorithm, 'results_{}.csv'.format(mut_mode)),
                'r') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=';')
            for ind, row in enumerate(csv_reader):
                if ind != 0:
                    comp_list = [float(r) for r in row[1:21]]

        # Determining the generating test environments (Section III of our paper)
        bounds = determine_boundaries(init_env=init_env, agents=agents, comp_list=comp_list,
                                      precision=[0.5, 0.05],
                                      environment_name=args.environment_name, test_mode=args.test_mode)
    else:
        bounds = ast.literal_eval(args.bounds)

    print("Environment to consider: ", bounds)

    # Creating Healthy Agents
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

    # Mutations to consider are all the mutations present in the csv_res/ directory
    considered_mutation = [f for f in os.listdir(os.path.join('..', 'csv_res', args.environment_name, args.algorithm))
                           if 'Healthy' not in f]
    m_agents = []
    m_list = []
    mut_dict = dict(settings.MUTATION_AGENT_LIST)
    mut_dict.update(settings.MUTATION_ENVIRONMENT_LIST)

    for mut in considered_mutation:
        res = re.match(r"(results_)(.*)(\.csv)", mut)
        mut_mode = res.groups()[1]
        sp = mut_mode.split('_')
        path = os.path.join('..', "experiments", 'Mutated_Agents', 'SingleOrderMutation',
                            list(mut_dict.keys())[list(mut_dict.values()).index(sp[0])], args.environment_name,
                            args.algorithm.upper())
        # If mutations have an argument such as PAC_ReLU or R_1.0
        if len(sp) == 2:
            path = os.path.join(path, sp[1])
        path = os.path.join(path, "logs")

        # Creating mutated agents for mutation "mut"
        m_agents.append([Agent(
            args.algorithm, {"n_episodes": 10}, args.environment_name, i, False, os.path.join(
                path, f"{mut_mode}_{args.algorithm}_{args.environment_name}_{i}", "best_model"
            )
        )
            for i in range(20)
        ])
        m_list.append(mut_mode)

    # File to write results
    file_name = 'mut_{}_{}{}.csv'.format(args.algorithm, args.environment_name,
                                         '_dtr' if args.test_mode == 'dtr' else '')
    with open(file_name, 'w') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=';')
        fields = ['Mutation'] + ['{}'.format(p) for p in bounds] + ['Total']
        csv_writer.writerow(fields)
        dict_mut = {}
        for ind, p in enumerate(bounds):
            print("\nTesting {}/{} environment, parameters {}".format(ind + 1, len(bounds), p))

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
                # Custom parameters
                kwargs={'params': p}
            )

            # Resetting correct custom environment
            for i in range(len(h_agents)):
                h_agents[i].environment = "Custom{}".format(args.environment_name)
                h_agents[i].init_agent(test=True)

            for m_ag in m_agents:
                for i in range(len(m_ag)):
                    m_ag[i].environment = "Custom{}".format(args.environment_name)
                    m_ag[i].init_agent(test=True)

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

            # Reordering the list as multiprocessing might mess up the order
            acc_choice = correct_reward_list(return_dict)

            for k in range(len(m_agents)):

                print("Evaluating mutations... {}/{}".format(k + 1, len(m_agents)))

                manager = mp.Manager()
                m_dict_temp = manager.dict()

                with mp.Pool(processes=MAX_CPU):
                    processes = [
                        mp.Process(target=m_agents[k][i].test, args=(m_dict_temp,))
                        for i in range(len(m_agents[k]))
                    ]

                    for process in processes:
                        process.start()

                    for process in processes:
                        process.join()

                acc_choice2 = correct_reward_list(m_dict_temp)

                # Evaluating with either 'r' or 'dtr' method
                if args.test_mode == 'dtr':
                    if m_list[k] in dict_mut:
                        dict_mut[m_list[k]].append(mt_dtr(acc_choice, acc_choice2))
                    else:
                        dict_mut[m_list[k]] = [mt_dtr(acc_choice, acc_choice2)]
                else:
                    if m_list[k] in dict_mut:
                        dict_mut[m_list[k]].append(mt_rew(acc_choice, acc_choice2))
                    else:
                        dict_mut[m_list[k]] = [mt_rew(acc_choice, acc_choice2)]

        for key in dict_mut:
            col = [key]
            col = col + dict_mut[key] + [sum(dict_mut[key])]
            csv_writer.writerow(col)
