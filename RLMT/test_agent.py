import agent, random, argparse
from typing import Callable
import os
import csv
from pathlib import Path
import settings
import json

import multiprocessing as mp

########################################################################################################################
# Parsing arguments
########################################################################################################################
# General arguments
parser = argparse.ArgumentParser(
    description="Mutation Testing for Deep Reinforcement Learning"
)
parser.add_argument(
    "-a", "--algorithm", help="DRL algorithm to use", required=False, default="ppo"
)
parser.add_argument(
    "-na",
    "--number_of_agents",
    help="Number of agents",
    required=False,
    default=1,
)
parser.add_argument(
    "-e",
    "--environment_name",
    help="Environment's name",
    required=False,
    default="CartPole-v1",
)
parser.add_argument(
    "-m",
    "--mutation",
    help="String encoding a dictionary with the mutations and their magnitudes (if applicable). For instance: '{\\\"missing_terminal_state\\\": \\\"None\\\", \\\"policy_optimizer_change\\\": \\\"SGD\\\"}'",
    required=False,
    default='{}'
)

arguments = parser.parse_args()

mutation_dict = json.loads(arguments.mutation)

assert arguments.algorithm.upper() in ["PPO", "A2C", "DQN"], "Algorithm {} is not a valid".format(
    arguments.algorithm.upper())

# If mutation dict is not empty, then it means we want to mutated something. Otherwise, Healthy Agent/Environment
if len(mutation_dict) == 0:
    path = os.path.join('..', "experiments", 'Healthy_Agents', arguments.environment_name, arguments.algorithm.upper(),
                        "logs")
    mut_mode = 'Healthy'
else:
    if len(mutation_dict) == 1:
        path = os.path.join('..', "experiments", 'Mutated_Agents', 'SingleOrderMutation',
                            '_'.join(mutation_dict.keys()), arguments.environment_name, arguments.algorithm.upper())
        if mutation_dict['_'.join(mutation_dict.keys())] != 'None':
            path = os.path.join(path, mutation_dict['_'.join(mutation_dict.keys())])
    else:
        path = os.path.join('..', "experiments", 'Mutated_Agents', 'HighOrderMutation', '_'.join(mutation_dict.keys()),
                            arguments.environment_name, arguments.algorithm.upper())
        extra_path = '_'
        # Combine magnitudes together, remove front '_' character at the end
        for key in mutation_dict.keys():
            if mutation_dict[key] != 'None':
                extra_path = extra_path + mutation_dict[key]
        path = os.path.join(path, extra_path[1:])

    path = os.path.join(path, "logs")
    mut_dict = dict(settings.MUTATION_AGENT_LIST)
    mut_dict.update(settings.MUTATION_ENVIRONMENT_LIST)
    mut_mode = '-'.join(map(lambda x: '_'.join(x),
                            [[mut_dict[key], mutation_dict[key]] if mutation_dict[key] != 'None' else [mut_dict[key]]
                             for key in mutation_dict.keys()]))

# Agents initialization
agents = [
    agent.Agent(
        arguments.algorithm, {"n_episodes": 10}, arguments.environment_name, i, False, os.path.join(
            path, f"{mut_mode}_{arguments.algorithm}_{arguments.environment_name}_{i}", "best_model"
        )
    )
    for i in range(int(arguments.number_of_agents))
]

for agent in agents:
    agent.init_agent(test=True)

manager = mp.Manager()
return_dict = manager.dict()

with mp.Pool(processes=mp.cpu_count() - 2):
    processes = [
        mp.Process(target=agents[i].test, args=(return_dict,))
        for i in range(len(agents))
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()


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


my_list = list(return_dict.keys())
my_list.sort(key=natural_keys)

path_dir = Path(os.path.join('..', 'csv_res', arguments.environment_name, arguments.algorithm.upper()))
path_dir.mkdir(parents=True, exist_ok=True)

fields = [' '] + ['pred_rewards_model_{}'.format(i) for i in range(int(arguments.number_of_agents))]
col = [None]

for value in [return_dict[k] for k in my_list if k in return_dict]:
    col.append(value[0])
with open(os.path.join(path_dir, 'results_{}.csv'.format(mut_mode)), 'w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=';')
    csvwriter.writerow(fields)
    csvwriter.writerow(col)
