import agent, random, argparse, json, os
import multiprocessing  # as mp
import multiprocessing.pool  # as p

import torch as th

th.set_num_threads(1)

from torch.utils import tensorboard

import json

import settings
from logger import Log

from typing import Callable


# needed for multiprocessing
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
        current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


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
    "-t", "--total_steps", help="Total time steps", required=False, default=int(1e5)
)
parser.add_argument(
    "-s",
    "--start_from",
    help="Agents' ID counter start point",
    required=False,
    default=int(0),
)

parser.add_argument(
    "-m",
    "--mutation",
    help="String encoding a dictionary with the mutations and their magnitudes (if applicable). For instance: '{\\\"missing_terminal_state\\\": \\\"None\\\", \\\"policy_optimizer_change\\\": \\\"SGD\\\"}'",
    required=False,
    default='{}'
)

parser.add_argument(
    "-v",
    "--vectorized",
    help="Vectorized version customized to work with multiprocessing. Experimental",
    default=False,
    action="store_true"
)

arguments = parser.parse_args()

mutation_dict = json.loads(arguments.mutation)
logger = Log("logger")

# Agents initialization
agents = []

assert arguments.algorithm.upper() in ["PPO", "A2C", "DQN"], "Algorithm {} is not a valid".format(
    arguments.algorithm.upper())

logger.info(f"Starting training agents using {arguments.algorithm} on environment {arguments.environment_name}")
if len(mutation_dict) != 0:
    logger.info("Mutations used:")
    for (key, val) in mutation_dict.items():
        if val != 'None':
            sent = f'{key} - {val}'
        else:
            sent = f'{key}'
        logger.info(sent)

import time

start_time = time.time()


# Worker function
def run_agent(i):
    # If mutation dict is not empty, then it means we want to mutated something. Otherwise, Healthy Agent/Environment
    if len(mutation_dict) == 0:
        settings.HYPER_PARAMS[arguments.environment_name][arguments.algorithm]['seed'] = int(arguments.start_from) + i
        agent_ = agent.Agent(
            algorithm=arguments.algorithm,
            hyper_params=settings.HYPER_PARAMS[arguments.environment_name][arguments.algorithm],
            environment=arguments.environment_name,
            agent_number=int(arguments.start_from) + i,
            vectorized=arguments.vectorized,
        )
    else:
        settings.HYPER_PARAMS[arguments.environment_name][arguments.algorithm]['seed'] = int(arguments.start_from) + i
        agent_ = agent.MutatedAgent(
            algorithm=arguments.algorithm,
            hyper_params=settings.HYPER_PARAMS[arguments.environment_name][arguments.algorithm],
            environment=arguments.environment_name,
            mutation=mutation_dict,
            agent_number=int(arguments.start_from) + i,
            vectorized=arguments.vectorized,
        )

    agent_.init_agent(test=False)
    return agent_.train(int(arguments.total_steps))


if arguments.vectorized:
    logger.info("Using vectorized implementation of the environment.")
    logger.warn("EXPERIMENTAL FEATURE! Might not work in all cases")
    logger.warn(
        "Please make sure adequate parameters in settings.py have been set. Use https://github.com/DLR-RM/rl-baselines3-zoo/tree/master/hyperparams for correct parameters.")


    # Original solution: https://stackoverflow.com/a/53180921
    # Overridding Multiprocessing pool to allow multiprocessing of vectorized environment
    class NoDaemonProcess(multiprocessing.Process):
        @property
        def daemon(self):
            return False

        @daemon.setter
        def daemon(self, value):
            pass


    class NoDaemonContext(type(multiprocessing.get_context())):
        Process = NoDaemonProcess


    # We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
    # because the latter is only a wrapper function, not a proper class.
    class NestablePool(multiprocessing.pool.Pool):
        def __init__(self, *args, **kwargs):
            kwargs['context'] = NoDaemonContext()
            super(NestablePool, self).__init__(*args, **kwargs)


    pool = NestablePool(processes=int(arguments.number_of_agents))
    res = pool.map(run_agent, [i for i in range(int(arguments.number_of_agents))])

    pool.close()
    pool.join()
else:
    logger.info("Using NON-vectorized implementation of the environment")
    logger.warn("Please make sure adequate parameters in settings.py have been set.")
    with multiprocessing.Pool(processes=int(arguments.number_of_agents)):
        processes = [
            multiprocessing.Process(target=run_agent, args=(i,))
            for i in range(int(arguments.number_of_agents))
        ]

        for process in processes:
            process.start()

        for process in processes:
            process.join()

logger.info("Computation time: {} seconds".format(time.time() - start_time))
