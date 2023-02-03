import gym, sys
import numpy as np
import stable_baselines3
import torch as th

from patsy import dmatrices
import statsmodels.api as sm
import statsmodels.stats.power as pw

from venv import create
from gym import spaces
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import settings

########################################################################################################################
# The goal is to simulate the faults that can occur in user's code.
########################################################################################################################

########################################################################################################################
# Implemented Algorithms
########################################################################################################################
SUPPORTED_ALGORITHMS = ["A2C", "PPO", "DQN"]


# Calculate p-value using GLM method
# Taken from DeepCrime's replication package
def p_value_glm(orig_accuracy_list, accuracy_list):
    list_length = len(orig_accuracy_list)

    zeros_list = [0] * list_length
    ones_list = [1] * list_length
    mod_lists = zeros_list + ones_list
    acc_lists = orig_accuracy_list + accuracy_list

    data = {'Acc': acc_lists, 'Mod': mod_lists}
    df = pd.DataFrame(data)

    response, predictors = dmatrices("Acc ~ Mod", df, return_type='dataframe')
    glm = sm.GLM(response, predictors)
    glm_results = glm.fit()
    glm_sum = glm_results.summary()
    pv = str(glm_sum.tables[1][2][4])
    p_value_g = float(pv)

    return p_value_g


# Calculates cohen's kappa value
# Taken from DeepCrime's replication package
def cohen_d(orig_accuracy_list, accuracy_list):
    nx = len(orig_accuracy_list)
    ny = len(accuracy_list)

    dof = nx + ny - 2
    pooled_std = np.sqrt(
        ((nx - 1) * np.std(orig_accuracy_list, ddof=1) ** 2 + (ny - 1) * np.std(accuracy_list, ddof=1) ** 2) / dof)
    result = (np.mean(orig_accuracy_list) - np.mean(accuracy_list)) / pooled_std
    return abs(result)


# Calculate test method power
# Taken from DeepCrime's replication package
def power(orig_accuracy_list, mutation_accuracy_list):
    eff_size = cohen_d(orig_accuracy_list, mutation_accuracy_list)
    pow_ = pw.FTestAnovaPower().solve_power(effect_size=eff_size,
                                            nobs=len(orig_accuracy_list) + len(mutation_accuracy_list), alpha=0.05)
    return pow_


def hellinger_distance(p, q):
    """
    Calculate the Hellinger distance between discrete distributions

    """
    from scipy.spatial.distance import euclidean

    return euclidean(np.sqrt(p), np.sqrt(q)) / np.sqrt(2.0)


def create_model(
        algorithm: str,
        environment: gym.Env,
        hyper_parameters: dict,
        tensorboard_directory: str,
        mutation: dict = None,
):
    """
    create an agent's model

    Args:
        algorithm (str): Algorthm to use
        environment (gym.Env): Environment for the agent to train in
        hyper_parameters (dict): Hyper parameters for the algorithm
        tensorboard_directory (str): Path to the tensorboard directory
        mutation (dict): Dictionary of mutations
    Returns:
        agent's model
    """

    assert (
            algorithm.upper() in SUPPORTED_ALGORITHMS
    ), f"Algorithm {algorithm} is not supported yet"
    assert "seed" in hyper_parameters.keys(), "Seed must be specified brother"

    if algorithm.upper() == "A2C":

        if mutation is None or len(list(set(mutation).intersection(settings.MUTATION_AGENT_LIST))) == 0:
            from stable_baselines3.a2c.a2c import A2C
            from stable_baselines3.a2c.policies import MlpPolicy

            model = A2C(
                policy=MlpPolicy,
                env=environment,
                learning_rate=hyper_parameters["learning_rate"]
                if "learning_rate" in hyper_parameters.keys()
                else 0.001,
                n_steps=hyper_parameters["n_steps"]
                if "n_steps" in hyper_parameters.keys()
                else 5,
                gamma=hyper_parameters["gamma"]
                if "gamma" in hyper_parameters.keys()
                else 0.99,
                ent_coef=hyper_parameters["ent_coef"]
                if "ent_coef" in hyper_parameters.keys()
                else 0.0,
                vf_coef=hyper_parameters["vf_coef"]
                if "vf_coef" in hyper_parameters.keys()
                else 0.5,
                use_rms_prop=hyper_parameters["use_rms_prop"]
                if "use_rms_prop" in hyper_parameters.keys()
                else False,
                normalize_advantage=hyper_parameters["normalize_advantage"]
                if "normalize_advantage" in hyper_parameters.keys()
                else False,
                tensorboard_log=tensorboard_directory,
                seed=hyper_parameters["seed"] if "seed" in hyper_parameters.keys() else None,
            )

        else:

            from mutated_algorithms import MutatedA2C
            from stable_baselines3.a2c.policies import MlpPolicy

            model = MutatedA2C(
                policy=MlpPolicy,
                env=environment,
                learning_rate=hyper_parameters["learning_rate"]
                if "learning_rate" in hyper_parameters.keys()
                else 0.001,
                n_steps=hyper_parameters["n_steps"]
                if "n_steps" in hyper_parameters.keys()
                else 5,
                gamma=hyper_parameters["gamma"]
                if "gamma" in hyper_parameters.keys()
                else 0.99,
                ent_coef=hyper_parameters["ent_coef"]
                if "ent_coef" in hyper_parameters.keys()
                else 0.0,
                vf_coef=hyper_parameters["vf_coef"]
                if "vf_coef" in hyper_parameters.keys()
                else 0.5,
                use_rms_prop=hyper_parameters["use_rms_prop"]
                if "use_rms_prop" in hyper_parameters.keys()
                else False,
                normalize_advantage=hyper_parameters["normalize_advantage"]
                if "normalize_advantage" in hyper_parameters.keys()
                else False,
                tensorboard_log=tensorboard_directory,
                seed=hyper_parameters["seed"] if "seed" in hyper_parameters.keys() else None,
                mutation=mutation
            )

    elif algorithm.upper() == "PPO":

        if mutation is None or len(list(set(mutation).intersection(settings.MUTATION_AGENT_LIST))) == 0:

            from stable_baselines3.ppo.ppo import PPO
            from stable_baselines3.ppo.policies import MlpPolicy

            model = PPO(
                policy=MlpPolicy,
                env=environment,
                seed=hyper_parameters["seed"] if "seed" in hyper_parameters.keys() else 0,
                n_steps=hyper_parameters["n_steps"]
                if "n_steps" in hyper_parameters.keys()
                else 2048,
                batch_size=hyper_parameters["batch_size"]
                if "batch_size" in hyper_parameters.keys()
                else 64,
                gae_lambda=hyper_parameters["gae_lambda"]
                if "gae_lambda" in hyper_parameters.keys()
                else 0.95,
                gamma=hyper_parameters["gamma"]
                if "gamma" in hyper_parameters.keys()
                else 0.99,
                n_epochs=hyper_parameters["n_epochs"]
                if "n_epochs" in hyper_parameters.keys()
                else 4,
                ent_coef=hyper_parameters["ent_coef"]
                if "ent_coef" in hyper_parameters.keys()
                else 0.0,
                tensorboard_log=tensorboard_directory,
            )

        else:

            from mutated_algorithms import MutatedPPO
            from stable_baselines3.ppo.policies import MlpPolicy

            model = MutatedPPO(
                policy=MlpPolicy,
                env=environment,
                seed=hyper_parameters["seed"] if "seed" in hyper_parameters.keys() else 0,
                n_steps=hyper_parameters["n_steps"]
                if "n_steps" in hyper_parameters.keys()
                else 2048,
                batch_size=hyper_parameters["batch_size"]
                if "batch_size" in hyper_parameters.keys()
                else 64,
                gae_lambda=hyper_parameters["gae_lambda"]
                if "gae_lambda" in hyper_parameters.keys()
                else 0.95,
                gamma=hyper_parameters["gamma"]
                if "gamma" in hyper_parameters.keys()
                else 0.99,
                n_epochs=hyper_parameters["n_epochs"]
                if "n_epochs" in hyper_parameters.keys()
                else 4,
                ent_coef=hyper_parameters["ent_coef"]
                if "ent_coef" in hyper_parameters.keys()
                else 0.0,
                tensorboard_log=tensorboard_directory,
                mutation=mutation
            )

    elif algorithm.upper() == "DQN":

        if mutation is None or len(list(set(mutation).intersection(settings.MUTATION_AGENT_LIST))) == 0:
            from stable_baselines3.dqn.dqn import DQN
            from stable_baselines3.dqn.policies import MlpPolicy

            model = DQN(
                policy=MlpPolicy,
                env=environment,
                learning_rate=hyper_parameters["learning_rate"]
                if "learning_rate" in hyper_parameters.keys()
                else 1e-4,
                gamma=hyper_parameters["gamma"]
                if "gamma" in hyper_parameters.keys()
                else 0.99,
                batch_size=hyper_parameters["batch_size"]
                if "batch_size" in hyper_parameters.keys()
                else 32,
                buffer_size=hyper_parameters["buffer_size"]
                if "buffer_size" in hyper_parameters.keys()
                else 1e6,
                learning_starts=hyper_parameters["learning_starts"]
                if "learning_starts" in hyper_parameters.keys()
                else 50000,
                target_update_interval=hyper_parameters["target_update_interval"]
                if "target_update_interval" in hyper_parameters.keys()
                else 10000,
                train_freq=hyper_parameters["train_freq"]
                if "train_freq" in hyper_parameters.keys()
                else 4,
                gradient_steps=hyper_parameters["gradient_steps"]
                if "gradient_steps" in hyper_parameters.keys()
                else 1,
                exploration_fraction=hyper_parameters["exploration_fraction"]
                if "exploration_fraction" in hyper_parameters.keys()
                else 0.1,
                exploration_final_eps=hyper_parameters["exploration_final_eps"]
                if "exploration_final_eps" in hyper_parameters.keys()
                else 0.05,
                policy_kwargs=hyper_parameters["policy_kwargs"]
                if "policy_kwargs" in hyper_parameters.keys()
                else None,
                tensorboard_log=tensorboard_directory,
                seed=hyper_parameters["seed"] if "seed" in hyper_parameters.keys() else None,
            )

        else:

            from mutated_algorithms import MutatedDQN
            from stable_baselines3.dqn.policies import MlpPolicy

            model = MutatedDQN(
                policy=MlpPolicy,
                env=environment,
                learning_rate=hyper_parameters["learning_rate"]
                if "learning_rate" in hyper_parameters.keys()
                else 1e-4,
                gamma=hyper_parameters["gamma"]
                if "gamma" in hyper_parameters.keys()
                else 0.99,
                batch_size=hyper_parameters["batch_size"]
                if "batch_size" in hyper_parameters.keys()
                else 32,
                buffer_size=hyper_parameters["buffer_size"]
                if "buffer_size" in hyper_parameters.keys()
                else 1e6,
                learning_starts=hyper_parameters["learning_starts"]
                if "learning_starts" in hyper_parameters.keys()
                else 50000,
                target_update_interval=hyper_parameters["target_update_interval"]
                if "target_update_interval" in hyper_parameters.keys()
                else 10000,
                train_freq=hyper_parameters["train_freq"]
                if "train_freq" in hyper_parameters.keys()
                else 4,
                gradient_steps=hyper_parameters["gradient_steps"]
                if "gradient_steps" in hyper_parameters.keys()
                else 1,
                exploration_fraction=hyper_parameters["exploration_fraction"]
                if "exploration_fraction" in hyper_parameters.keys()
                else 0.1,
                exploration_final_eps=hyper_parameters["exploration_final_eps"]
                if "exploration_final_eps" in hyper_parameters.keys()
                else 0.05,
                policy_kwargs=hyper_parameters["policy_kwargs"]
                if "policy_kwargs" in hyper_parameters.keys()
                else None,
                tensorboard_log=tensorboard_directory,
                seed=hyper_parameters["seed"] if "seed" in hyper_parameters.keys() else None,
                mutation=mutation
            )

    return model


def load_model(algorithm: str, environment: gym.Env, model_path: str):
    """
    load agent's model from specified path

    Args:
        algorithm (str): agent's algorithm
        environment (gym.Env): agent's enviornment
        model_path (str): path to the previous model

    Returns:
        agent's trained model
    """

    if algorithm.upper() == "A2C":

        from stable_baselines3.a2c.a2c import A2C

        model = A2C.load(
            model_path,
            custom_objects={
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
            },

        )
        model.set_env(environment)

    elif algorithm.upper() == "PPO":

        from stable_baselines3.ppo.ppo import PPO

        model = PPO.load(
            model_path,
            custom_objects={
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.0,
            },
        )
        model.set_env(environment)

    elif algorithm.upper() == "DQN":

        from stable_baselines3.dqn.dqn import DQN

        model = DQN.load(
            model_path,
            custom_objects={
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.0,
                "exploration_schedule": None
            },
        )
        model.set_env(environment)
    else:
        raise Exception("Model not implemented")

    return model


if __name__ == "__main__":
    test_model = create_model(
        "ma2c",
        gym.make("CartPole-v1"),
        {"seed": 0},
        "runs/healthy",
        mutation={"incorrect_loss_function": None},
    )
    test_model.learn(total_timesteps=100000)
    print("hi")
