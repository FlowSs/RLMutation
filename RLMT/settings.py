import os

from typing import Callable


# needed for multiprocessing
# Putting it here for now
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


MUTATION_AGENT_LIST = {"no_reverse": "NR", "no_discount_factor": "NDF", "missing_terminal_state": "MTS",
                       "missing_state_update": "MSU", "incorrect_loss_function": "ILF",
                       "policy_optimizer_change": "POC", "policy_activation_change": "PAC"}
MUTATION_ENVIRONMENT_LIST = {"repeat": "R", "random": "Ra", "mangled": "M", "reward_noise": "RN"}

CUSTOM_ENV_SETTINGS ={'CartPole-v1': {
    "max_episode_steps": 500,
    "reward_threshold": 475.0
},
    'LunarLander-v2': {
    "max_episode_steps": 1000,
    "reward_threshold": 200
}
}
HYPER_PARAMS = {
    "CartPole-v1": {
        "ppo": {
        },
        "a2c": {
            "ent_coef": 0.0
        },
        "dqn": {
            "learning_rate": float(2.3e-3),
            "batch_size": 64,
            "buffer_size": 100000,
            "learning_starts": 1000,
            "gamma": 0.99,
            "target_update_interval": 10,
            "train_freq": 256,
            "gradient_steps": 128,
            "exploration_fraction": 0.16,
            "exploration_final_eps": 0.04,
            "policy_kwargs": dict(net_arch=[256, 256])
        }
    },
    "LunarLander-v2": {
        "a2c": {
            "gamma": 0.995,
            "n_steps": 5,
            "learning_rate": linear_schedule(0.00083),
            "ent_coef": 0.00001
        },
        "ppo": {
            "n_steps": 1024,
            "batch_size": 64,
            "gae_lambda": 0.98,
            "gamma": 0.999,
            "n_epochs": 4,
            "ent_coef": 0.01
        },
        "dqn": {
            "learning_rate": float(6.3e-4),
            "batch_size": 128,
            "buffer_size": 50000,
            "learning_starts": 0,
            "gamma": 0.99,
            "target_update_interval": 250,
            "train_freq": 4,
            "gradient_steps": -1,
            "exploration_fraction": 0.12,
            "exploration_final_eps": 0.1,
            "policy_kwargs": dict(net_arch=[256, 256])
        }
    }
}
