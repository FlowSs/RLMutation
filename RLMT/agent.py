import custom_callbacks, utils, mutated_env
import custom_callbacks
import os, gym, sys
from torch.utils import tensorboard

from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

########################################################################################################################
# The goal is to simulate the faults that can occur in user's code.
########################################################################################################################

import settings

# !some bug for stable-baselines3 causes recursion error. (https://stackoverflow.com/questions/67986157/multiprocessing-how-to-debug-pickle-picklingerror-could-not-pickle-object-as)
sys.setrecursionlimit(1000)


class Agent:
    def __init__(
        self,
        algorithm: str,
        hyper_params: dict,
        environment: str,
        agent_number: int,
        vectorized: bool,
        log_dir=None,
    ):
        """
        Since we'll be using multiprocessing, we need to make a class for the agents.
        Each agent is going to be responsible for its own training and logging.
        Args:
            algorithm (str): Which RL algorithm to use
            hyper_params (dict): Hyperparameters for the algorithm
            environment (str): Name of the gym environment to use
            agent_number (int): Agent number
            log_dir (str): Path to log directory
        """
        self.algorithm = algorithm
        self.hyper_params = hyper_params
        self.environment = environment
        # Create the ID. we'll use Healthy as the prefix for agents without mutations
        self.id = f"Healthy_{algorithm}_{environment}_{agent_number}"
        # Set the finished training flag to false. I'll need this flag to which agents to not train again
        self.finsihed_training = False
        self.init_finished = False
        self.log_dir = log_dir
        self.vectorized = vectorized

    def init_log(self):
        """
        Create a directory for each agent to store their models and logs
        """
        if self.log_dir is None:
            # create the logs directory
            self.log_dir = os.path.join(os.getcwd(), "logs", self.id)
            # check to see if the directory exists and if not, create it
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

    def init_env(self, test=False):
        """
        Initialize the environment.
        Using Montior to log the environment.
        Using Monitor also allows for setting up mutations in the environment.
        
        args:
            test (bool): if true, the environment will be set to test mode
        """
        # If not using vectorized environment or if we are just evaluating the environment (no need for vectorized!)
        if not self.vectorized or test:
            self.environment = gym.make(self.environment)
            # Fixing the seed when evaluating to allow for comparison across agents and mutations
            if test:
                self.environment.seed(420)
            self.environment = Monitor(
                   self.environment, self.log_dir if not test else None
            )
        else:
            # DummyVecEnv is better for CartPole it seems
            if self.environment == 'CartPole-v1':
                self.environment = VecMonitor(
                    make_vec_env(self.environment, n_envs=8, vec_env_cls=DummyVecEnv), self.log_dir
                )
            # On lunar lander, SubProcVecEnv is better
            elif self.environment == 'LunarLander-v2':
                self.environment = VecMonitor(
                    make_vec_env(self.environment, n_envs=16, vec_env_cls=SubprocVecEnv, vec_env_kwargs={'start_method': 'fork'}), self.log_dir
                )
            else:
                raise Exception("Environment {} not implemented for vectorized environment".format(self.environment))
        

    def init_model(self):
        """
        Initialize the agent's model
        """
        self.model = utils.create_model(
            algorithm=self.algorithm,
            environment=self.environment,
            hyper_parameters=self.hyper_params,
            tensorboard_directory=f"runs/{self.id}",
        )

    def init_agent(self, test=False):
        if test and self.log_dir is None:
            raise Exception("Test flag is provided, please input the log_dir argument")
        self.init_log()
        self.init_env(test)

    def train(self, total_timesteps: int):
        """
        Agent's training function.

        Args:
            total_timesteps (int): number of time steps to train for
        """
        # the model needs to be created in the same process as the agent
        self.init_model()
        # best callback saves the best model
        best_callback = custom_callbacks.SaveOnBestTrainingRewardCallback(
            check_freq=1000, log_dir=self.log_dir
        )
        # checkpoint callback saves the model every 1000 time steps
        checkpoint_callback = custom_callbacks.CheckpointCallback(
            save_freq=1000, log_dir=self.log_dir
        )

        callback = CallbackList([best_callback, checkpoint_callback])
        # start training
        self.model.learn(total_timesteps, callback=callback)
        self.finsihed_training = True

    def test(self, return_dict: dict):
        """
        Agent's testing function.

        Args:
            return_dict (dict): dictionary to store the results

        """
        self.model = utils.load_model(self.algorithm, self.environment, self.log_dir)
        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.model.get_env(),
            n_eval_episodes=self.hyper_params["n_episodes"],
        )

        return_dict[self.id] = [mean_reward, std_reward]


class MutatedAgent(Agent):
    def __init__(
        self,
        algorithm: str,
        hyper_params: dict,
        environment: str,
        agent_number: int,
        vectorized: bool,
        mutation: dict = None,
        log_dir=None,
    ):
        """
        This is a mutated agent.
        It inherits from the Agent class.
        Depending on the mutation target, it will either simulated a noisy environment or a faulty agent.
        
        args:
            algorithm (str): Which RL algorithm to use
            hyper_params (dict): Hyperparameters for the algorithm
            environment (str): Name of the gym environment to use
            agent_number (int): Agent number
            mutation (dict): dictionary of mutation and their magnitude
            log_dir (str): Path to log directory
        """

        super(MutatedAgent, self).__init__(
            algorithm, hyper_params, environment, agent_number, vectorized, log_dir
        )

        self.mutation = mutation

        mut_dict = dict(settings.MUTATION_AGENT_LIST)
        mut_dict.update(settings.MUTATION_ENVIRONMENT_LIST)
        mut_mode = '-'.join(map(lambda x: '_'.join(x), [[mut_dict[key], self.mutation[key]] if self.mutation[key] != 'None' else [mut_dict[key]] for key in self.mutation.keys()]))
        self.id = f"{mut_mode}_{algorithm}_{environment}_{agent_number}"

    def init_env(self, mutation_magnitude: any = None, test: bool = False):
        """
        This function is responsible for initializing the mutated environment.
        If the mutation target is anything else that environment_noise, then a healthy environment will be initialized.
        
        args:
            mutation_magnitude : If the mutation target is environment, the mutation magnitude to use
            test (bool): if true, the environment will be set to test mode
        """

        assert len(list(set(self.mutation).intersection(settings.MUTATION_AGENT_LIST))) != 0 or len(list(set(self.mutation).intersection(settings.MUTATION_ENVIRONMENT_LIST))) != 0, "Invalid mutation mode"
        
        if not self.vectorized:
            # Is the environment modified or not
            if len(list(set(self.mutation).intersection(settings.MUTATION_ENVIRONMENT_LIST))) != 0:
                # if the mutation target is environment, we need to create a noisy environment
                self.environment = gym.make(self.environment)
                self.environment = mutated_env.MutatedMonitor(
                    self.environment,
                    filename=self.log_dir,
                    mutation=self.mutation,
                )

            else:
                self.eval_env = gym.make(self.environment)
                self.environment = gym.make(self.environment)
                self.environment = Monitor(
                    self.environment,
                    self.log_dir,
                )
        else:
            # Is the environment modified or not
            if len(list(set(self.mutation).intersection(settings.MUTATION_ENVIRONMENT_LIST))) != 0:
                raise NotImplementedError("Mutating a vectorized monitor is not handled yet!")

            else:
                # DummyVecEnv is better for CartPole it seems
                if self.environment == 'CartPole-v1':
                    self.environment = VecMonitor(
                        make_vec_env(self.environment, n_envs=8, vec_env_cls=DummyVecEnv), self.log_dir
                    )
                # On lunar lander, SubProcVecEnv is better
                elif self.environment == 'LunarLander-v2':
                    self.environment = VecMonitor(
                        make_vec_env(self.environment, n_envs=16, vec_env_cls=SubprocVecEnv, vec_env_kwargs={'start_method': 'fork'}), self.log_dir
                    )
                else:
                    raise Exception("Environment {} not implemented for vectorized environment".format(self.environment))

    def init_model(self):
        """
        Initialize the agent's model
        """
        self.model = utils.create_model(
            algorithm=self.algorithm,
            environment=self.environment,
            hyper_parameters=self.hyper_params,
            tensorboard_directory=f"runs/{self.id}",
            mutation=self.mutation,
        )

    def init_agent(self, test: bool = False):
        if test:
            raise Exception("Use a normal Agent object to evaluate the RL algorithm")
            
        self.init_log()

        if len(list(set(self.mutation).intersection(settings.MUTATION_ENVIRONMENT_LIST))) != 0:
            self.init_env(self.mutation, test=False)
        else:
            self.init_env(test=test)


if __name__ == "__main__":
    """
    Sanity check for the agent class.
    """
    # Agent with fault
    agent3 = MutatedAgent(
        algorithm="A2C",
        hyper_params={
            "seed": 0,
            "gamma": 0.995,
            "n_steps": 5,
            "learning_rate": 0.00083,
            "ent_coef": 0.00001,
        },
        environment="LunarLander-v2",
        agent_number=2,
        mutation={"missing_terminal_state": "None", "policy_optimizer_change": "SGD", "repeat": "0.5"},
    )
    agent3.init_agent()

    print("\033[91m" + "Starting agent with simulated fault training" + "\033[0m")
    agent3.train(total_timesteps=int(1e5))
    print("\033[92m" + "Finished agent with simulated fault training" + "\033[0m")

