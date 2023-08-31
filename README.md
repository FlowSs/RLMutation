# Mutation Testing for Deep Reinforcement Learning

This replication package is intended for the paper "_Mutation Testing of Deep Reinforcement Learning
Based on Real Faults_" accepted to the International Conference on Software Testing (ICST) 2023. A preprint version is available
on [arxiv](https://arxiv.org/abs/2301.05651) and the published version is available on the publisher's website [IEEE](https://ieeexplore.ieee.org/abstract/document/10132198).

## Index

- [Requirements](#requirements)
- [Mutation operators comparison with Lu et al. paper](#mutation-operators-comparison-with-lu-et-al-paper)
- [Training of agents](#training-of-agents)
- [Evaluating agents](#evaluating-agents)
- [Mutation Test on Initial Envs](#mutation-test-on-the-initial-environment)
- [Generating Test Envs](#generating-test-environments)
- [Evaluating Higher Order Mutations Props](#evaluating-higher-order-mutation-properties)
- [References](#references)

## Requirements

Python 3.8 was used alongside the following libraries:

<details>
  <summary>Click to see</summary>
  
```
absl-py==1.2.0
Box2D==2.3.10
cachetools==5.2.0
charset-normalizer==2.1.1
cloudpickle==2.2.0
contourpy==1.0.5
cycler==0.11.0
fonttools==4.37.4
google-auth==2.12.0
google-auth-oauthlib==0.4.6
grpcio==1.49.1
gym==0.21.0
idna==3.4
importlib-metadata==4.13.0
kiwisolver==1.4.4
Markdown==3.4.1
MarkupSafe==2.1.1
matplotlib==3.6.1
numpy==1.23.1
oauthlib==3.2.1
packaging==21.3
pandas==1.5.0
patsy==0.5.3
Pillow==9.2.0
protobuf==3.19.6
pyasn1==0.4.8
pyasn1-modules==0.2.8
pyparsing==3.0.9
python-dateutil==2.8.2
pytz==2022.4
requests==2.28.1
requests-oauthlib==1.3.1
rsa==4.9
scipy==1.9.2
six==1.16.0
stable-baselines3==1.6.2
statsmodels==0.13.2
tensorboard==2.10.1
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
torch==1.11.0
tqdm==4.64.1
typing_extensions==4.4.0
urllib3==1.26.12
Werkzeug==2.2.2
zipp==3.9.0
```
</details>

## Mutation operators comparison with Lu et al. paper

We give an extended version of the descriptions of our mutation operators as well as to which paper it references to.
<details>
  <summary>Click to see the table</summary>

| Our Mutation | Description    | Reason for inclusion     | 
| :---        |    :----:   |  :--- |
| _Reward Noise (RN)_      | In RL, one of the most important signals that the agent receives for evaluating its performance is the reward. The agent uses the reward signal to assess the quality of the actions it has taken in the states that it has visited. The _RN_ mutation operator adds a noise to the true reward that the agent was meant to receive and returns it to the agent. We denote the noisy reward $r_t^\*$. The new reward is calculated as $$r_t^\* = r_t + \mathcal{N}(0,~0.1 \times r_t)$$      | The agent may observe incorrect rewards from the environment due to sensory faults, bugs in environment implementation, or nefarious attacks [6]|  
| _Mangled (M)_      | As explained in Background section, the agent collects its interactions with the environment in the form of $(s_t, a_t, r_t, s_{t+1})$ tuple. The correlation (like order, the resultant state/reward from previous actions) between these collected interactions, allows the agent to learn from its experiences. The _M_ mutation operator damages the correlation between collected experiences. This operator returns a random $s_{t+1}$ and $r_t$ which are not the state and reward the agent should receive according to its current state and the taken action. We denote the new randomly selected $s_{t+1}$ and $r_t$ as $s'_{t+1}$ and $r'\_t$, respectively. Therefore, instead of receiving $(s_t, a_t, r_t, s\_{t+1})$, the agent receives $(s\_t, a\_t, r'\_t, s'\_{t+1})$ | The agent may observe incorrect (state, reward) tuple from the environment due to sensory faults, bugs in environment implementation, or nefarious attacks [6]|  
| _Random (Ra)_      | Similar to the mangled mutation operator, the \textit{Ra} mutation returns a $(s_t, a_t, r'\_t, s'\_{t+1})$ tuple to the agent. However, unlike the _M_ operator where $s'\_{t+1}$ and $r'\_t$ are selected randomly and are not associated with each other, the random operator returns a $s'_{t+1}$ and $r'_t$ to the agent which were sampled from the same experience tuple but have no association with $s_t$ and $a_t$. | Same as above|  
| _Repeat (R)_      | Returns the previous observation to the agent. If the agent has two consecutive experiences in the form of $(s_t, a_t, r_t, s_{t+1})$ and $(s_{t+1}, a_{t+1}, r_{t+1}, s_{t+2})$, this operator returns $(s\_{t+1}, a\_{t+1}, r\_{t+1}, s\_{t+2})$ with $r\_{t+1} = r\_t, s\_{t+2} = s\_{t+1}$  | Same as above|  
| _No Discount Factor (NDF)_      | In order for the agent to learn a balance between short-term and long-term received rewards, the discount factor is used. If the developer does not implement this concept for calculating the rewards during agent's training, the resulting agent will put equal importance on all the actions it has taken and the rewards it has received. As a result, such an agent will either be unable to correctly learn the mechanics of the environment or have difficulty in doing so.  | _Wrong network update_ category [3]|  
| _Missing Terminal State (MTS)_      | In RL environments, the terminal state is defined as the state which was the last state the agent transitioned to before the environment was terminated. The termination criteria for an environment can be reaching the goal state, stepping into a trap state, or reaching a time limit. Normally in designing an RL environment, the terminal state contains a different reward signal than the rest of the transitions, e.g., if the agent falls in a trap, it should receive a negative reward. This mutation operator simulates the cases where the developer incorrectly implements identifying the termination criteria. As a result, the agent will not receive the termination signal and will be unable to correctly determine the correlation between the actions taken and the results achieved.  | _Missing terminal state_ category [3]|  
| _No Reverse (NR)_      | For calculating the returns of an episode, recent rewards are discounted less than the earlier ones. During implementing this concept, it is imperative that the practitioner pays attention to how this concept is applied. Wrong network update, a common mistake that developers make is forgetting to reverse the order of the received rewards during calculating the returns. The mutation operator simulates such cases where the resulting agents, do not reverse the order of the received rewards and therefore learn an incorrect association between the experiences.  | _Wrong update rule_ category [3]|  
| _Missing State Update (MSU)_      | The agent receives its interaction with the environment as a $(s\_t, a\_t, r\_t, s\_{t+1})$ tuple. Therefore, the developer needs to pay attention to correctly updating the state that agent is in, after taking an action. This means that after the agent takes another step and transitions from $s\_{t+1}$ to $s\_{t+2}$, the developer needs to update the experience tuple from $(s\_t, a\_t, r\_t, s\_{t+1})$ to $(s\_{t+1}, a\_{t+1}, r\_{t+1}, s\_{t+2})$. This mutation operator simulates the condition in which the developer makes a mistake in doing so. As a result, the agent always sees the same state in its experience tuple (e.g. $(s\_{t}, a\_{t+1}, r\_{t+1}, s\_{t+2})$ ) during training and is unable to learn the mechanics of the environment  | _Missing stepping the environment_ category [3] |  
| _Incorrect Loss Function (ILF)_      | During implementation of a deep RL algorithm, one of the most important steps is the correct implementation of the loss function. The loss function calculates the error of the agent's estimations and updates its Neural Network (NN) accordingly. This mutation operator, simulates the cases where a developer implements the loss function incorrectly. As a result, the error for agent's NN outputs is incorrectly calculated and the agent will be unable to learn the mechanics of the environment.  | _Wrong update rule_ category [3] |  
| _Policy Activation Change (PAC)_      | NNs generally reliy on non-linear activation functions such as ReLU and TanhH in order to introduce non-linearity which helps in modeling complex behaviors like object detection or image classification. Choosing the activation function correctly is important as it can have a great impact on how the model learns the policy. This mutation changes the default activation used in the policy network, i.e., NN that learns the agent’s policy.  | DeepCrime [1] _Change Activation Function (ACH)_ |  
| _Policy Optimizer Change (POC)_      | To search for the best set of weights for a given task, NNs use optimizers. An optimizer is an algorithm leveraged to minimize a loss function, with Gradient Descent methods being the most popular. As with activation, the choice of the optimizer used by an NN is crucial for the proper learning of the policy. This mutation modifies the default optimizer used for each algorithm, while keeping the learning rate the same. | DeepCrime [1] _Change Optimisation Function (OCH)_ | 

</details>

We list down the mutations used in Lu et al. [2].  We describe if a mutation is included to which of our mutations it refers to.
If not included we give a reason why it was not included, generally there is a lack of
evidence that such operator is based on a real fault, that is they did not provide a reference and
we did not find evidence of such fault in the taxonomy we refer to.

<details>
  <summary>Click to see the table</summary>

| Their Mutation | Description    | Included or not?     | 
| :---        |    :----:   |  :--- |
| _Reward Reduction_      | Reduce High Reward       | Similar to our _Reward Noise_ except we add a noise instead which mimic a sensor failure better. Manually setting is also more complex and prone to bias. |  
| _Reward Increase_      | Increase Low Reward       | Same as above   |  
| _Reward Instability_      | Make rewards unstable       | Same as above   |  
| _State Record Crash_      | Hide or Duplicate state-action pairs       | Similar to our _Repeat_ for the duplicate part. It's not clear how a state could *hide*, since something has to be returned to the agent. Nonetheless, we did not find a fault which would cause the program to _hide_ as described.   |  
| _State Delay_      | Improperly associate a state with a subsequent action       | Similar to our _Mangled_  |  
| _State Repetition_      | Associate a fixed state with a series of continuous actions       | Similar to our _Repeat_, but seems to be more general.  |  
| _State Error_      | Associate a fixed state with a series of continuous actions       | Similar to our _Random_  |  
| _Q-Table Fuzzing_      | Fuzz Q-Table       | Too specific to simple Q-learning. Not observed in the faults we had.  |  
| _Input-Layer Neuron Removal_      | Remove input neurons       | While removed neurons can happen as they are describe in DeepCrime [1], they note that their implementation is "tricky as they trigger a cascade of changes" and "The usage of such operators might be limited depending on the structure of the network under test as some of the generated mutants might be causing crashes, which is not particularly useful for the mutation testing". In particular, Input/Output layers are sensitive ones as any change in the shape expected can cause problem since it won't match with the input fed to the network or expected by the labels which limits the relevance of generated mutants. Thus, we chose not to implement them.  |  
| _Output-Layer Neuron Removal_      | Remove output neurons       | Same as above  |  
| _Output-Layer Neuron Addition_      | Add neuron(s) in output layer       | Same as above  |  
| _Changing Exploration_      | Switch exploration approach       | Specific to Q-Learning. Not observed in the faults we had.  |  
| _Mutating Epsilon_      | Mutate ε value       | Specific to Q-Learning. Not observed in the faults we had.  |  
| _Changing Noise_      | Change noise injected into the parameters of Q-function       | For DQN, this would be akin to fuzzing the weights. Something that was done in for instance DeepMutation [4] in supervised learning. Yet, this mutation seems not to be based on a real fault as DeepCrime's taxonomy does not mention it. Similarly, we did not observe such fault in RL. Not implemented.  |  
| _Switching Buffer_      | Switch replay buffer mechanism       | We did not observed a fault where the switching off the replay buffer. The closest we have is implemented with the _No Reverse_ which affect the order in which rewards are discounted.  |  
| _Fuzzing Buffer_      | Lose, repeat or modify experiences      | This mutation seems similar to some of their state level ones in their effect, expect it intervenes at the buffer level. Since it affects the experiences of the agents (i.e. tuple (State, action, reward)), our operators such as _Repeat_ or _Reward Noise_ act similarly. Nonetheless, we did not observe mutations that affected the buffer in that way (i.e. losing/modifying experiences in the buffer once they have been observed). |  
| _Shuffling Replay Priority_      | Change the priority of replays       | A random version of our _No Reverse_. Is also covered partly in the _No Discount Factor_ which is a special case of their mutation (i.e. all replays have the same priority). We did not observe a _shuffling_ of the replay buffer. |  
| _Precision Mutation_      | Mutate the precision of discretizing continuous spaces       | We did not observe such mutation. Moreover, in our experiments, we do not have environment with continuous spaces.  |  
| _Approximation Mutation_      | Change approximation methods       | Same as above |  
</details>

## Training of agents

Trained agents are available on [Zenodo](https://zenodo.org/record/7233122) as `.zip` file (~ 2GB). The archive should be
unzipped in the `experiments/` folder.

If you wish to retrain the agents as we did, you can run the following script:
```bash
python train.py -a [algorithm] -na [number_of_agents] -e [environment] -t [total_steps] -s [start_from] -m [mutation] 
```

`algorithm` is the algorithm name for instance `ppo` and `environment` the environment name
for instance `CartPole-v1`. `number_of_agents` is the total number of agents to train (in our case 20),
`total_steps` is the number of training steps, `start_from` is used to start training 
from a certain agent (for instance, `-s 0` will start from agent of seed 0). Finally, `mutation` is
the mutations are to be applied, as a string representing a dict. Omitting this parameter will train healthy
agents by default.

For instance:
```bash
python train.py -a ppo -na 20 -e CartPole-v1 -t 200000 -s 0 -m '{"no_reverse": "None"}' 
```

will train 20 mutated PPO agents on CartPole-v1 for 200000 steps starting at seed 0 using
the _no_reverse_ mutation (note that `"None"` means _no_reverse_ do not take extra parameters).
This will create a `logs/` and `runs/` directory, with `logs/` containing the checkpoints/best weights
of the agents and `runs/` some date needed to plot the training evolution over tensorboard. Only `logs/`
is needed for the rest of the procedure.

IMPORTANT NOTE: In our case, we trained everything on CPUs and so the number of agents being trained
equals the number of CPUs on which they will be trained (so in our case 20 CPUS).

Hyper-parameters used to train the models are given in `settings.py` in the `HYPER_PARAMS` variable.
They are the same as the tuned ones presented in the [RL-Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
repositories. The only difference being that, since we could not use the vectorized option of 
stable baselines 3 as it conflicts with our multiprocess training/mutation injection, the number of vectorized environments
for `PPO` and `A2C` was set to 1 and we increased the number of total steps to compensate. We checked
that obtained value were inline were benchmark ones. The total steps for each environment/algorithm is
as follow:

- `PPO`/`CartPole-v1`: 200,000
- `PPO`/`LunarLander-v2`: 2,000,000
- `A2C`/`CartPole-v1`: 1,000,000
- `A2C`/`LunarLander-v2`: 2,000,000
- `DQN`/`CartPole-v1`: 50,000
- `DQN`/`LunarLander-v2`: 100,000

## Evaluating agents

Trained agents should be properly put in the `experiments/` folder. The `.zip` file is already
with the correct architecture and can be unzipped in the directory as it is. If agents were trained
from scratch, the `logs/` directory obtained should be placed in the correct sub-directory.

The next step is to evaluate the agents on the environment and to get their reward. This can
be done with the following scripts:
```bash
python test_agent.py -a [algorithm] -na [number_of_agents] -e [environment] -m [mutation] 
```

Parameters are the same as in the [Training of agents](#training-of-agents). This will generate
a `.csv` file in the `csv_res` directory (already by default in the repository both in `csv_res/`
and in `results/rewards_evaluation/`) with the reward of the agents. 

The evaluation is done over 10 episodes with a fixed seed to make sure the agents are evaluated
on the same episodes for a fair comparison.

## Mutation test on the initial environment

This part concerns the results of RQ1. Once the rewards of mutated and healthy agents are
obtained, we can run the following scripts to obtain the results presented in Table III.
```bash
python eval_mut.py -a [algorithm] -e [environment] -m [mutation]
```
Parameters are the same as before except `mutation` which here is coded with the acronym of the mution.
For instance, _no_reverse_ is _NR_.

Thus,
```bash
python eval_mut.py -a ppo -e CartPole-v1 -m NR
```

will return

```buildoutcfg
Environement: CartPole-v1, Model ppo, Mutation NR
Average Mutated/Healthy Ratio Test : 1.0
Reward Distribution Test : Killed
Distance Distribution Test: Killed
```

All the results are already stored in the file `mutations_results_initial_environment.txt` in
the `results` directory.

## Generating test environments

As described in RQ2, in order to find relevant First Order Mutations (FOM) following Jia et al. [5],
we need to come up with multiple test cases to test our FOM on, _i.e._ in our case multiple
test environments. Candidate test environments are generated by modifying two parameters of an 
initial environments. Those custom environments can be found in `custom_test_env.py`. The parameters
modified as well as the initial/boundaries values are as follow:

| Environment | Parameters     | Initial Values     | Boundaries Values     |
| :---        |    :----:   | :----:   |          ---: |
| CartPole-v1      | _cart's mass_, _pole's mass_       | [1.0, 0.1]   |  [[1.0, 20.0], [0.1, 20.0]]   |
| LunarLander-v2   | _gravity_, _side_engine power_        | [-10.0, 0.6]       | [[-20.0, 0], [0, 5.0]]   |

To search for the boundary test environments as well as to evaluate said test environments on
the FOM, the following script should be run:
```bash
python algo.py -a [algorithm] -e [environment] -i [init_val] -l [limits] -b [bounds] -t [test_mode] -n [number_of_cpus]
```

`init_val` is a list of the initial value of the environment, `limits` are the search limits for the
binary search for each parameter given as `lower_param_1 upper_param_1 lower_param_2...`, `bounds` are
the test environments parameters if already computed (avoid the search). `test_mode` is the mutation test
method being applied. Both are distribution based following DeepCrime's approach, only the distribution metric
changes with `r` for reward based (default) and `dtr` for inter/intra distance based (see RQ1 results). `number_of_cpus` is simply
the number of cpus used (by default, maximum available - 1).

For instance:
```bash
python algo.py -a ppo -e CartPole-v1 -i 1.0 0.1 -l 1.0 20.0 0.1 20.0
```
will calculate the generated test environments for `CartPole-v1` with `PPO` algorithm and the initial/boundaries values
presented in the previous table. This will generate a `.csv` file named `mut_ppo_CartPole-v1.csv` with the list of
generated test environments and a boolean for each FOM depending on whether said test environment killed the FOM or
not given the test method reward based. Results for all FOM are already present in the `results/` directory which
yield the table IV in our paper.

## Evaluating Higher Order Mutation properties

Based of results of the previous script (RQ2 in our paper), relevant FOM are identified and combined for 
each algorithm/environment type/test method to obtain Higher Order Mutation (HOM). At that stage, new mutated agents
using those HOM need to be trained following a similar procedure as in [Training of agents](#training-of-agents),
with the `mutation` properly set. So for instance for a HOM including _no_reverse_ and _missing_state_update_, the
flag in `train.py` becomes `'{"no_reverse": "None", "missing_state_update": "None"}'`.

Models trained need to have the `logs/` directory set in the correct sub-directory. Once again, if you uncompress
the `.zip` from Zenodo, everything will be set accordingly.

`.csv` files from the previous RQ should be put into the sub-directory `fom_test_env_killed` (by default, the files
are already there).

Finally, the following script should be run:
```bash
python hom_prop.py -a [algorithm] -e [environment] -t [test_mode] -n [number_of_cpus]
```

Parameters are the same as before. All information about test environments parameters... will be taken from the
relevant `.csv` file from the directory.

For instance,
```bash
python hom_prop.py -a ppo -e CartPole-v1 -t r
```
will calculate the killed HOM for each test environments previously generated and stored in `mut_ppo_CartPole-v1.csv`,
as well as the HOM properties, _i.e._ Subsuming/Non-Subsuming and Coupled/Decoupled. Obtained file is `hom_ppo_CartPole-v1.csv`.

## Hot Fixes

07/02/2023: Error in the 'no_discount_factor' mutation applied to DQN: it should be `(1 - replay_data.dones)` and not `(replay_data.dones)`. Updated the code as well as provided trained agents with the correct mutation in `results/corrected_NDF_DQN/`.

When testing on the initial environment, the following results were obtained with the new agents:

```
Environement: CartPole-v1, Model dqn, Mutation NDF
Average Mutated/Healthy Ratio Test : 0.55
Reward Distribution Test : Killed
Distance Distribution Test: Killed
```

```
Environement: LunarLander-v2, Model dqn, Mutation NDF
Average Mutated/Healthy Ratio Test : 0.95
Reward Distribution Test : Killed
Distance Distribution Test: Killed
```

i.e. only a change on the AVG method when the environment is CartPole-v1 (from 1.0 to 0.55), rest is unchanged.

## References

[1] N. Humbatova, G. Jahangirova, and P. Tonella, “Deepcrime: mutation
testing of deep learning systems based on real faults,” in Proceedings of
the 30th ACM SIGSOFT International Symposium on Software Testing
and Analysis, pp. 67–78, 2021.

[2] Y. Lu, W. Sun, and M. Sun, “Towards mutation testing of reinforcement
learning systems,” Journal of Systems Architecture, vol. 131, p. 102701, 2022.

[3] A. Nikanjam, M. M. Morovati, F. Khomh, and H. Ben Braiek, “Faults
in deep reinforcement learning programs: a taxonomy and a detection
approach,” Automated Software Engineering, vol. 29, no. 1, pp. 1–32, 2022

[4] L. Ma, F. Zhang, J. Sun, M. Xue, B. Li, F. Juefei-Xu, C. Xie, L. Li,
Y. Liu, J. Zhao, et al., “Deepmutation: Mutation testing of deep learning
systems,” in 2018 IEEE 29th International Symposium on Software
Reliability Engineering (ISSRE), pp. 100–111, IEEE, 2018.

[5] Y. Jia and M. Harman, “Higher order mutation testing,” Information
and Software Technology, vol. 51, no. 10, pp. 1379–1393, 2009. Source
Code Analysis and Manipulation, SCAM 2008.

[6] T. Everitt, V. Krakovna, L. Orseau, M. Hutter, and S. Legg, “Reinforce-
ment learning with a corrupted reward channel,” 2017.
