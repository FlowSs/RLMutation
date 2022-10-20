#!/bin/bash
CUDA_VISIBLE_DEVICES='-1' python algo.py -e CartPole-v1 -a ppo -i 1.0 0.1 -l 1.0 20.0 0.1 20.0 -t dtr
CUDA_VISIBLE_DEVICES='-1' python algo.py -e CartPole-v1 -a a2c -i 1.0 0.1 -l 1.0 20.0 0.1 20.0 -t dtr
CUDA_VISIBLE_DEVICES='-1' python algo.py -e CartPole-v1 -a dqn -i 1.0 0.1 -l 1.0 20.0 0.1 20.0 -t dtr
CUDA_VISIBLE_DEVICES='-1' python algo.py -e LunarLander-v2 -a ppo -i -10.0 0.6 -l -20.0 0 0 5 -t dtr
CUDA_VISIBLE_DEVICES='-1' python algo.py -e LunarLander-v2 -a a2c -i -10.0 0.6 -l -20.0 0 0 5 -t dtr
CUDA_VISIBLE_DEVICES='-1' python algo.py -e LunarLander-v2 -a dqn -i -10.0 0.6 -l -20.0 0 0 5 -t dtr
