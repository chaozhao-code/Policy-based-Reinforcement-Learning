from util import LearningCurvePlot, LearningCurvePlotNoError
import random
import numpy as np
import torch
from torch import nn
from catch import Catch
import os
import gymnasium as gym
from copy import deepcopy
from reinforce import REINFORCEAgent
from ActorCritic import ACAgent
import tqdm

REPETITION = 1
EPISODE = 5000

baseline_config = {"type": "R",                                   ## must be 'R' for "REINFORCE" or 'A' for "ActorCritic"
                   "alpha": 0.001,                                 ## learning rate of Network
                   "gamma": 0.99,                                 ## discount rate of Network
                   "lamda": 0.01,                                 ## the strength of the entropy regularization term in the loss
                   "device": "cpu",                               ## if you want to use GPU, on Apple Silicon, set it as "mps", else set it as "cuda"
                   "layers": 1,                                   ## number of layers of q-network
                   "neurons": 12,                                 ## number of neurons of network, must be int or list with length layers+1
                   "activation": nn.ReLU(),                       ## activation method
                   "initialization": nn.init.xavier_normal_,      ## initialization method
                   "if_conv": False,                              ## whether to use convolutional layers, only effective for REINFORCE
                   "step": 1,                                     ## depth of bootstrapping of ActorCritic
                   "bootstrapping": True,                         ## control if use the bootstrapping technology
                   "baseline": True,                              ## control if use the baseline subtraction technology
                    # parameters below are related to the environment
                   "rows": 7,
                   "columns": 7,
                   "speed": 1.0,
                   "max_steps": 250,
                   "max_misses": 10,
                   "observation_type": 'pixel',  # 'vector'
                   "seed": None,
                   }

def run(config):

    env = Catch(rows=config['rows'], columns=config['columns'], speed=config['speed'], max_steps=config['max_steps'],
                max_misses=config['max_misses'], observation_type=config['observation_type'], seed=config['seed'])
    if config['observation_type'] == 'pixel':
        n_states = config['rows'] * config['columns'] * 2
    else:
        n_states = 3
    n_actions = env.action_space.n

    interrupt = []

    # if config['type'] == 'R':
    #     agent = REINFORCEAgent(env, n_states, n_actions=n_actions, learning_rate=config['alpha'], lamda=config['lamda'], gamma=config['gamma'], step=config['step'], if_conv=config['if_conv'])
    # else:
    #     agent = ACAgent(env, n_states, n_actions=n_actions, learning_rate=config['alpha'], lamda=config['lamda'], gamma=config['gamma'], step=config['step'], if_conv=config['if_conv'])
    #
    episodeReward = np.zeros((REPETITION, EPISODE))
    for i in tqdm.trange(REPETITION):
        if config['type'] == 'R':
            agent = REINFORCEAgent(env, n_states, n_actions=n_actions, learning_rate=config['alpha'],
                                   lamda=config['lamda'], gamma=config['gamma'], step=config['step'],
                                   if_conv=config['if_conv'])
        else:
            agent = ACAgent(env, n_states, n_actions=n_actions, learning_rate=config['alpha'], lamda=config['lamda'],
                            gamma=config['gamma'], step=config['step'], if_conv=config['if_conv'], bootstrapping=config['bootstrapping'], baseline=config['baseline'])

        for j in range(EPISODE):
            try:
                episodeReward[i, j] = agent.train(j)
            except:
                interrupt.append(j)
                break

    if len(interrupt) != 0:
        minEpisode = min(interrupt)
        return episodeReward[:, :minEpisode]

    return episodeReward

# Actor Critic Step

depth = [100, 200]
depthName = ['100', '200']
for act in range(len(depth)):
    config = deepcopy(baseline_config) # we must use deepcope to avoid changing the value of original baseline config
    config['step'] = depth[act]
    config['type'] = 'A'
    episodeReward = run(config)
    mean_reward = np.mean(episodeReward, axis=0)
    std_reward = np.std(episodeReward, axis=0)
    # print(mean_reward)
    # fileName = 'arrayResults/AC_depth=' + depthName[act] + '.npy'
    # np.save(fileName, episodeReward)


