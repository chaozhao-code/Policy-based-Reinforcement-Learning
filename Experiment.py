from util import LearningCurvePlot, LearningCurvePlotNoError
import random
import numpy as np
import torch
from torch import nn
from catch import Catch
import os
import gymnasium as gym
from copy import deepcopy
from REINFORCE import REINFORCEAgent
from ActorCritic import ACAgent
import tqdm

REPETITION = 50
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
                            gamma=config['gamma'], step=config['step'], if_conv=config['if_conv'])
        for j in range(EPISODE):
            episodeReward[i, j] = agent.train(j)

    return episodeReward


## Tune Activation Function
Plot = LearningCurvePlot(title = 'REINFORCE with Different Network Architecture')
PlotNoError = LearningCurvePlotNoError(title = 'REINFORCE with Different Network Architecture')
if_conv = [True, False]
if_conv_name = ['Convolution', 'Full Connected']
for act in range(len(if_conv)):
    config = deepcopy(baseline_config) # we must use deepcope to avoid changing the value of original baseline config
    config['if_conv'] = if_conv[act]
    episodeReward = run(config)
    mean_reward = np.mean(episodeReward, axis=0)
    std_reward = np.std(episodeReward, axis=0)
    fileName = 'arrayResults/REINFORCE_' + if_conv_name[act] + '.npy'
    np.save(fileName, episodeReward)
    Plot.add_curve(mean_reward, std_reward, label=r'{}'.format(if_conv_name[act]))
    PlotNoError.add_curve(mean_reward, std_reward, label=r'{}'.format(if_conv_name[act]))
    Plot.save("plotResults/Architecture.png")
    PlotNoError.save("plotResults/ArchitectureNoError.png")

Plot.save("plotResults/Architecture.png")
PlotNoError.save("plotResults/ArchitectureNoError.png")
