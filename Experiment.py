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
            print(agent.device)
        else:
            agent = ACAgent(env, n_states, n_actions=n_actions, learning_rate=config['alpha'], lamda=config['lamda'],
                            gamma=config['gamma'], step=config['step'], if_conv=config['if_conv'], bootstrapping=config['bootstrapping'], baseline=config['baseline'])

            print(agent.device)
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


## Architecture
# Plot = LearningCurvePlot(title = 'REINFORCE with Different Network Architecture')
# PlotNoError = LearningCurvePlotNoError(title = 'REINFORCE with Different Network Architecture')
# if_conv = [True, False]
# if_conv_name = ['Convolution', 'Full Connected']
# for act in range(len(if_conv)):
#     config = deepcopy(baseline_config) # we must use deepcope to avoid changing the value of original baseline config
#     config['if_conv'] = if_conv[act]
#     episodeReward = run(config)
#     mean_reward = np.mean(episodeReward, axis=0)
#     std_reward = np.std(episodeReward, axis=0)
#     fileName = 'arrayResults/REINFORCE_' + if_conv_name[act] + '.npy'
#     np.save(fileName, episodeReward)
#     Plot.add_curve(mean_reward, std_reward, label=r'{}'.format(if_conv_name[act]))
#     PlotNoError.add_curve(mean_reward, std_reward, label=r'{}'.format(if_conv_name[act]))
#     Plot.save("plotResults/Architecture.png")
#     PlotNoError.save("plotResults/ArchitectureNoError.png")
#
# Plot.save("plotResults/Architecture.png")
# PlotNoError.save("plotResults/ArchitectureNoError.png")

## REINFORCE Learning Rate
# Plot = LearningCurvePlot(title = 'REINFORCE with Different Learning Rates')
# PlotNoError = LearningCurvePlotNoError(title = 'REINFORCE with Different Learning Rates')
# LR = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
# LRName = ['1', '0,1', '0,01', '0,001', '0,0001', '0,00001']
# for act in range(len(LR)):
#     config = deepcopy(baseline_config) # we must use deepcope to avoid changing the value of original baseline config
#     config['alpha'] = LR[act]
#     episodeReward = run(config)
#     mean_reward = np.mean(episodeReward, axis=0)
#     std_reward = np.std(episodeReward, axis=0)
#     # print(mean_reward)
#     fileName = 'arrayResults/REINFORCE_LR=' + LRName[act] + '.npy'
#     np.save(fileName, episodeReward)
#     Plot.add_curve(mean_reward, std_reward, label=r'$\alpha=${}'.format(LR[act]))
#     PlotNoError.add_curve(mean_reward, std_reward, label=r'$\alpha=${}'.format(LR[act]))
#     Plot.save("plotResults/LearningRate.png")
#     PlotNoError.save("plotResults/LearningRateNoError.png")
#
# Plot.save("plotResults/LearningRate.png")
# PlotNoError.save("plotResults/LearningRateNoError.png")

## REINFORCE Entropy Coefficient
# Plot = LearningCurvePlot(title = 'REINFORCE with Different Entropy Regularization Coefficients')
# PlotNoError = LearningCurvePlotNoError(title = 'REINFORCE with Different Entropy Regularization Coefficients')
# regu = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0]
# reguName = ['1', '0,1', '0,01', '0,001', '0,0001', '0,00001', '0']
# for act in range(len(regu)):
#     config = deepcopy(baseline_config) # we must use deepcope to avoid changing the value of original baseline config
#     config['lamda'] = regu[act]
#     episodeReward = run(config)
#     mean_reward = np.mean(episodeReward, axis=0)
#     std_reward = np.std(episodeReward, axis=0)
#     # print(mean_reward)
#     fileName = 'arrayResults/REINFORCE_Lambda=' + reguName[act] + '.npy'
#     np.save(fileName, episodeReward)
#     Plot.add_curve(mean_reward, std_reward, label=r'$\lambda=${}'.format(regu[act]))
#     PlotNoError.add_curve(mean_reward, std_reward, label=r'$\lambda=${}'.format(regu[act]))
#     Plot.save("plotResults/Entropy.png")
#     PlotNoError.save("plotResults/EntropyNoError.png")
#
# Plot.save("plotResults/Entropy.png")
# PlotNoError.save("plotResults/EntropyNoError.png")


## ActorCritic Learning Rate

# Plot = LearningCurvePlot(title = 'Actor Critic with Different Learning Rates')
# PlotNoError = LearningCurvePlotNoError(title = 'Actor Critic with Different Learning Rates')
# LR = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
# LRName = ['1', '0,1', '0,01', '0,001', '0,0001', '0,00001']
# for act in range(len(LR)):
#     config = deepcopy(baseline_config) # we must use deepcope to avoid changing the value of original baseline config
#     config['alpha'] = LR[act]
#     config['type'] = 'A'
#     episodeReward = run(config)
#     mean_reward = np.mean(episodeReward, axis=0)
#     std_reward = np.std(episodeReward, axis=0)
#     # print(mean_reward)
#     fileName = 'arrayResults/AC_LR=' + LRName[act] + '.npy'
#     np.save(fileName, episodeReward)
#     Plot.add_curve(mean_reward, std_reward, label=r'$\alpha=${}'.format(LR[act]))
#     PlotNoError.add_curve(mean_reward, std_reward, label=r'$\alpha=${}'.format(LR[act]))
#     Plot.save("plotResults/ACLearningRate.png")
#     PlotNoError.save("plotResults/ACLearningRateNoError.png")
#
# Plot.save("plotResults/ACLearningRate.png")
# PlotNoError.save("plotResults/ACLearningRateNoError.png")


# # Actor Critic Step
# Plot = LearningCurvePlot(title = 'Actor Critic with Different Depths')
# PlotNoError = LearningCurvePlotNoError(title = 'Actor Critic with Different Depths')
# depth = [1, 2, 5, 10, 20, 100, 200]
# depthName = ['1', '2', '5', '10', '20', '100', '200']
# for act in range(len(depth)):
#     config = deepcopy(baseline_config) # we must use deepcope to avoid changing the value of original baseline config
#     config['step'] = depth[act]
#     config['type'] = 'A'
#     episodeReward = run(config)
#     mean_reward = np.mean(episodeReward, axis=0)
#     std_reward = np.std(episodeReward, axis=0)
#     # print(mean_reward)
#     fileName = 'arrayResults/AC_depth=' + depthName[act] + '.npy'
#     np.save(fileName, episodeReward)
#     Plot.add_curve(mean_reward, std_reward, label=r'depth = {}'.format(depth[act]))
#     PlotNoError.add_curve(mean_reward, std_reward, label=r'depth = {}'.format(depth[act]))
#     Plot.save("plotResults/ACDepth.png")
#     PlotNoError.save("plotResults/ACDepthNoError.png")
#
# Plot.save("plotResults/ACDepth.png")
# PlotNoError.save("plotResults/ACDepthNoError.png")

#Gamma Actor
Plot = LearningCurvePlot(title = 'Actor Critic with Different Discount Factors')
PlotNoError = LearningCurvePlotNoError(title = 'Actor Critic with Different Discount Factors')
depth = [0.1, 0.5, 0.8, 0.85, 0.9, 0.95, 0.99, 1]
for act in range(len(depth)):
    config = deepcopy(baseline_config) # we must use deepcope to avoid changing the value of original baseline config
    config['gamma'] = depth[act]
    config['type'] = 'A'
    episodeReward = run(config)
    mean_reward = np.mean(episodeReward, axis=0)
    std_reward = np.std(episodeReward, axis=0)
    # print(mean_reward)
    fileName = 'arrayResults/AC_gamma=' + str(depth[act]) + '.npy'
    np.save(fileName, episodeReward)
    Plot.add_curve(mean_reward, std_reward, label=r'$\gamma=${}'.format(depth[act]))
    PlotNoError.add_curve(mean_reward, std_reward, label=r'$\gamma=${}'.format(depth[act]))
    Plot.save("plotResults/ACGamma.png")
    PlotNoError.save("plotResults/ACGammaNoError.png")

Plot.save("plotResults/ACGamma.png")
PlotNoError.save("plotResults/ACGammaNoError.png")

#Gamma REINFROCE
Plot = LearningCurvePlot(title = 'REINFORCE with Different Discount Factors')
PlotNoError = LearningCurvePlotNoError(title = 'REINFORCE with Different Discount Factors')
depth = [0.1, 0.5, 0.8, 0.85, 0.9, 0.95, 0.99, 1]
for act in range(len(depth)):
    config = deepcopy(baseline_config) # we must use deepcope to avoid changing the value of original baseline config
    config['gamma'] = depth[act]
    episodeReward = run(config)
    mean_reward = np.mean(episodeReward, axis=0)
    std_reward = np.std(episodeReward, axis=0)
    # print(mean_reward)
    fileName = 'arrayResults/R_gamma=' + str(depth[act]) + '.npy'
    np.save(fileName, episodeReward)
    Plot.add_curve(mean_reward, std_reward, label=r'$\gamma=${}'.format(depth[act]))
    PlotNoError.add_curve(mean_reward, std_reward, label=r'$\gamma=${}'.format(depth[act]))
    Plot.save("plotResults/RGamma.png")
    PlotNoError.save("plotResults/RGammaNoError.png")

Plot.save("plotResults/RGamma.png")
PlotNoError.save("plotResults/RGammaNoError.png")

#Actor Critic Entropy Coefficient
Plot = LearningCurvePlot(title = 'Actor Critic with Different Entropy Regularization Coefficients')
PlotNoError = LearningCurvePlotNoError(title = 'Actor Critic with Different Entropy Regularization Coefficients')
regu = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0]
reguName = ['1', '0,1', '0,01', '0,001', '0,0001', '0,00001', '0']
for act in range(len(regu)):
    config = deepcopy(baseline_config) # we must use deepcope to avoid changing the value of original baseline config
    config['lamda'] = regu[act]
    config['type'] = 'A'
    episodeReward = run(config)
    mean_reward = np.mean(episodeReward, axis=0)
    std_reward = np.std(episodeReward, axis=0)
    # print(mean_reward)
    fileName = 'arrayResults/AC_Lambda=' + reguName[act] + '.npy'
    np.save(fileName, episodeReward)
    Plot.add_curve(mean_reward, std_reward, label=r'$\lambda=${}'.format(regu[act]))
    PlotNoError.add_curve(mean_reward, std_reward, label=r'$\lambda=${}'.format(regu[act]))
    Plot.save("plotResults/ACEntropy.png")
    PlotNoError.save("plotResults/ACEntropyNoError.png")

Plot.save("plotResults/ACEntropy.png")
PlotNoError.save("plotResults/ACEntropyNoError.png")

#Actor Critic bootstrap and baseline
Plot = LearningCurvePlot(title = 'Actor Critic with Different combinations of Bootstrap and Baseline')
PlotNoError = LearningCurvePlotNoError(title = 'Actor Critic with Different combinations of Bootstrap and Baseline')
bootstrap = [True, False]
baseline = [True, False]
reguName = ['True-True', 'True-False', 'False-True', 'False-False']
i = 0
for boot in bootstrap:
    for base in baseline:
        config = deepcopy(baseline_config) # we must use deepcope to avoid changing the value of original baseline config
        config['bootstrap'] = boot
        config['baseline'] = base
        config['type'] = 'A'
        episodeReward = run(config)
        mean_reward = np.mean(episodeReward, axis=0)
        std_reward = np.std(episodeReward, axis=0)
        # print(mean_reward)
        fileName = 'arrayResults/AC_bootbase=' + reguName[i] + '.npy'
        np.save(fileName, episodeReward)
        Plot.add_curve(mean_reward, std_reward, label=r'$\boot-base=${}'.format(boot, "-", base))
        PlotNoError.add_curve(mean_reward, std_reward, label=r'$\boot-base=${}'.format(boot, "-", base))
        Plot.save("plotResults/ACBootbase.png")
        PlotNoError.save("plotResults/ACBootbaseNoError.png")
        i += 1

Plot.save("plotResults/ACBootbase.png")
PlotNoError.save("plotResults/ACBootbaseNoError.png")
