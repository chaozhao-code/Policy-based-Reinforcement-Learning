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

REPETITION = 10
EPISODE = 5000

best_AC_config = {"type":  "A",                                   ## must be 'R' for "REINFORCE" or 'A' for "ActorCritic"
                   "alpha": 0.001,                                 ## learning rate of Network
                   "gamma": 0.5,                                 ## discount rate of Network
                   "lamda": 0.1,                                 ## the strength of the entropy regularization term in the loss
                   "device": "cpu",                               ## if you want to use GPU, on Apple Silicon, set it as "mps", else set it as "cuda"
                   "if_conv": False,                              ## whether to use convolutional layers, only effective for REINFORCE
                   "step": 10,                                     ## depth of bootstrapping of ActorCritic
                   "bootstrapping": True,                         ## control if use the bootstrapping technology
                   "baseline": True,                              ## control if use the baseline subtraction technology
                   "ClipPPO": True,
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

# Plot = LearningCurvePlot(title = 'Comparison between Actor-Critic and REINFORCE with Optimal Parameters')
# PlotNoError = LearningCurvePlotNoError(title = 'Comparison between Actor-Critic and REINFORCE with Optimal Parameters')
#
# fileName = 'arrayResults/BestACNoClipPPO' + '.npy'
# episodeReward = np.load(fileName)
# mean_reward = np.mean(episodeReward, axis=0)
# std_reward = np.std(episodeReward, axis=0)
#
# Plot.add_curve(mean_reward, std_reward, label=r'Actor-Critic (without Clip-PPO)')
# PlotNoError.add_curve(mean_reward, std_reward, label=r'Actor-Critic (without Clip-PPO)')
#
#
#
# fileName = 'arrayResults/BestACClipPPO' + '.npy'
# episodeReward = np.load(fileName)
# mean_reward = np.mean(episodeReward, axis=0)
# std_reward = np.std(episodeReward, axis=0)
#
# Plot.add_curve(mean_reward, std_reward, label=r'Actor-Critic (with Clip-PPO)')
# PlotNoError.add_curve(mean_reward, std_reward, label=r'Actor-Critic (with Clip-PPO)')
#
# fileName = 'arrayResults/BestREINFORCENoClipPPO' + '.npy'
# episodeReward = np.load(fileName)
# mean_reward = np.mean(episodeReward, axis=0)
# std_reward = np.std(episodeReward, axis=0)
# Plot.add_curve(mean_reward, std_reward, label=r'REINFORCE (without Clip-PPO)')
# PlotNoError.add_curve(mean_reward, std_reward, label=r'REINFORCE (without Clip-PPO)')
#
#
# fileName = 'arrayResults/BestREINFORCEClipPPO' + '.npy'
# episodeReward = np.load(fileName)
# mean_reward = np.mean(episodeReward, axis=0)
# std_reward = np.std(episodeReward, axis=0)
#
#
# Plot.add_curve(mean_reward, std_reward, label=r'REINFORCE (with Clip-PPO)')
# PlotNoError.add_curve(mean_reward, std_reward, label=r'REINFORCE (with Clip-PPO)')
#
#
#
# Plot.save("plotResults/BestModel.png")
# PlotNoError.save("plotResults/BestModelNoError.png")







#Environment Experiments
#Rows
# print("Change The number of Rows")
# Plot = LearningCurvePlot(title = 'Actor Critic with different environment sizes')
# PlotNoError = LearningCurvePlotNoError(title = 'Actor Critic with different environment sizes')
# i = 0
# regu = [7, 8, 9, 10, 15, 25]
# reguName = ['7x7', '8x8', '9x9', '10x10', '15x15', '25x25']
# for act in regu:
#     config = deepcopy(best_AC_config) # we must use deepcope to avoid changing the value of original baseline config
#     config["rows"] = act
#     config["columns"] = act
#     episodeReward = run(config)
#     mean_reward = np.mean(episodeReward, axis=0)
#     std_reward = np.std(episodeReward, axis=0)
#     # print(mean_reward)
#     fileName = 'arrayResults/part2_size=' + reguName[i] + '.npy'
#     np.save(fileName, episodeReward)
#     Plot.add_curve(mean_reward, std_reward, label=r'size={}'.format(reguName[i]))
#     PlotNoError.add_curve(mean_reward, std_reward, label=r'size={}'.format(reguName[i]))
#     Plot.save("plotResults/part2_size.png")
#     PlotNoError.save("plotResults/part2_sizeNoError.png")
#     i += 1
#
# Plot.save("plotResults/part2_size.png")
# PlotNoError.save("plotResults/part2_sizeNoError.png")

# config = deepcopy(best_AC_config) # we must use deepcope to avoid changing the value of original baseline config
# config['speed'] = 2
# episodeReward = run(config)

##Speed change
# print("Change the Speed")
# Plot = LearningCurvePlot(title = 'Actor Critic with different environment speeds')
# PlotNoError = LearningCurvePlotNoError(title = 'Actor Critic with different environment speeds')
# i = 0
# regu = [0.5, 0.75, 1.0, 1.25, 1.5, 2]
# reguName = ['0.5', '0.75', '1', '1.25', '1.5', '2']
# regu = [2]
# reguName = ['2']
# for act in regu:
#     config = deepcopy(best_AC_config) # we must use deepcope to avoid changing the value of original baseline config
#     config['speed'] = act
#     episodeReward = run(config)
#     mean_reward = np.mean(episodeReward, axis=0)
#     std_reward = np.std(episodeReward, axis=0)
#     # print(mean_reward)
#     fileName = 'arrayResults/part2_speed=' + reguName[i] + '.npy'
#     np.save(fileName, episodeReward)
#     # Plot.add_curve(mean_reward, std_reward, label=r'speed={}'.format(reguName[i]))
#     # PlotNoError.add_curve(mean_reward, std_reward, label=r'speed={}'.format(reguName[i]))
#     # Plot.save("plotResults/part2_speed.png")
#     # PlotNoError.save("plotResults/part2_speedNoError.png")
#     i+=1
#
# Plot.save("plotResults/part2_speed.png")
# PlotNoError.save("plotResults/part2_speedNoError.png")
#
#Change to observation type vector with speed 1 and speed >1

print("Change the observation Type")
# Plot = LearningCurvePlot(title = 'Actor Critic with different observation types')
# PlotNoError = LearningCurvePlotNoError(title = 'Actor Critic with different observation types')
# speeds = [0.5, 1, 2]
speeds = [0.5, 1, 2]


for act in speeds:
    config = deepcopy(best_AC_config) # we must use deepcope to avoid changing the value of original baseline config
    config['observation_type'] = 'vector'
    config['speed'] = act
    print("Speed: ", act)
    episodeReward = run(config)
    mean_reward = np.mean(episodeReward, axis=0)
    std_reward = np.std(episodeReward, axis=0)
    # print(mean_reward)
    fileName = 'arrayResults/part2_types=vector,speed=' + str(act) + '.npy'
    np.save(fileName, episodeReward)
    # Plot.add_curve(mean_reward, std_reward, label=r'type={}'.format(act))
    # PlotNoError.add_curve(mean_reward, std_reward, label=r'type={}'.format(act))
    # Plot.save("plotResults/part2_types.png")
    # PlotNoError.save("plotResults/part2_typesNoError.png")

# Plot.save("plotResults/part2_types.png")
# PlotNoError.save("plotResults/part2_typesNoError.png")
# #
# #
# #
#Non square with speed change
# Plot = LearningCurvePlot(title = 'Actor Critic with a non-square environment')
# PlotNoError = LearningCurvePlotNoError(title = 'Actor Critic with a non-square environment')
#
# config = deepcopy(best_AC_config) # we must use deepcope to avoid changing the value of original baseline config
# config['speed'] = 0.5
# config['rows'] = 14
# config['columns'] = 7
# episodeReward = run(config)
# mean_reward = np.mean(episodeReward, axis=0)
# std_reward = np.std(episodeReward, axis=0)
# # print(mean_reward)
# fileName = 'arrayResults/part2_combination=' + '.npy'
# np.save(fileName, episodeReward)
# Plot.add_curve(mean_reward, std_reward, label='14x7, with speed=0.5')
# PlotNoError.add_curve(mean_reward, std_reward, label="14x7, with speed=0.5")
# Plot.save("plotResults/part2_combination.png")
# PlotNoError.save("plotResults/part2_combination.png")