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


# Plot = LearningCurvePlot(title = 'Actor Critic with Different Environment Sizes')
# PlotNoError = LearningCurvePlotNoError(title = 'Actor Critic with Different Environment Sizes')
# i = 0
# regu = [7, 8, 9, 10, 15, 25]
# reguName = ['7x7', '8x8', '9x9', '10x10', '15x15', '25x25']
# for act in regu:
#     fileName = 'arrayResults/part2_size=' + reguName[i] + '.npy'
#     episodeReward = np.load(fileName)
#     mean_reward = np.mean(episodeReward, axis=0)
#     std_reward = np.std(episodeReward, axis=0)
#     Plot.add_curve(mean_reward, std_reward, label=r'size={}'.format(reguName[i]))
#     PlotNoError.add_curve(mean_reward, std_reward, label=r'size={}'.format(reguName[i]))
#     Plot.save("plotResults/part2_size.png")
#     PlotNoError.save("plotResults/part2_sizeNoError.png")
#     i += 1
#
# Plot.save("plotResults/part2_size.png")
# PlotNoError.save("plotResults/part2_sizeNoError.png")

# ##Speed change
# print("Change the Speed")
# Plot = LearningCurvePlot(title = 'Actor Critic with Different Environment Speeds')
# PlotNoError = LearningCurvePlotNoError(title = 'Actor Critic with Different Environment Speeds')
# i = 0
# regu = [0.5, 0.75, 1.0, 1.25, 1.5, 2]
# reguName = ['0.5', '0.75', '1', '1.25', '1.5', '2']
# for act in regu:
#     fileName = 'arrayResults/part2_speed=' + reguName[i] + '.npy'
#     episodeReward = np.load(fileName)
#     mean_reward = np.mean(episodeReward, axis=0)
#     std_reward = np.std(episodeReward, axis=0)
#     # print(mean_reward)
#     np.save(fileName, episodeReward)
#     Plot.add_curve(mean_reward, std_reward, label=r'speed={}'.format(reguName[i]))
#     PlotNoError.add_curve(mean_reward, std_reward, label=r'speed={}'.format(reguName[i]))
#     Plot.save("plotResults/part2_speed.png")
#     PlotNoError.save("plotResults/part2_speedNoError.png")
#     i+=1
#
# Plot.save("plotResults/part2_speed.png")
# PlotNoError.save("plotResults/part2_speedNoError.png")

print("Change the observation Type")
Plot = LearningCurvePlot(title = 'Actor Critic with different observation types')
PlotNoError = LearningCurvePlotNoError(title = 'Actor Critic with different observation types')
speeds = [0.5, 1, 2]


for act in speeds:
    fileName = 'arrayResults/part2_types=vector,speed=' + str(act) + '.npy'
    episodeReward = np.load(fileName)
    mean_reward = np.mean(episodeReward, axis=0)
    std_reward = np.std(episodeReward, axis=0)
    Plot.add_curve(mean_reward, std_reward, label=r'vector, speed={}'.format(act))
    PlotNoError.add_curve(mean_reward, std_reward, label=r'vector, speed={}'.format(act))


for act in speeds:
    fileName = 'arrayResults/part2_speed=' + str(act) + '.npy'
    episodeReward = np.load(fileName)
    mean_reward = np.mean(episodeReward, axis=0)
    std_reward = np.std(episodeReward, axis=0)
    Plot.add_curve(mean_reward, std_reward, label=r'pixel, speed={}'.format(act))
    PlotNoError.add_curve(mean_reward, std_reward, label=r'pixel, speed={}'.format(act))

Plot.save("plotResults/part2_types.png")
PlotNoError.save("plotResults/part2_typesNoError.png")

#Non square with speed change
# Plot = LearningCurvePlot(title = 'Actor Critic with a Non-square Environment')
# PlotNoError = LearningCurvePlotNoError(title = 'Actor Critic with a Non-square Environment')
# fileName = 'arrayResults/part2_combination=' + '.npy'
# episodeReward = np.load(fileName)
# mean_reward = np.mean(episodeReward, axis=0)
# std_reward = np.std(episodeReward, axis=0)
# np.save(fileName, episodeReward)
# Plot.add_curve(mean_reward, std_reward, label='size 14x7, with speed=0.5')
# PlotNoError.add_curve(mean_reward, std_reward, label="size 14x7, with speed=0.5")
#
# fileName = 'arrayResults/part2_speed=0.5' + '.npy'
# episodeReward = np.load(fileName)
# mean_reward = np.mean(episodeReward, axis=0)
# std_reward = np.std(episodeReward, axis=0)
# np.save(fileName, episodeReward)
# Plot.add_curve(mean_reward, std_reward, label='size 7x7, with speed=0.5')
# PlotNoError.add_curve(mean_reward, std_reward, label="size 7x7, with speed=0.5")
# Plot.save("plotResults/part2_combination.png")
# PlotNoError.save("plotResults/part2_combinationNoError.png")