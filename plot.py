# from util import LearningCurvePlot, LearningCurvePlotNoError
# import random
# import numpy as np
# import torch
# from torch import nn
# from catch import Catch
# import os
# import gymnasium as gym
# from copy import deepcopy
# from reinforce import REINFORCEAgent
# from ActorCritic import ACAgent
# import tqdm
#
#
# # Plot = LearningCurvePlot(title = 'Actor Critic with Different Environment Sizes')
# # PlotNoError = LearningCurvePlotNoError(title = 'Actor Critic with Different Environment Sizes')
# # i = 0
# # regu = [7, 8, 9, 10, 15, 25]
# # reguName = ['7x7', '8x8', '9x9', '10x10', '15x15', '25x25']
# # for act in regu:
# #     fileName = 'arrayResults/part2_size=' + reguName[i] + '.npy'
# #     episodeReward = np.load(fileName)
# #     mean_reward = np.mean(episodeReward, axis=0)
# #     std_reward = np.std(episodeReward, axis=0)
# #     Plot.add_curve(mean_reward, std_reward, label=r'size={}'.format(reguName[i]))
# #     PlotNoError.add_curve(mean_reward, std_reward, label=r'size={}'.format(reguName[i]))
# #     Plot.save("plotResults/part2_size.png")
# #     PlotNoError.save("plotResults/part2_sizeNoError.png")
# #     i += 1
# #
# # Plot.save("plotResults/part2_size.png")
# # PlotNoError.save("plotResults/part2_sizeNoError.png")
#
# # ##Speed change
# # print("Change the Speed")
# # Plot = LearningCurvePlot(title = 'Actor Critic with Different Environment Speeds')
# # PlotNoError = LearningCurvePlotNoError(title = 'Actor Critic with Different Environment Speeds')
# # i = 0
# # regu = [0.5, 0.75, 1.0, 1.25, 1.5, 2]
# # reguName = ['0.5', '0.75', '1', '1.25', '1.5', '2']
# # for act in regu:
# #     fileName = 'arrayResults/part2_speed=' + reguName[i] + '.npy'
# #     episodeReward = np.load(fileName)
# #     mean_reward = np.mean(episodeReward, axis=0)
# #     std_reward = np.std(episodeReward, axis=0)
# #     # print(mean_reward)
# #     np.save(fileName, episodeReward)
# #     Plot.add_curve(mean_reward, std_reward, label=r'speed={}'.format(reguName[i]))
# #     PlotNoError.add_curve(mean_reward, std_reward, label=r'speed={}'.format(reguName[i]))
# #     Plot.save("plotResults/part2_speed.png")
# #     PlotNoError.save("plotResults/part2_speedNoError.png")
# #     i+=1
# #
# # Plot.save("plotResults/part2_speed.png")
# # PlotNoError.save("plotResults/part2_speedNoError.png")
#
# print("Change the observation Type")
# Plot = LearningCurvePlot(title = 'Actor Critic with different observation types')
# PlotNoError = LearningCurvePlotNoError(title = 'Actor Critic with different observation types')
# speeds = [0.5, 1, 2]
#
#
# for act in speeds:
#     fileName = 'arrayResults/part2_types=vector,speed=' + str(act) + '.npy'
#     episodeReward = np.load(fileName)
#     mean_reward = np.mean(episodeReward, axis=0)
#     std_reward = np.std(episodeReward, axis=0)
#     Plot.add_curve(mean_reward, std_reward, label=r'vector, speed={}'.format(act))
#     PlotNoError.add_curve(mean_reward, std_reward, label=r'vector, speed={}'.format(act))
#
#
# for act in speeds:
#     fileName = 'arrayResults/part2_speed=' + str(act) + '.npy'
#     episodeReward = np.load(fileName)
#     mean_reward = np.mean(episodeReward, axis=0)
#     std_reward = np.std(episodeReward, axis=0)
#     Plot.add_curve(mean_reward, std_reward, label=r'pixel, speed={}'.format(act))
#     PlotNoError.add_curve(mean_reward, std_reward, label=r'pixel, speed={}'.format(act))
#
# Plot.save("plotResults/part2_types.png")
# PlotNoError.save("plotResults/part2_typesNoError.png")
#
# #Non square with speed change
# # Plot = LearningCurvePlot(title = 'Actor Critic with a Non-square Environment')
# # PlotNoError = LearningCurvePlotNoError(title = 'Actor Critic with a Non-square Environment')
# # fileName = 'arrayResults/part2_combination=' + '.npy'
# # episodeReward = np.load(fileName)
# # mean_reward = np.mean(episodeReward, axis=0)
# # std_reward = np.std(episodeReward, axis=0)
# # np.save(fileName, episodeReward)
# # Plot.add_curve(mean_reward, std_reward, label='size 14x7, with speed=0.5')
# # PlotNoError.add_curve(mean_reward, std_reward, label="size 14x7, with speed=0.5")
# #
# # fileName = 'arrayResults/part2_speed=0.5' + '.npy'
# # episodeReward = np.load(fileName)
# # mean_reward = np.mean(episodeReward, axis=0)
# # std_reward = np.std(episodeReward, axis=0)
# # np.save(fileName, episodeReward)
# # Plot.add_curve(mean_reward, std_reward, label='size 7x7, with speed=0.5')
# # PlotNoError.add_curve(mean_reward, std_reward, label="size 7x7, with speed=0.5")
# # Plot.save("plotResults/part2_combination.png")
# # PlotNoError.save("plotResults/part2_combinationNoError.png")


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
#
# best_agent_config = {"type": "A",                                   ## must be 'R' for "REINFORCE" or 'A' for "ActorCritic"
#                    "alpha": 0.001,                                 ## learning rate of Network
#                    "gamma": 0.85,                                 ## discount rate of Network
#                    "lamda": 0.1,                                 ## the strength of the entropy regularization term in the loss
#                    "device": "cpu",                               ## if you want to use GPU, on Apple Silicon, set it as "mps", else set it as "cuda"
#                    "layers": 1,                                   ## number of layers of q-network
#                    "neurons": 12,                                 ## number of neurons of network, must be int or list with length layers+1
#                    "activation": nn.ReLU(),                       ## activation method
#                    "initialization": nn.init.xavier_normal_,      ## initialization method
#                    "if_conv": False,                              ## whether to use convolutional layers, only effective for REINFORCE
#                    "step": 200,                                     ## depth of bootstrapping of ActorCritic
#                    "bootstrapping": True,                         ## control if use the bootstrapping technology
#                    "baseline": True,                              ## control if use the baseline subtraction technology
#                     # parameters below are related to the environment
#                    "rows": 7,
#                    "columns": 7,
#                    "speed": 1.0,
#                    "max_steps": 250,
#                    "max_misses": 10,
#                    "observation_type": 'pixel',  # 'vector'
#                    "seed": None,
#                    }


best_AC_config = {"type":  "A",                                   ## must be 'R' for "REINFORCE" or 'A' for "ActorCritic"
                   "alpha": 0.001,                                 ## learning rate of Network
                   "gamma": 0.5,                                 ## discount rate of Network
                   "lamda": 0.1,                                 ## the strength of the entropy regularization term in the loss
                   "device": "cpu",                               ## if you want to use GPU, on Apple Silicon, set it as "mps", else set it as "cuda"
                   "if_conv": False,                              ## whether to use convolutional layers, only effective for REINFORCE
                   "step": 10,                                     ## depth of bootstrapping of ActorCritic
                   "bootstrapping": True,                         ## control if use the bootstrapping technology
                   "baseline": True,                              ## control if use the baseline subtraction technology
                   "ClipPPO": False,
                    # parameters below are related to the environment
                   "rows": 7,
                   "columns": 7,
                   "speed": 1.0,
                   "max_steps": 250,
                   "max_misses": 10,
                   "observation_type": 'pixel',  # 'vector'
                   "seed": None,
                   }


best_R_config = {"type":  "R",                                   ## must be 'R' for "REINFORCE" or 'A' for "ActorCritic"
                   "alpha": 0.001,                                 ## learning rate of Network
                   "gamma": 0.5,                                 ## discount rate of Network
                   "lamda": 0.1,                                 ## the strength of the entropy regularization term in the loss
                   "device": "cpu",                               ## if you want to use GPU, on Apple Silicon, set it as "mps", else set it as "cuda"
                   "if_conv": False,                              ## whether to use convolutional layers, only effective for REINFORCE
                   "step": 10,                                     ## depth of bootstrapping of ActorCritic
                   "bootstrapping": True,                         ## control if use the bootstrapping technology
                   "baseline": True,                              ## control if use the baseline subtraction technology
                   "ClipPPO": False,
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
                                   if_conv=config['if_conv'], ClipPPO=config['ClipPPO'])
        else:
            agent = ACAgent(env, n_states, n_actions=n_actions, learning_rate=config['alpha'], lamda=config['lamda'],
                            gamma=config['gamma'], step=config['step'], if_conv=config['if_conv'], bootstrapping=config['bootstrapping'], baseline=config['baseline'], ClipPPO=config['ClipPPO'])

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


# Architecture
Plot = LearningCurvePlot(title = 'REINFORCE with Different Network Architecture')
PlotNoError = LearningCurvePlotNoError(title = 'REINFORCE with Different Network Architecture')
if_conv = [True, False]
if_conv_name = ['Convolution', 'Full Connected']
if_conv_name_two = ['Convolution Layers', 'Full Connected Layers']
for act in range(len(if_conv)):
    fileName = 'arrayResults/REINFORCE_' + if_conv_name[act] + '.npy'
    episodeReward = np.load(fileName)
    index = np.random.choice(50, 10, replace=False)
    episodeReward = episodeReward[index, :]
    mean_reward = np.mean(episodeReward, axis=0)
    std_reward = np.std(episodeReward, axis=0)
    Plot.add_curve(mean_reward, std_reward, label=r'{}'.format(if_conv_name_two[act]))
    PlotNoError.add_curve(mean_reward, std_reward, label=r'{}'.format(if_conv_name_two[act]))
    Plot.save("plotResults/Architecture.png")
    PlotNoError.save("plotResults/ArchitectureNoError.png")

Plot.save("plotResults/Architecture.png")
PlotNoError.save("plotResults/ArchitectureNoError.png")

# REINFORCE Learning Rate
Plot = LearningCurvePlot(title = 'REINFORCE with Different Learning Rates')
PlotNoError = LearningCurvePlotNoError(title = 'REINFORCE with Different Learning Rates')
LR = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
LRName = ['1', '0,1', '0,01', '0,001', '0,0001', '0,00001']
for act in range(len(LR)):
    fileName = 'arrayResults/REINFORCE_LR=' + LRName[act] + '.npy'
    episodeReward = np.load(fileName)
    index = np.random.choice(50, 10, replace=False)
    episodeReward = episodeReward[index, :]
    mean_reward = np.mean(episodeReward, axis=0)
    std_reward = np.std(episodeReward, axis=0)
    # print(mean_reward)

    Plot.add_curve(mean_reward, std_reward, label=r'$\alpha=${}'.format(LR[act]))
    PlotNoError.add_curve(mean_reward, std_reward, label=r'$\alpha=${}'.format(LR[act]))
    Plot.save("plotResults/LearningRate.png")
    PlotNoError.save("plotResults/LearningRateNoError.png")

Plot.save("plotResults/LearningRate.png")
PlotNoError.save("plotResults/LearningRateNoError.png")

# REINFORCE Entropy Coefficient
Plot = LearningCurvePlot(title = 'REINFORCE with Different Entropy Regularization Coefficients')
PlotNoError = LearningCurvePlotNoError(title = 'REINFORCE with Different Entropy Regularization Coefficients')
regu = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0]
reguName = ['1', '0,1', '0,01', '0,001', '0,0001', '0,00001', '0']
for act in range(len(regu)):
    fileName = 'arrayResults/REINFORCE_Lambda=' + reguName[act] + '.npy'
    episodeReward = np.load(fileName)
    index = np.random.choice(50, 10, replace=False)
    episodeReward = episodeReward[index, :]
    mean_reward = np.mean(episodeReward, axis=0)
    std_reward = np.std(episodeReward, axis=0)
    # print(mean_reward)
    Plot.add_curve(mean_reward, std_reward, label=r'$\lambda=${}'.format(regu[act]))
    PlotNoError.add_curve(mean_reward, std_reward, label=r'$\lambda=${}'.format(regu[act]))
    Plot.save("plotResults/Entropy.png")
    PlotNoError.save("plotResults/EntropyNoError.png")

Plot.save("plotResults/Entropy.png")
PlotNoError.save("plotResults/EntropyNoError.png")


#Gamma REINFROCE
print("REINFROCE Gamma Experiments")
Plot = LearningCurvePlot(title = 'REINFORCE with Different Discount Factors')
PlotNoError = LearningCurvePlotNoError(title = 'REINFORCE with Different Discount Factors')
depth = [0.1, 0.5, 0.8, 0.85, 0.9, 0.95, 0.99, 1]
for act in range(len(depth)):
    fileName = 'arrayResults/R_gamma=' + str(depth[act]) + '.npy'
    episodeReward = np.load(fileName)
    index = np.random.choice(50, 10, replace=False)
    episodeReward = episodeReward[index, :]
    mean_reward = np.mean(episodeReward, axis=0)
    std_reward = np.std(episodeReward, axis=0)
    # print(mean_reward)
    Plot.add_curve(mean_reward, std_reward, label=r'$\gamma=${}'.format(depth[act]))
    PlotNoError.add_curve(mean_reward, std_reward, label=r'$\gamma=${}'.format(depth[act]))
    Plot.save("plotResults/RGamma.png")
    PlotNoError.save("plotResults/RGammaNoError.png")

Plot.save("plotResults/RGamma.png")
PlotNoError.save("plotResults/RGammaNoError.png")

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

# #Gamma Actor
# print("Actor Critic Gamma Experiments")
# Plot = LearningCurvePlot(title = 'Actor Critic with Different Discount Factors')
# PlotNoError = LearningCurvePlotNoError(title = 'Actor Critic with Different Discount Factors')
# depth = [0.1, 0.5, 0.8, 0.85, 0.9, 0.95, 0.99, 1]
# for act in range(len(depth)):
#     config = deepcopy(baseline_config) # we must use deepcope to avoid changing the value of original baseline config
#     config['gamma'] = depth[act]
#     config['type'] = 'A'
#     episodeReward = run(config)
#     mean_reward = np.mean(episodeReward, axis=0)
#     std_reward = np.std(episodeReward, axis=0)
#     # print(mean_reward)
#     fileName = 'arrayResults/AC_gamma=' + str(depth[act]) + '.npy'
#     np.save(fileName, episodeReward)
#     Plot.add_curve(mean_reward, std_reward, label=r'$\gamma=${}'.format(depth[act]))
#     PlotNoError.add_curve(mean_reward, std_reward, label=r'$\gamma=${}'.format(depth[act]))
#     Plot.save("plotResults/ACGamma.png")
#     PlotNoError.save("plotResults/ACGammaNoError.png")
#
# Plot.save("plotResults/ACGamma.png")
# PlotNoError.save("plotResults/ACGammaNoError.png")
#


# #Actor Critic Entropy Coefficient
# print("Actor Critic Entropy Coefficients Experiments")
# Plot = LearningCurvePlot(title = 'Actor Critic with Different Entropy Regularization Coefficients')
# PlotNoError = LearningCurvePlotNoError(title = 'Actor Critic with Different Entropy Regularization Coefficients')
# regu = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0]
# reguName = ['1', '0,1', '0,01', '0,001', '0,0001', '0,00001', '0']
# for act in range(len(regu)):
#     config = deepcopy(baseline_config) # we must use deepcope to avoid changing the value of original baseline config
#     config['lamda'] = regu[act]
#     config['type'] = 'A'
#     episodeReward = run(config)
#     mean_reward = np.mean(episodeReward, axis=0)
#     std_reward = np.std(episodeReward, axis=0)
#     # print(mean_reward)
#     fileName = 'arrayResults/AC_Lambda=' + reguName[act] + '.npy'
#     np.save(fileName, episodeReward)
#     Plot.add_curve(mean_reward, std_reward, label=r'$\lambda=${}'.format(regu[act]))
#     PlotNoError.add_curve(mean_reward, std_reward, label=r'$\lambda=${}'.format(regu[act]))
#     Plot.save("plotResults/ACEntropy.png")
#     PlotNoError.save("plotResults/ACEntropyNoError.png")
#
# Plot.save("plotResults/ACEntropy.png")
# PlotNoError.save("plotResults/ACEntropyNoError.png")

# # Actor Critic bootstrap and baseline
# print("Actor Critic Bootstrap and Baseline Experiments")
# Plot = LearningCurvePlot(title = 'Actor Critic with Different combinations of Bootstrap and Baseline')
# PlotNoError = LearningCurvePlotNoError(title = 'Actor Critic with Different combinations of Bootstrap and Baseline')
# bootstrap = [True, False]
# baseline = [True, False]
# reguName = ['True-True', 'True-False', 'False-True', 'False-False']
# i = 0
# for boot in bootstrap:
#     for base in baseline:
#         config = deepcopy(baseline_config) # we must use deepcope to avoid changing the value of original baseline config
#         config['step'] = 10
#         config['bootstrap'] = boot
#         config['baseline'] = base
#         config['type'] = 'A'
#         episodeReward = run(config)
#         mean_reward = np.mean(episodeReward, axis=0)
#         std_reward = np.std(episodeReward, axis=0)
#         # print(mean_reward)
#         fileName = 'arrayResults/AC_bootbase=' + reguName[i] + '.npy'
#         np.save(fileName, episodeReward)
#         Plot.add_curve(mean_reward, std_reward, label=r'boot-base={}'.format(reguName[i]))
#         PlotNoError.add_curve(mean_reward, std_reward, label=r'boot-base={}'.format(reguName[i]))
#         Plot.save("plotResults/ACBootbase.png")
#         PlotNoError.save("plotResults/ACBootbaseNoError.png")
#         i += 1
#
# Plot.save("plotResults/ACBootbase.png")
# PlotNoError.save("plotResults/ACBootbaseNoError.png")

# Clip-PPO and Non Clip-PPO



# Plot = LearningCurvePlot(title = 'Comparison between Actor-Critic and REINFORCE with Optimal Parameters')
# PlotNoError = LearningCurvePlotNoError(title = 'Comparison between Actor-Critic and REINFORCE with Optimal Parameters')
#
# # print("BEST AC WITHOUT CLIPPPO")
# # config = deepcopy(best_AC_config) # we must use deepcope to avoid changing the value of original baseline config
# # episodeReward = run(config)
# # mean_reward = np.mean(episodeReward, axis=0)
# # std_reward = np.std(episodeReward, axis=0)
# # fileName = 'arrayResults/BestACNoClipPPO' + '.npy'
# # np.save(fileName, episodeReward)
# # Plot.add_curve(mean_reward, std_reward, label=r'Actor-Critic (without Clip-PPO)')
# # PlotNoError.add_curve(mean_reward, std_reward, label=r'Actor-Critic (without Clip-PPO)')
# # Plot.save("plotResults/BestModel.png")
# # PlotNoError.save("plotResults/BestModelNoError.png")
#
# print("BEST AC WITH CLIPPPO")
# config = deepcopy(best_AC_config) # we must use deepcope to avoid changing the value of original baseline config
# config['ClipPPO'] = True
# episodeReward = run(config)
# mean_reward = np.mean(episodeReward, axis=0)
# std_reward = np.std(episodeReward, axis=0)
# fileName = 'arrayResults/BestACClipPPO' + '.npy'
# np.save(fileName, episodeReward)
# Plot.add_curve(mean_reward, std_reward, label=r'Actor-Critic (with Clip-PPO)')
# PlotNoError.add_curve(mean_reward, std_reward, label=r'Actor-Critic (with Clip-PPO)')
# Plot.save("plotResults/BestModel.png")
# PlotNoError.save("plotResults/BestModelNoError.png")
#
# print("BEST R WITHOUT CLIPPPO")
# config = deepcopy(best_R_config) # we must use deepcope to avoid changing the value of original baseline config
# episodeReward = run(config)
# mean_reward = np.mean(episodeReward, axis=0)
# std_reward = np.std(episodeReward, axis=0)
# fileName = 'arrayResults/BestREINFORCENoClipPPO' + '.npy'
# np.save(fileName, episodeReward)
# Plot.add_curve(mean_reward, std_reward, label=r'REINFORCE (without Clip-PPO)')
# PlotNoError.add_curve(mean_reward, std_reward, label=r'REINFORCE (without Clip-PPO)')
# Plot.save("plotResults/BestModel.png")
# PlotNoError.save("plotResults/BestModelNoError.png")
#
# print("BEST R WITH CLIPPPO")
# config = deepcopy(best_R_config) # we must use deepcope to avoid changing the value of original baseline config
# config['ClipPPO'] = True
# episodeReward = run(config)
# mean_reward = np.mean(episodeReward, axis=0)
# std_reward = np.std(episodeReward, axis=0)
# fileName = 'arrayResults/BestREINFORCEClipPPO' + '.npy'
# np.save(fileName, episodeReward)
# Plot.add_curve(mean_reward, std_reward, label=r'REINFORCE (with Clip-PPO)')
# PlotNoError.add_curve(mean_reward, std_reward, label=r'REINFORCE (with Clip-PPO)')
# Plot.save("plotResults/BestModel.png")
# PlotNoError.save("plotResults/BestModelNoError.png")




# ##Environment Experiments
# #Rows
# Plot = LearningCurvePlot(title = 'Actor Critic with different environment sizes')
# PlotNoError = LearningCurvePlotNoError(title = 'Actor Critic with different environment sizes')
# i = 0
# regu = [7, 8, 9, 10, 15, 25]
# reguName = ['7x7', '8x8', '9x9', '10x10', '15x15', '25x25']
# for act in regu:
#     config = deepcopy(best_agent_config) # we must use deepcope to avoid changing the value of original baseline config
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
#
# #Speed change
# Plot = LearningCurvePlot(title = 'Actor Critic with different environment speeds')
# PlotNoError = LearningCurvePlotNoError(title = 'Actor Critic with different environment speeds')
# i = 0
# regu = [0.5, 0.75, 1.0, 1.25, 1.5, 2]
# reguName = ['0.5', '0.75', '1', '1.25', '1.5', '2']
# for act in regu:
#     config = deepcopy(best_agent_config) # we must use deepcope to avoid changing the value of original baseline config
#     config['speed'] = act
#     episodeReward = run(config)
#     mean_reward = np.mean(episodeReward, axis=0)
#     std_reward = np.std(episodeReward, axis=0)
#     # print(mean_reward)
#     fileName = 'arrayResults/part2_speed=' + reguName[i] + '.npy'
#     np.save(fileName, episodeReward)
#     Plot.add_curve(mean_reward, std_reward, label=r'speed={}'.format(reguName[i]))
#     PlotNoError.add_curve(mean_reward, std_reward, label=r'speed={}'.format(reguName[i]))
#     Plot.save("plotResults/part2_speed.png")
#     PlotNoError.save("plotResults/part2_speedNoError.png")
#     i+=1
#
# Plot.save("plotResults/part2_speed.png")
# PlotNoError.save("plotResults/part2_speedNoError.png")
#
# #Change to observation type vector with speed 1 and speed >1
# Plot = LearningCurvePlot(title = 'Actor Critic with different observation types')
# PlotNoError = LearningCurvePlotNoError(title = 'Actor Critic with different observation types')
# i = 0
#
# types = ['pixel', 'vector']
# for act in types:
#     config = deepcopy(best_agent_config) # we must use deepcope to avoid changing the value of original baseline config
#     config['observation_type'] = act
#     episodeReward = run(config)
#     mean_reward = np.mean(episodeReward, axis=0)
#     std_reward = np.std(episodeReward, axis=0)
#     # print(mean_reward)
#     fileName = 'arrayResults/part2_types=' + reguName[i] + '.npy'
#     np.save(fileName, episodeReward)
#     Plot.add_curve(mean_reward, std_reward, label=r'type={}'.format(act))
#     PlotNoError.add_curve(mean_reward, std_reward, label=r'type={}'.format(act))
#     Plot.save("plotResults/part2_types.png")
#     PlotNoError.save("plotResults/part2_typesNoError.png")
#     i+=1
#
# Plot.save("plotResults/part2_types.png")
# PlotNoError.save("plotResults/part2_typesNoError.png")
#
#
#
# #Non square with speed change
# Plot = LearningCurvePlot(title = 'Actor Critic with a non-square environment')
# PlotNoError = LearningCurvePlotNoError(title = 'Actor Critic with a non-square environment')
#
# config = deepcopy(best_agent_config) # we must use deepcope to avoid changing the value of original baseline config
# config['speed'] = 0.5
# config['rows'] = 14
# config['columns'] = 7
# episodeReward = run(config)
# mean_reward = np.mean(episodeReward, axis=0)
# std_reward = np.std(episodeReward, axis=0)
# # print(mean_reward)
# fileName = 'arrayResults/part2_combination=' + reguName[i] + '.npy'
# np.save(fileName, episodeReward)
# Plot.add_curve(mean_reward, std_reward, label='14x7, with speed=0.5')
# PlotNoError.add_curve(mean_reward, std_reward, label="14x7, with speed=0.5")
# Plot.save("plotResults/part2_combination.png")
# PlotNoError.save("plotResults/part2_combination.png")

