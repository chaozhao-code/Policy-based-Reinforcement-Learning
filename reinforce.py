import sys

import torch
from torch import nn
import random
import numpy as np
import math
import sys
import time
from torchsummary import summary



class Policy(nn.Module):
    def __init__(self, n_states, n_actions, layers=1, neurons=128, if_conv=False, activation=nn.SiLU(), initialization=nn.init.xavier_uniform_):
        super(Policy, self).__init__()

        self.n_actions = n_actions
        self.initialization = initialization
        self.conv = if_conv
        modules = []
        if if_conv:
            # print("Convolution Duealing DQN can only be used for CartPole-v1 environment!")
            modules.append(nn.Conv2d(2, 32, kernel_size=(4, 4), stride=1))
            modules.append(activation)
            modules.append(nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2))
            modules.append(activation)
            modules.append(nn.Flatten())
            modules.append(nn.Linear(64, n_actions))
            modules.append(nn.Softmax(dim=1))


        else:
            if type(neurons) == int:
                modules.append(nn.Linear(n_states, neurons))
                modules.append(activation)
                for i in range(layers):
                    modules.append(nn.Linear(neurons, neurons))
                    modules.append(activation)
                modules.append(nn.Linear(neurons, n_actions))
                modules.append(nn.Softmax(dim=0))
            elif type(neurons) == list:
                if len(neurons) != 1 + layers:
                    raise KeyError("Length of neurons must be (layers+1)")
                modules.append(nn.Linear(n_states, neurons[0]))
                modules.append(activation)
                for i in range(layers):
                    modules.append(nn.Linear(neurons[i], neurons[i + 1]))
                    modules.append(activation)

                modules.append(nn.Linear(neurons[-1], n_actions))
                modules.append(nn.Softmax(dim=0))
            else:
                raise TypeError("Only Int and List Are Allowed")
        self.network = nn.Sequential(*modules)
        self.network.apply(self.initialize_weights)

    def initialize_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            self.initialization(module.weight)
            ## always initialize bias as 0
            nn.init.zeros_(module.bias)


    def select_action(self, a_distribution):
        return np.random.choice(range(self.n_actions), p=a_distribution.squeeze(0).detach().cpu().numpy())

    def forward(self, state):
        if self.conv:
            state = state.transpose(2, 0, 1)
            state = torch.tensor(state.reshape(-1, 2, 7, 7))
        else:
            state = torch.tensor(state.reshape(-1, ))
        action_distribution = self.network(state)
        if self.conv:
            action_distribution = action_distribution.reshape(-1, )
        # print(action_distribution, action_distribution.shape)
        # print(action_distribution)
        action = self.select_action(action_distribution)
        log_prob_action = torch.log(action_distribution.squeeze(0))[action]

        entropy = - torch.sum(action_distribution * torch.log(action_distribution))
        return action, log_prob_action, entropy

    def log_prob_based_on_action(self, state, action):
        if self.conv:
            state = state.transpose(2, 0, 1)
            state = torch.tensor(state.reshape(-1, 2, 7, 7))
        else:
            state = torch.tensor(state.reshape(-1, ))
        action_distribution = self.network(state)
        if self.conv:
            action_distribution = action_distribution.reshape(-1, )
        # print(action_distribution, action_distribution.shape)
        # print(action_distribution)
        log_prob_action = torch.log(action_distribution.squeeze(0))[action]
        return log_prob_action

class REINFORCEAgent():
    def __init__(self, env, n_states, n_actions = 3, learning_rate=0.001, lamda=0.01, gamma=0.99, step=1, if_conv=False, ClipPPO=False):

        self.learning_rate = learning_rate
        self.lamda = lamda # control the strength of the entropy regularization term in the loss
        self.gamma = gamma
        self.n_states = n_states
        self.env = env
        self.n_actions = n_actions
        self.actions = range(self.n_actions)
        # self.steps = steps
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.policy = Policy(n_states, n_actions, neurons = 128, if_conv=if_conv).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.type = torch.float32
        self.steps = 0
        self.old_policy = Policy(n_states, n_actions, neurons = 128, if_conv=if_conv).to(self.device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        # print(sum([param.nelement() for param in self.policy.parameters()]))
        # sys.exit(0)
        self.ClipPPO = ClipPPO



    def train(self, episode):
        s = self.env.reset()
        trace = []
        log_probs = []
        old_probs = []
        entropies = []
        for step in range(10000):
            a, log_prob, entropy = self.policy(s)
            old_prob = self.old_policy.log_prob_based_on_action(s, a)
            next_s, r, terminal, _ = self.env.step(a)
            trace.append((s, a, r))
            log_probs.append(log_prob)
            entropies.append(entropy)
            old_probs.append(old_prob.squeeze(0).detach().cpu().numpy())
            if not terminal:
                s = next_s
            else:
                break
        # print("Episode: ", episode, "\n", log_probs, "\n", old_probs)
        # print("****************************************************")
        self.old_policy.load_state_dict(self.policy.state_dict())
        # sys.exit(0)

        discounted_rewards = []
        gt_pre = 0
        for t in range(len(trace)-1, -1, -1):
            gt = trace[t][2] + self.gamma * gt_pre
            discounted_rewards.append(gt)
            gt_pre = gt
        discounted_rewards.reverse()

        discounted_rewards = np.array(discounted_rewards)
        discounted_rewards_tensor = torch.tensor(discounted_rewards, dtype=self.type, device=self.device)

        log_probs = torch.stack(log_probs)
        old_probs = torch.tensor(np.array(old_probs), dtype=self.type, device=self.device)
        entropies = torch.stack(entropies)



        if self.ClipPPO:
            # ratio = log_probs / old_probs
            ratio = torch.exp(log_probs - old_probs)
            termOne = ratio * discounted_rewards_tensor
            termTwo = torch.clamp(ratio, 0.8, 1.2) * discounted_rewards_tensor
            policy_gradient = - (torch.min(termOne, termTwo) + self.lamda * entropies) / len(trace)
        else:
            policy_gradient = -(discounted_rewards_tensor * log_probs + self.lamda * entropies) / len(trace)
        self.policy.zero_grad()
        policy_gradient.sum().backward()
        self.optimizer.step()

        total_reward = np.sum([trace[i][2] for i in range(len(trace))])
        # print("Episode : {}, Reward : {:.2f}".format(episode, total_reward))
        return total_reward