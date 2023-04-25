import sys

import torch
from torch import nn
import random
import numpy as np
import math
import sys

def softmax(x, temp):
    x = x / temp # scale by temperature
    z = x - max(x) # substract max to prevent overflow of softmax
    # print(np.exp(z)/np.sum(np.exp(z)))
    return np.exp(z)/np.sum(np.exp(z)) # compute softmax

def weights_init(module):
    if isinstance(module, nn.Conv2d):
        module.weight.data.normal_(mean=0.0, std=1.0)
        if module.bias is not None:
            module.bias.data.zero_()

class DQN(nn.Module):
    def __init__(self, n_states, n_actions, layers=1, neurons=128, activation=nn.ReLU()):
        super(DQN, self).__init__()

        self.n_actions = n_actions

        modules = []
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


    def select_action(self, a_distribution):
        return np.random.choice(range(self.n_actions), p=a_distribution.squeeze(0).detach().cpu().numpy())

    def forward(self, state):
        state = torch.tensor(state.reshape(-1, ))
        action_distribution = self.network(state)
        action = self.select_action(action_distribution)
        log_prob_action = torch.log(action_distribution.squeeze(0))[action]

        entropy = - torch.sum(action_distribution * torch.log(action_distribution))
        return action, log_prob_action, entropy

class Agent():
    def __init__(self, env, n_states, n_actions = 3, learning_rate=0.001, lamda=0.1, gamma=0.99, epsilon=0.3, steps=5):

        self.learning_rate = learning_rate
        self.lamda = lamda # control the strength of the entropy regularization term in the loss
        self.epsilon = epsilon
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_states = n_states
        self.env = env
        self.n_actions = n_actions
        self.actions = range(self.n_actions)
        self.steps = steps
        self.device = "cpu"
        self.policy = DQN(n_states, n_actions, neurons = [128, 128]).to(self.device)
        weights_init(self.policy)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.type = torch.float32
        self.steps = 0

    # def entropy_regularisation(self, state):
    #     return -np.sum(self.policy(state) * np.log(self.policy(state)))


    def train(self, episode):


        s = self.env.reset()
        trace = []
        log_probs = []
        entropies = []
        for step in range(10000):
            a, log_prob, entropy = self.policy(s)

            next_s, r, terminal, _ = self.env.step(a)
            trace.append((s, a, r))
            log_probs.append(log_prob)
            entropies.append(entropy)
            if not terminal:
                s = next_s
            else:
                break

        total_reward = np.sum([trace[i][2] for i in range(len(trace))])
        print("Episode : {}, Reward : {:.2f}".format(episode, total_reward))

        discounted_rewards = []
        for t in range(len(trace)):
            gt = 0
            for next_t in range(t, len(trace)):
                gt += self.gamma**(next_t-t) * trace[next_t][2]
            discounted_rewards.append(gt)

        # print([trace[i][2] for i in range(len(trace))])
        # print(discounted_rewards)

        #https://github.com/kvsnoufal/reinforce/blob/main/train_play_cartpole.py
        discounted_rewards = np.array(discounted_rewards)

        discounted_rewards_tensor = torch.tensor(discounted_rewards, dtype=self.type, device=self.device)
        # discounted_rewards_normalized = (discounted_rewards_tensor - torch.mean(discounted_rewards_tensor)) / (torch.std(discounted_rewards_tensor))

        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)
        policy_gradient = -(discounted_rewards_tensor * log_probs + self.lamda * entropies)
        self.policy.zero_grad()
        policy_gradient.sum().backward()
        self.optimizer.step()


