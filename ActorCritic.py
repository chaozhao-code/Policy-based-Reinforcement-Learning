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

class ActorNetwork(nn.Module):
    def __init__(self, n_states, n_actions, layers=1, neurons=128, activation=nn.ReLU()):
        super(ActorNetwork, self).__init__()

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


class CriticNetwork(nn.Module):
    def __init__(self, n_states, layers=1, neurons=128, activation=nn.ReLU()):
        super(CriticNetwork, self).__init__()

        modules = []
        if type(neurons) == int:
            modules.append(nn.Linear(n_states, neurons))
            modules.append(activation)
            for i in range(layers):
                modules.append(nn.Linear(neurons, neurons))
                modules.append(activation)
            modules.append(nn.Linear(neurons, 1))
        elif type(neurons) == list:
            if len(neurons) != 1 + layers:
                raise KeyError("Length of neurons must be (layers+1)")
            modules.append(nn.Linear(n_states, neurons[0]))
            modules.append(activation)
            for i in range(layers):
                modules.append(nn.Linear(neurons[i], neurons[i + 1]))
                modules.append(activation)

            modules.append(nn.Linear(neurons[-1], 1))
        else:
            raise TypeError("Only Int and List Are Allowed")
        self.network = nn.Sequential(*modules)

    def forward(self, state):
        state = torch.tensor(state.reshape(-1, ))
        return self.network(state)


class Agent():
    def __init__(self, n_states, env, n_actions = 3, learning_rate=0.001, lamda=0.1, gamma=0.99, epsilon=0.3, steps=5):

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
        self.actor = ActorNetwork(n_states, n_actions, neurons = [128, 128]).to(self.device)
        weights_init(self.actor)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic = CriticNetwork(n_states, neurons = [128, 128]).to(self.device)
        weights_init(self.critic)
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        self.type = torch.float32
        self.steps = 0
        self.n = 5

    # def entropy_regularisation(self, state):
    #     return -np.sum(self.policy(state) * np.log(self.policy(state)))


    def train(self, episode):


        s = self.env.reset()
        trace = []
        log_probs = []
        entropies = []
        values = []
        for step in range(10000):
            a, log_prob, entropy = self.actor(s)
            values.append(self.critic(s))
            next_s, r, terminal, _ = self.env.step(a)
            trace.append((s, a, r))
            log_probs.append(log_prob)
            entropies.append(entropy)
            if not terminal:
                s = next_s
            else:
                values.append(self.critic(next_s))
                break

        total_reward = np.sum([trace[i][2] for i in range(len(trace))])
        print("Episode : {}, Reward : {:.2f}".format(episode, total_reward))

        estimated_Q = []

        for t in range(len(trace) + 1 -self.n):
            Gt = self.gamma ** self.n * values[t+self.n]
            for i in range(t, t + self.n):
                Gt += self.gamma**(i-t) * trace[i][2]



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


