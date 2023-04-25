import torch
from torch import nn
import random
import numpy as np
import math
import sys


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
    def __init__(self, env, n_states, n_actions = 3, learning_rate=0.001, lamda=0.01, gamma=0.99, steps=3):

        self.learning_rate = learning_rate
        self.lamda = lamda # control the strength of the entropy regularization term in the loss

        self.gamma = gamma
        self.n_states = n_states
        self.env = env
        self.n_actions = n_actions
        self.actions = range(self.n_actions)
        self.n = steps  # for bootstrapping, estimation depth
        self.device = "cpu"

        self.actor = ActorNetwork(n_states, n_actions, neurons = 128).to(self.device)
        weights_init(self.actor)
        self.ActorOptimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)

        self.critic = CriticNetwork(n_states, neurons = 128).to(self.device)
        weights_init(self.critic)
        self.criticLoss = torch.nn.MSELoss().to(self.device)
        self.CriticOptimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        self.type = torch.float32


        ##
        self.bootstrapping = False  # if use bootstrapping method
        self.baseline = True       # if use baseline subtraction method


    def train(self, episode):


        s = self.env.reset()
        trace = []
        log_probs = []
        entropies = []
        values = []
        advantage = []
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
        if episode % 1000 == 0:
            print("Episode : {}, Reward : {:.2f}".format(episode, total_reward))

        if self.bootstrapping:
            estimated_Q = []
            for t in range(len(trace) + 1 -self.n):
                Gt = self.gamma ** self.n * values[t+self.n]
                for i in range(t, t + self.n):
                    Gt += self.gamma**(i-t) * trace[i][2]
                estimated_Q.append(Gt)

            estimated_Q = torch.tensor(estimated_Q, dtype=self.type, device=self.device)

            log_probs = torch.stack(log_probs[:len(trace) + 1 - self.n])
            entropies = torch.stack(entropies[:len(trace) + 1 - self.n])
            values = torch.stack(values[:len(trace) + 1 - self.n])
            values = torch.reshape(values, (-1,))
        else:
            discounted_rewards = []
            for t in range(len(trace)):
                gt = 0
                for next_t in range(t, len(trace)):
                    gt += self.gamma ** (next_t - t) * trace[next_t][2]
                discounted_rewards.append(gt)
            estimated_Q = torch.tensor(np.array(discounted_rewards), dtype=self.type, device=self.device)

            log_probs = torch.stack(log_probs)
            entropies = torch.stack(entropies)
            values = torch.stack(values[:len(trace)])
            values = torch.reshape(values, (-1,))

        if self.baseline:
            advantage = estimated_Q - values
            actor_gradient = -(advantage.detach() * log_probs + self.lamda * entropies)/len(trace)
            critic_loss = torch.square(advantage)
            self.critic.zero_grad()
            self.actor.zero_grad()
            critic_loss.sum().backward()
            actor_gradient.sum().backward()
            self.CriticOptimizer.step()
            self.ActorOptimizer.step()
        else:
            actor_gradient = -(estimated_Q * log_probs + self.lamda * entropies)/len(trace)

            self.critic.zero_grad()
            self.actor.zero_grad()

            critic_loss = self.criticLoss(estimated_Q.detach(), values)
            critic_loss.backward()
            actor_gradient.sum().backward()
            self.CriticOptimizer.step()
            self.ActorOptimizer.step()









