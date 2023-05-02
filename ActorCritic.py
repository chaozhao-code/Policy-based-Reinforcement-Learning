import torch
from torch import nn
import random
import numpy as np
import math
import sys




class ActorNetwork(nn.Module):
    def __init__(self, n_states, n_actions, layers=1, neurons=128, activation=nn.SiLU(), initialization=nn.init.xavier_uniform_):
        super(ActorNetwork, self).__init__()

        self.n_actions = n_actions
        self.initialization = initialization

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
        self.network.apply(self.initialize_weights)

    def initialize_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            self.initialization(module.weight)
            ## always initialize bias as 0
            nn.init.zeros_(module.bias)


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
    def __init__(self, n_states, layers=1, neurons=128, activation=nn.ReLU(), initialization = nn.init.xavier_uniform_):
        super(CriticNetwork, self).__init__()
        self.initialization = initialization

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
        self.network.apply(self.initialize_weights)

    def initialize_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            self.initialization(module.weight)
            ## always initialize bias as 0
            nn.init.zeros_(module.bias)

    def forward(self, state):
        state = torch.tensor(state.reshape(-1, ))
        return self.network(state)


class ACAgent():
    def __init__(self, env, n_states, n_actions=3, learning_rate=0.001, lamda=0.01, gamma=0.99, step=1, if_conv=False, bootstrapping=True, baseline=True):

        self.learning_rate = learning_rate
        self.lamda = lamda # control the strength of the entropy regularization term in the loss

        self.gamma = gamma
        self.n_states = n_states
        self.env = env
        self.n_actions = n_actions
        self.actions = range(self.n_actions)
        self.n = step  # for bootstrapping, estimation depth
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.actor = ActorNetwork(n_states, n_actions, neurons = 128).to(self.device)
        self.ActorOptimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)

        self.critic = CriticNetwork(n_states, neurons = 128).to(self.device)
        self.criticLoss = torch.nn.MSELoss().to(self.device)
        self.CriticOptimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        self.type = torch.float32
        self.if_conv = if_conv


        ##
        self.bootstrapping = bootstrapping  # if use bootstrapping method
        self.baseline = baseline       # if use baseline subtraction method


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
            gt_pre = 0
            for t in range(len(trace) - 1, -1, -1):
                gt = trace[t][2] + self.gamma * gt_pre
                discounted_rewards.append(gt)
                gt_pre = gt
            discounted_rewards.reverse()

            estimated_Q = torch.tensor(np.array(discounted_rewards), dtype=self.type, device=self.device)

            log_probs = torch.stack(log_probs)
            entropies = torch.stack(entropies)
            values = torch.stack(values[:len(trace)])
            values = torch.reshape(values, (-1,))

        if self.baseline:
            advantage = estimated_Q - values
            actor_gradient = -(advantage.detach() * log_probs + self.lamda * entropies) / len(values)
            critic_loss = torch.square(advantage) / len(values)
            self.critic.zero_grad()
            self.actor.zero_grad()
            critic_loss.sum().backward()
            actor_gradient.sum().backward()
            self.CriticOptimizer.step()
            self.ActorOptimizer.step()
        else:
            actor_gradient = -(estimated_Q.detach() * log_probs + self.lamda * entropies) / len(values)

            self.critic.zero_grad()
            self.actor.zero_grad()

            critic_loss = torch.square(estimated_Q.detach() - values) / len(values)
            # self.criticLoss(estimated_Q.detach(), values)
            critic_loss.sum().backward()
            actor_gradient.sum().backward()
            self.CriticOptimizer.step()
            self.ActorOptimizer.step()

        total_reward = np.sum([trace[i][2] for i in range(len(trace))])
        return total_reward
        # if episode % 1000 == 0:
        # print("Episode : {}, Reward : {:.2f}".format(episode, total_reward))









