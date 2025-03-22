# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the Neural Network


class Network(nn.Module):

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, nb_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


# Implementing Experience Replay


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


# Implementing Deep Q Learning


class Dqn:

    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile=True)) * 50)  # T=100
        action = probs.multinomial(
            1
        )  # samples an action using multinomial distribution
        return action.data[0, 0]

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = (
            self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        )  # Old NN
        # Selects only the Q-values corresponding to the actions that were actually taken
        # .gather(1, ...): Indexes along dimension 1 (the action dimension) to get only the relevant Q-values
        next_outputs = self.model(batch_next_state).detach().max(1)[0]  # New NN
        # .detach(): Detaches the tensor from the computation graph, preventing gradients from flowing back through this path
        target = (
            self.gamma * next_outputs + batch_reward
        )  # Q = reward + gamma * max Q(next_state)
        td_loss = F.smooth_l1_loss(outputs, target)  # Temporal Loss
        # Uses the Huber loss (smooth L1) which is less sensitive to outliers than MSE
        # For small differences, it behaves like MSE (quadratic)
        # For large differences, it behaves like MAE (linear)
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph=True)
        self.optimizer.step()

    def update(self, reward, new_signal):
        # Manages the interaction between agent and environment
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push(
            (
                self.last_state,
                new_state,
                torch.LongTensor([int(self.last_action)]),
                torch.Tensor([self.last_reward]),
            )
        )  # Creates a transition tuple: (state, next_state, action, reward)
        action = self.select_action(new_state)  # Selects an action based on new state
        if len(self.memory.memory) > 200:
            batch_state, batch_next_state, batch_action, batch_reward = (
                self.memory.sample(200)
            )  # Checks if there are enough transitions stored in the replay memory (at least 100)
            # Learning only starts after collecting enough experiences
            # Unpacks the sampled transitions into batches of states, next states, actions, and rewards
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
            # Maintains a fixed-size window of the 1000 most recent rewards
            # If the window exceeds 1000 elements, removes the oldest reward
        return action

    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1.0)

    def save(self):
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            "last_brain.pth",
        )

    def load(self):
        if os.path.isfile("last_brain.pth"):
            print("=> loading checkpoint... ")
            checkpoint = torch.load("last_brain.pth")
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            print("done !")
        else:
            print("no checkpoint found...")
