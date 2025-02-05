import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Weight Initialization for Neural Networks
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

# =============================
#        ACTOR NETWORK
# =============================
class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

# =============================
#        CRITIC NETWORK
# =============================
class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# =============================
#  PRIORITIZED EXPERIENCE REPLAY BUFFER
# =============================
class PrioritizedReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, alpha=0.6):
        self.memory = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.alpha = alpha
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done, td_error=1.0):
        e = self.experience(state, action, reward, next_state, done)
        priority = (abs(td_error) + 1e-5) ** self.alpha
        self.memory.append(e)
        self.priorities.append(priority)

    def sample(self):
        probabilities = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.memory), self.batch_size, p=probabilities)
        experiences = [self.memory[i] for i in indices]
        return experiences, indices

    def update_priorities(self, indices, td_errors):
        for i, error in zip(indices, td_errors):
            self.priorities[i] = (abs(error) + 1e-5) ** self.alpha

    def __len__(self):
        return len(self.memory)

# =============================
#      ELASTIC STEP DDPG
# =============================
class MultiStepReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, n_step=3, gamma=0.99):
        self.memory = deque(maxlen=buffer_size)
        self.n_step_buffer = deque(maxlen=n_step)
        self.batch_size = batch_size
        self.n_step = n_step
        self.gamma = gamma

    def add(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_step:
            state, action, rewards, next_state, done = self.get_n_step_info()
            self.memory.append((state, action, rewards, next_state, done))

    def get_n_step_info(self):
        state, action, _, _, _ = self.n_step_buffer[0]
        next_state, done = self.n_step_buffer[-1][-2], self.n_step_buffer[-1][-1]
        reward = sum([self.gamma**i * r for i, (_, _, r, _, _) in enumerate(self.n_step_buffer)])
        return state, action, reward, next_state, done

    def sample(self):
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        experiences = [self.memory[i] for i in indices]
        return experiences, indices

    def __len__(self):
        return len(self.memory)

# =============================
#      OU NOISE PROCESS
# =============================
class OUNoise:
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.standard_normal(self.state.shape)
        self.state += dx
        return self.state
