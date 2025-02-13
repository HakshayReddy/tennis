"""
Define and build neural networks for a DDPG model
Adapted from the Udacity Deep Reinforcement Learning Nanodegree course materials.
Specificaly, from the code for the solution to the OpenAI Gym's pendulum environment
(https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum).
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
For initialization of hidden layers
"""
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

"""
Actor (Policy) network maps states to actions
Uses RELU activations.
2 fully connected layers.
Initially used batch normalization between all layers. 
It didn't seem to help in my current project so I removed it except for at the inputs.

"""
class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units = 128, fc2_units = 64):
        """
        Initialize parameters and build neural network.
            state_size (int): Number of parameters characterizing the environment state
            action_size (int): Number of possible actions
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden fully connected layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
        self.bn1 = nn.BatchNorm1d(state_size)
        #self.bn2 = nn.BatchNorm1d(fc1_units)
        #self.bn3 = nn.BatchNorm1d(fc2_units)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.fc1(self.bn1(state)))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

"""
Critic (Value) network returns a Q-value for a state-action pair.
Uses RELU activation.
Has 2 fully connected layers.
"""
class Critic(nn.Module):

    def __init__(self, state_size, action_size, seed, fcs1_units = 128, fc2_units = 64):
        """
        Initialize parameters and build neural network.
            state_size (int): Number of parameters characterizing the environament state
            action_size (int): Number of possible actions
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden fully connected layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        
        self.bn1 = nn.BatchNorm1d(state_size)

        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        xs = F.relu(self.fcs1(self.bn1(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)    
    
"""
Actor (Policy) Model. Maps states to actions.
Uses SELU activation. Has 3 fully connected layers.
"""
class Actor_SELU(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_units = 128, fc2_units = 64, fc3_units = 32):
        """
        Initialize parameters and build neural network.
            state_size (int): Number of parameters characterizing the environment state
            action_size (int): Number of possible actions
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden fully connected layer
            fc2_units (int): Number of nodes in second hidden layer
            fc3_units (int): Number of nodes in third hidden layer
        """
        super(Actor_SELU, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)

        self.selu1 = nn.SELU(fc1_units)
        self.selu2 = nn.SELU(fc2_units)
        self.selu3 = nn.SELU(fc3_units)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = self.selu1(self.fc1(state))
        x = self.selu2(self.fc2(x))
        x = self.selu3(self.fc3(x))
        return torch.tanh(self.fc4(x))


"""
Critic (Value) Model. Maps (state, action) pairs to Q-values.
Uses SELU activation.
Has 3 fully connected layers.
"""
class Critic_SELU(nn.Module):

    def __init__(self, state_size, action_size, seed, fcs1_units = 128, fc2_units = 64, fc3_units = 32):
        """
        Initialize parameters and build neural network.
            state_size (int): Number of parameters characterizing the environment state
            action_size (int): Number of possible actions
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden fully connected layer
            fc2_units (int): Number of nodes in second hidden layer
            fc3_units (int): Number of nodes in third hidden layer
        """
        super(Critic_SELU, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)

        self.selu1 = nn.SELU(fcs1_units)
        self.selu2 = nn.SELU(fc2_units)
        self.selu3 = nn.SELU(fc3_units)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        xs = self.selu1(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = self.selu2(self.fc2(x))
        x = self.selu3(self.fc3(x))
        return self.fc4(x)
    
