'''
This code defines the learning Agent for performing learning with the Twin Delayed Deep Deterministic Policy Gradient method.
Agent can actually consist of multiple agents which are learning simultaneously.
The code was adapted from code provided by Udacity's Deep Reinforcement Learning Nanodegree.
Specifically, it was adapted from the code for solving OpenAI Gym's pendulum environment
(https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum).
'''

import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic, Actor_SELU, Critic_SELU

import torch
import torch.nn.functional as F
import torch.optim as optim

# L2 weight decay for the optimizer. I didn't use this.
WEIGHT_DECAY = 0

# Use the the GPU if available.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class Agent():
    """
    The learning agent(s) for learning with the Twin Delayed Deep Deterministic Policy Gradient method.
    """
    
    def __init__(
        self,
        state_size,
        action_size,
        num_agents,
        random_seed = 0,
        batch_size = 128,
        lr_actor =  1e-4,
        lr_critic = 1e-3,
        noise_theta = 0.15,
        noise_sigma = 0.2,
        actor_fc1 = 128,
        actor_fc2 = 64,
        actor_fc3 = 32,
        critic_fc1 = 128,
        critic_fc2 = 64,
        critic_fc3 = 32,
        update_every = 1,
        num_updates = 1,
        buffer_size = int(2e6),
        network = 'RELU',
        policy_delay = 1,
        target_noise = 0.2,
        noise_clip = 0.5
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)
        self.batch_size = batch_size
        self.update_every = update_every
        self.num_updates = num_updates
        self.buffer_learn_size = min(batch_size * 10, buffer_size)
        self.policy_delay = policy_delay
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.update_count = 0

        if network == 'RELU':
            # Actor Network (w/ Target Network)
            self.actor_local = Actor(state_size, action_size, random_seed, actor_fc1, actor_fc2).to(device)
            self.actor_target = Actor(state_size, action_size, random_seed, actor_fc1, actor_fc2).to(device)
            self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

            # Critic Networks (w/ Target Networks)
            self.critic_local_1 = Critic(state_size, action_size, random_seed, critic_fc1, critic_fc2).to(device)
            self.critic_target_1 = Critic(state_size, action_size, random_seed, critic_fc1, critic_fc2).to(device)
            self.critic_optimizer_1 = optim.Adam(self.critic_local_1.parameters(), lr=lr_critic, weight_decay=WEIGHT_DECAY)

            self.critic_local_2 = Critic(state_size, action_size, random_seed, critic_fc1, critic_fc2).to(device)
            self.critic_target_2 = Critic(state_size, action_size, random_seed, critic_fc1, critic_fc2).to(device)
            self.critic_optimizer_2 = optim.Adam(self.critic_local_2.parameters(), lr=lr_critic, weight_decay=WEIGHT_DECAY)

        elif network == 'SELU':
            # Actor Network (w/ Target Network)
            self.actor_local = Actor_SELU(state_size, action_size, random_seed, actor_fc1, actor_fc2).to(device)
            self.actor_target = Actor_SELU(state_size, action_size, random_seed, actor_fc1, actor_fc2).to(device)
            self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

            # Critic Networks (w/ Target Networks)
            self.critic_local_1 = Critic_SELU(state_size, action_size, random_seed, critic_fc1, critic_fc2).to(device)
            self.critic_target_1 = Critic_SELU(state_size, action_size, random_seed, critic_fc1, critic_fc2).to(device)
            self.critic_optimizer_1 = optim.Adam(self.critic_local_1.parameters(), lr=lr_critic, weight_decay=WEIGHT_DECAY)

            self.critic_local_2 = Critic_SELU(state_size, action_size, random_seed, critic_fc1, critic_fc2).to(device)
            self.critic_target_2 = Critic_SELU(state_size, action_size, random_seed, critic_fc1, critic_fc2).to(device)
            self.critic_optimizer_2 = optim.Adam(self.critic_local_2.parameters(), lr=lr_critic, weight_decay=WEIGHT_DECAY)
            
        # Noise process
        self.noise = OUNoise((num_agents, action_size), random_seed, theta=noise_theta, sigma=noise_sigma)

        # Replay memory
        self.memory = PrioritizedReplayBuffer(action_size, buffer_size, self.batch_size, random_seed)

        # Initialize the time step counter for updating each UPDATE_EVERY number of steps)
        self.t_step = 0
        
    # Don't forget to make add_noise False when not training.
    def act(self, state, add_noise=True, noise_scale=1.0):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += (noise_scale * self.noise.sample())
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()                

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Weighted average. Smaller tau means more of the updated target model is
            weighted towards the current target model.
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
        tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def step(self, states, actions, rewards, next_states, dones, gamma=0.96, tau=0.001):
        batch = []
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            batch.append((state, action, reward, next_state, done))

        # Convert batch to tensors
        state_batch = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=device)
        action_batch = torch.tensor([b[1] for b in batch], dtype=torch.float32, device=device)
        reward_batch = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
        next_state_batch = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=device)
        done_batch = torch.tensor([b[4] for b in batch], dtype=torch.uint8, device=device)

        # Fix: Ensure TD error computation is only done if batch size > 1
        if state_batch.shape[0] > 1:
            with torch.no_grad():
                Q_expected_1 = self.critic_local_1(state_batch, action_batch)
                Q_expected_2 = self.critic_local_2(state_batch, action_batch)
                Q_expected = torch.min(Q_expected_1, Q_expected_2)
                actions_next = self.actor_target(next_state_batch)
                Q_next_1 = self.critic_target_1(next_state_batch, actions_next)
                Q_next_2 = self.critic_target_2(next_state_batch, actions_next)
                Q_next = torch.min(Q_next_1, Q_next_2)
        else:
            # Expand dimensions to avoid BatchNorm errors
            state_batch = state_batch.unsqueeze(0)
            action_batch = action_batch.unsqueeze(0)
            next_state_batch = next_state_batch.unsqueeze(0)

            with torch.no_grad():
                Q_expected_1 = self.critic_local_1(state_batch, action_batch)
                Q_expected_2 = self.critic_local_2(state_batch, action_batch)
                Q_expected = torch.min(Q_expected_1, Q_expected_2)
                actions_next_batch = self.actor_target(next_state_batch)
                Q_next_1 = self.critic_target_1(next_state_batch, actions_next_batch)
                Q_next_2 = self.critic_target_2(next_state_batch, actions_next_batch)
                Q_next = torch.min(Q_next_1, Q_next_2)

        Q_target = reward_batch + (gamma * Q_next * (1 - done_batch))
        errors = torch.abs(Q_expected - Q_target).cpu().numpy().flatten()
        errors = np.maximum(errors, 0.1)  # Prevents extremely small TD errors

        # Store experiences with computed priorities
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            self.memory.add(state, action, reward, next_state, done, errors[i])

        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.buffer_learn_size:
                for _ in range(self.num_updates):
                    experiences, indices, weights = self.memory.sample(self.batch_size)
                    if len(indices) > 0 and len(weights) > 0:
                        self.learn(experiences, indices, weights, gamma, tau)


    def learn(self, experiences, indices, weights, gamma, tau):
        """Update policy and value parameters using a batch of experience tuples."""
        
        states, actions, rewards, next_states, dones = experiences
        
        actions_next = self.actor_target(next_states)
        
        # Compute Q targets
        Q_targets_next_1 = self.critic_target_1(next_states, actions_next)
        Q_targets_next_2 = self.critic_target_2(next_states, actions_next)
        Q_targets_next = torch.min(Q_targets_next_1, Q_targets_next_2)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute critic losses
        Q_expected_1 = self.critic_local_1(states, actions)
        critic_loss_1 = F.mse_loss(Q_expected_1, Q_targets)

        Q_expected_2 = self.critic_local_2(states, actions)
        critic_loss_2 = F.mse_loss(Q_expected_2, Q_targets)
        
        Q_expected = torch.min(Q_expected_1, Q_expected_2)

        # Minimize the loss
        self.critic_optimizer_1.zero_grad()
        critic_loss_1.backward(retain_graph=True)
        self.critic_optimizer_1.step()

        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        self.critic_optimizer_2.step()

        # Compute TD errors (Important for PER)
        td_errors = torch.abs(Q_expected - Q_targets).detach().cpu().squeeze().numpy()
        td_errors = np.maximum(td_errors, 0.1)  # 🔹 Ensure errors are not too small
        
        # Update actor (every `policy_delay` steps)
        self.update_count = (self.update_count + 1) % self.policy_delay
        if self.update_count == 0:
            actions_pred = self.actor_local(states)
            actor_loss = -self.critic_local_1(states, actions_pred).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            self.soft_update(self.critic_local_1, self.critic_target_1, tau)
            self.soft_update(self.critic_local_2, self.critic_target_2, tau)
            self.soft_update(self.actor_local, self.actor_target, tau)

        # Update experience priorities
        self.memory.update_priorities(indices, td_errors)

class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples with priorities."""

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        Initialize a PrioritizedReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            alpha (float): prioritization factor (0 = uniform sampling, 1 = full prioritization)
            beta (float): importance sampling correction factor (increases over time)
            beta_increment (float): increment rate for beta
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # Maintain deque structure
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.priorities = deque(maxlen=buffer_size)  # Store priorities
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.eps = 1e-5  # Small value to prevent zero priority
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done, error):
        """Add a new experience with priority to memory."""
        priority = (abs(error) + self.eps) ** self.alpha
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.priorities.append(priority)

    def sample(self, batch_size):
        """Sample a batch of experiences from memory based on priority."""
        
        priorities = np.array(self.priorities)

        if priorities.sum() == 0:
            priorities += 1e-3  # 🔹 Prevent zero probabilities

        sampling_probabilities = priorities / priorities.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=sampling_probabilities)


        experiences = [self.memory[idx] for idx in indices]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        # Importance Sampling Weights
        weights = (len(self.memory) * sampling_probabilities[indices]) ** -self.beta
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)  # Increase beta over time

        # print("Sampled")

        return (states, actions, rewards, next_states, dones), indices, torch.tensor(weights, dtype=torch.float32, device=device)

    def update_priorities(self, indices, errors):
        """Update priorities of sampled experiences."""
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (abs(error) + self.eps) ** self.alpha

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
    
'''
Load checkpoints for the actor and critic networks for the pre-trained agent.
Run the agent for 100 episodes and average the scores

Paramenters:
    agent: The agent
    env: The environment
    actor_checkpoint, critic_checkpoint: The filenames of the checkpoints for the actor and critic neural networks
    n_episodes: The number of episodes to run. The scores will be printed and then the final average.
'''
def load_and_run(agent, env, actor_checkpoint, critic_checkpoint, n_episodes):
    
    # load the weights from file
    agent.actor_local.load_state_dict(torch.load(actor_checkpoint))
    agent.critic_local.load_state_dict(torch.load(critic_checkpoint))
    
    # Get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    total_score = 0.0
    for i in range(n_episodes):
        # Reset the environment
        env_info = env.reset(train_mode=False)[brain_name] 
        # Get the current state
        state = env_info.vector_observations
        # Initialize the scores
        score = np.zeros(agent.num_agents)
        while True:
            # Choose actions.
            # Don't forget to make add_noise False when not training.
            action = agent.act(state, add_noise = False)
            # Send actions to the environment
            env_info = env.step(action)[brain_name]
            # Get the next state
            next_state = env_info.vector_observations
            # Get rewards
            reward = env_info.rewards
            # Check if the episode is finishised
            done = env_info.local_done
            # Add rewards to the scores
            score += reward
            # Replace the current state with the next state for the next timestep
            state = next_state
            # Exit the loop if the episode is finished
            if np.any(done):
                break
        print("Ep {}\tScore: {:.2f}".format(i + 1, np.mean(score)))
        total_score += np.mean(score)
    print("Avg over {} episodes: {:.2f}".format(n_episodes, total_score / n_episodes))
