import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(next_state, dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),  # Aplatir l'entrée
            nn.Linear(448, 256),  # Ajuste la dimension d'entrée en fonction de l'état
            nn.ReLU(),
            nn.Linear(256, 128),         # 256 (couche cachée) -> 128 (couche cachée)
            nn.ReLU(),
            nn.Linear(128, output_dim),  # 128 (couche cachée) -> output_dim (actions possibles)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Aplatir l'entrée pour qu'elle soit compatible avec la couche linéaire
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        env,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        buffer_size=100000,
        min_replay_size=1000,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=5000,
        target_update_freq=100,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.env = env
        self.obs_shape = env.observation_space.shape
        self.n_actions = env.action_space.n
        self.device = device
        state_dim = np.prod(env.observation_space.shape)  # Par exemple 448
        print(f"Forme de l'état : {self.obs_shape}")  # Afficher la forme de l'état pour le débogage
        self.q_net = QNetwork(state_dim, self.n_actions).to(self.device)
        self.target_q_net = QNetwork(state_dim, self.n_actions).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.min_replay_size = min_replay_size
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.step_count = 0
        self.target_update_freq = target_update_freq

        self.rewards = []

    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state = np.array(state, dtype=np.float32).flatten()  # Applatir l'état

            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)  # Ajout de la dimension batch
            with torch.no_grad():
                q_values = self.q_net(state_tensor)  # Passer l'état aplati à travers le réseau
            return q_values.argmax().item()

    def train(self, num_episodes=500):
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0

            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                self.step_count += 1
                self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                    np.exp(-1. * self.step_count / self.epsilon_decay)

                if len(self.replay_buffer) > self.min_replay_size:
                    self.update()

                if self.step_count % self.target_update_freq == 0:
                    self.target_q_net.load_state_dict(self.q_net.state_dict())

            self.rewards.append(total_reward)
            print(f"Episode {episode} | Total reward: {total_reward:.2f} | Epsilon: {self.epsilon:.3f}")

    def update(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q_values = self.target_q_net(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def plot_rewards(self):
        import matplotlib.pyplot as plt
        plt.plot(self.rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total reward")
        plt.title("Training Performance")
        plt.show()
