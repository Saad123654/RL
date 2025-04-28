import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import gymnasium as gym
import time
import matplotlib.pyplot as plt
import highway_env

from config_2 import config

# Environment setup remains similar
env = gym.make("racetrack-v0", render_mode="rgb_array")



if hasattr(env.unwrapped, 'configure'):
    env.unwrapped.configure(config)

# Hardcoded observation size based on config2
OBS_SIZE = 288
N_DISCRETE_ACTIONS = 21 # Keep this explicit for discretization
ACTION_MIN = -1.0
ACTION_MAX = 1.0


def eval_agent(agent, env, n_episodes=10):
    episode_rewards = np.zeros(n_episodes)
    for i in range (n_episodes):
        state, _ = env.reset() # Using original env
        done = False
        total_reward = 0
        while not done:
            action_idx, action_val = agent.get_action(state)
            # Using action_val which is list/tuple
            next_state, reward, terminated, truncated, info = env.step(action_val)
            total_reward += reward
            state = next_state
            # Original termination condition
            done = terminated or truncated or info.get('crashed', False) or (info.get('rewards', {}).get('collision_reward', 0) != 0) or (not info.get('rewards', {}).get('on_road_reward', True))

        episode_rewards[i] = total_reward
    return np.mean(episode_rewards) # Return mean directly


def train(env, agent, n_episodes, eval_every=10, reward_threshold = 300, n_eval = 10):
    total_time = 0
    best_reward = -np.inf # Initialize properly
    reward_over_time = []
    for ep in range(n_episodes):
        done = False
        state, _ = env.reset()
        total_reward = 0
        while not done:
            action_idx, action_val = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action_val)

            # Original reward modification
            if not info.get('rewards', {}).get('on_road_reward', True): # If the car is off the road
                 reward -= 1.0
            # reward += 0.01 # Reward for staying on the road (Removed)

            # Original termination condition
            done = terminated or truncated or info.get('crashed', False) or (info.get('rewards', {}).get('collision_reward', 0) != 0) or (not info.get('rewards', {}).get('on_road_reward', True))

            # Update agent
            agent.update(state, action_idx, reward, done, next_state)

            state = next_state
            total_time += 1
            total_reward += reward

        reward_over_time.append(total_reward)

        if (ep+1) % eval_every == 0:
            # Evaluation mode handled internally if needed, or assumed no dropout/BN
            mean_reward = eval_agent(agent, env, n_eval)
            print(f"Episode {ep+1}, Mean Eval Reward: {mean_reward:.2f}")
            if mean_reward > best_reward:
                best_reward = mean_reward
                # Save best model
                print(f"Saving new best model with reward {best_reward:.2f}")
                torch.save(agent.policy_net.state_dict(), "reinforcement_agent.pth")

            if mean_reward > reward_threshold:
                print(f"Solved in {ep+1} episodes!")
                break
    print("Finished training")
    return reward_over_time


# Renamed back from map_index_to_action
def resize(action_index, n_actions=N_DISCRETE_ACTIONS, action_min=ACTION_MIN, action_max=ACTION_MAX):
    if n_actions <= 1: return (action_min + action_max) / 2
    step = (action_max - action_min) / (n_actions - 1)
    return action_min + action_index * step


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
         # Flatten input simpler way
         if isinstance(x, np.ndarray):
             x = torch.tensor(x, dtype=torch.float32)
         if x.dim() > 1:
             x = x.view(x.size(0), -1)
         elif x.dim() == 1:
             x = x.flatten()
         return self.net(x)


# Combined Base and Batch functionality, renamed to Agent
class Agent:
    def __init__(
        self,
        obs_size,
        n_actions,
        gamma,
        episode_batch_size,
        learning_rate,
        hidden_size=256, # Hardcoded default
    ):
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.gamma = gamma
        self.episode_batch_size = episode_batch_size
        self.learning_rate = learning_rate

        self.policy_net = Net(obs_size, hidden_size, n_actions)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        self.current_episode_tuples = [] # Renamed
        self.batch_scores = [] # Renamed
        self.n_eps_in_batch = 0 # Renamed

    # Using original gradient_returns logic
    def gradient_returns(self, rewards, gamma):
        G = 0
        returns_list = []
        # Calculate returns backwards
        for r in reversed(rewards):
            G = r + gamma * G
            returns_list.insert(0, G) # Prepend to maintain order

        returns_t = torch.tensor(returns_list, dtype=torch.float32)
        # Normalize
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-9)
        return returns_t

    def get_action(self, state):
        flat_state = torch.tensor(state.flatten(), dtype=torch.float32)
        with torch.no_grad():
            logits = self.policy_net(flat_state)
            action_probs = F.softmax(logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            action_index = action_dist.sample().item()

        continuous_action = resize(action_index, self.n_actions)
        return action_index, [continuous_action]

    def update(self, state, action_index, reward, done, next_state):
        self.current_episode_tuples.append((state, action_index, reward))

        if done:
            self.n_eps_in_batch += 1
            states, action_indices, rewards = zip(*self.current_episode_tuples)

            returns = self.gradient_returns(rewards, self.gamma)

            state_t = torch.tensor(np.array(states), dtype=torch.float32)
            action_idx_t = torch.tensor(action_indices, dtype=torch.long)

            logits = self.policy_net(state_t)
            log_probs_all = F.log_softmax(logits, dim=1)
            log_probs_taken = log_probs_all.gather(1, action_idx_t.unsqueeze(1)).squeeze(1)

            episode_score = (log_probs_taken * returns).sum() # Sum over steps
            self.batch_scores.append(episode_score)
            self.current_episode_tuples = []

            if self.n_eps_in_batch >= self.episode_batch_size:
                self.optimizer.zero_grad()
                # Average over batch
                batch_loss = -torch.stack(self.batch_scores).mean()
                batch_loss.backward()
                self.optimizer.step()

                self.batch_scores = []
                self.n_eps_in_batch = 0
        return


def main():
    TRAIN = True # Set to False to load and run

    agent = Agent(
        obs_size=OBS_SIZE,
        n_actions=N_DISCRETE_ACTIONS,
        gamma=0.99,
        episode_batch_size=32,
        learning_rate=0.0005,
        hidden_size=256 
    )

    if TRAIN :
        start_time = time.time()
        total_reward_history = train(
            env, agent,
            n_episodes=2500,
            eval_every=50,
            reward_threshold=200,
            n_eval=10
        )
        end_time = time.time()
        print(f"Training time: {(end_time - start_time)//60} min" )

        # Simplified Plotting
        plt.figure()
        plt.plot(total_reward_history)
        plt.title("Reward over time")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        # plt.savefig("reward.png") # Optional save
        plt.show()

        # Load best model for potential run after training
        try:
            agent.policy_net.load_state_dict(torch.load("reinforcement_agent.pth"))
        except FileNotFoundError:
            print("Best model file not found after training.")


    else:
        print("Loading existing model...")
        try:
            agent.policy_net.load_state_dict(torch.load("reinforcement_agent.pth"))
        except FileNotFoundError:
             print("Model file 'reinforcement_agent.pth' not found. Cannot run.")
             env.close()
             sys.exit(1) # Exit if model required but not found


    # Run agent directly if not training, or after training (using potentially best loaded model)
    if not TRAIN: # Only run render loop if we didn't just train
        print("Running loaded agent...")
        run_env = gym.make(env.spec.id, render_mode="human")
        if hasattr(run_env.unwrapped, 'configure'):
            run_env.unwrapped.configure(config)

        state, _ = run_env.reset()
        done = False
        total_reward = 0
        agent.policy_net.eval() # Still good practice

        while not done:
            action_idx, action_val = agent.get_action(state)
            state, reward, terminated, truncated, info = run_env.step(action_val)

            # Original termination condition again
            done = terminated or truncated or info.get('crashed', False) or (info.get('rewards', {}).get('collision_reward', 0) != 0) or (not info.get('rewards', {}).get('on_road_reward', True))

            run_env.render()
            total_reward += reward
            # time.sleep(0.05) # Removed explicit sleep

        print(f"Run finished. Final reward: {total_reward:.2f}")
        run_env.close()


    env.close()


if __name__ == "__main__":
    main()