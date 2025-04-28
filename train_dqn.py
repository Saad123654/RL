from config_1 import env
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

agent = DQNAgent(env)
agent.train(num_episodes=300)
agent.plot_rewards()



def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_rewards_with_smoothing(rewards, window=10):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Récompense brute')
    if len(rewards) >= window:
        smoothed = moving_average(rewards, window)
        plt.plot(range(window - 1, len(rewards)), smoothed, label=f'Moyenne mobile ({window})')
    plt.xlabel('Épisode')
    plt.ylabel('Récompense')
    plt.title('Évolution des récompenses par épisode')
    plt.legend()
    plt.grid()
    plt.show()


plot_rewards_with_smoothing(agent.rewards)


