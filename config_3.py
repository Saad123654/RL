import highway_env
import gymnasium as gym
from math import pi

env = gym.make("parking-v0", render_mode="rgb_array")

config = {
    "observation": {
        "type": "KinematicsGoal",
        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
        "scales": [100, 100, 5, 5, 1, 1],
        "normalize": False,
    },
    "action": {"type": "ContinuousAction"},
    "reward_weights": [1, 0.3, 0, 0, 0.03, 0.02],
    "success_goal_reward": 1,
    "collision_reward": -5,
    "steering_range": pi / 3,
    "simulation_frequency": 40,
    "policy_frequency": 5,
    "duration": 100,
    "screen_width": 600,
    "screen_height": 300,
    "centering_position": [0.5, 0.5],
    "scaling": 7,
    "controlled_vehicles": 1,
    "vehicles_count": 7,
    "add_walls": True,
    "offscreen_rendering": False,
    "show_trajectories": False,
    "render_agent": True,
}

env.unwrapped.configure(config)
