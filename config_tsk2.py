

import pickle


config_dict = {
    "observation": {
        "type": "OccupancyGrid",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20],
        },
        "grid_size": [[-20, 20], [-20, 20]],
        "grid_step": [5, 5],
        "absolute": False,
    },
    "action": {
        "type": "ContinuousAction",  # Changement ici pour des actions continues
        "steering_range": [-1, 1],   # Plage pour le contrôle de la direction
        "throttle_range": [0, 1],   # Plage pour le contrôle de l'accélération
    },
    "lanes_count": 4,
    "vehicles_count": 15,
    "duration": 60,  # [s]
    "initial_spacing": 0,
    "collision_reward": -1,
    "right_lane_reward": 0.5,
    "high_speed_reward": 0.1,
    "lane_change_reward": 0,
    "reward_speed_range": [20, 30],  # [m/s]
    "simulation_frequency": 5,
    "policy_frequency": 1,
    "screen_width": 600,
    "screen_height": 150,
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": True,
    "render_agent": True,
    "offscreen_rendering": False,
    "disable_collision_checks": True,
}


with open("config_tsk2.pkl", "wb") as f:
    pickle.dump(config_dict, f)

# env = gym.make("highway-fast-v0", render_mode="rgb_array")
# env.unwrapped.configure(config)
# print(env.reset())