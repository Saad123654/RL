import highway_env
import gymnasium as gym

env = gym.make("racetrack-v0", render_mode="rgb_array")

config = {
    "observation": {
        "type": "OccupancyGrid",
        "features": ["presence", "on_road"],
        "grid_size": [[-18, 18], [-18, 18]],
        "grid_step": [3, 3],
        "as_image": False,
        "align_to_vehicle_axes": True,
    },
    "action": {"type": "ContinuousAction", "longitudinal": False, "lateral": True},
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 300,
    "collision_reward": -2,
    "lane_centering_cost": 4,
    "action_reward": -0.3,
    "controlled_vehicles": 1,
    "other_vehicles": 1,
    "screen_width": 600,
    "screen_height": 600,
    "centering_position": [0.5, 0.5],
    "scaling": 7,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False,
}


env.unwrapped.configure(config)
