# Licensing information: You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the authors.
#
# Authors: Avishek Biswas (avisheb@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)

import numpy as np
from gym.envs.registration import register
from frogger_env.envs.abstract import AbstractEnv
from frogger_env.agent.road import Road, Vehicle
from frogger_env.agent.frogger_agent import Agent


class ReachGoalEnv(AbstractEnv):
    """
    A reach goal variant of the frogger environment.

    The agent has to cross a static highway and reach a target at other side.
    """

    GOAL_REWARD = 1.0
    GOAL_EPSILON = 2
    COLLISION_REWARD = -0.5
    DISTANCE_REWARD = 0.05
    ACTIONS_MAP = np.array([
                           [0, 1],
                           [0, -1],
                           [1, 0],
                           [-1, 0]])

    def default_config(self):
        config = super().default_config()
        config.update({
            "observation": {
                "position":
                {
                    "frame_history": 1,
                    "flatten_observation": True,
                    "include_goal_distance": False,
                    "include_goal_local_coodinates": True,
                    "augment": "None", #"lidar" "occupancy_grid" "None"
                    "grid_step": [2, 2],
                    "grid_size": [[-2.5*2, 2.5*2], [-2.5*2, 2.5*2]],
                    "angle_resolution": 20,
                    "sensing_distance": 5,
                    "origin": [0., 0.]  
                }
            },
            "observation_type": "position",
            "world_bounds": [0., 0., 50., 50],
            "lanes_count": 3,
            "vehicles_count": 20,  # upper bound
            "duration": 60,
            "vehicle_spacing": 3.,
            "vehicle_speed": 0,
            "vehicle_width": 2.,
            "random_init": 0,
            "random_goals":0,
            "random_maps": 0
        })
        return config

    def _reset(self):
        self._create_road()
        self._create_agent()

    def _create_road(self):
        """Create a road composed of straight adjacent lanes and populate it with vehicles."""
        self.road = Road(vehicles=[], lanes=[], np_random=self.np_random,
                         bidirectional=self.config["bidirectional"])
        self.road.generate_lanes(self.config["lanes_count"], length=50.)
        
        if not self.config["random_maps"]:        
            data = np.loadtxt("map.txt", delimiter = " ")
            for d in data:
                v = Vehicle(np.array([d[0], d[1]]), self.road.lanes[int(d[2])], 0, d[4], d[5])
                self.road.vehicles.append(v)
        else: 
            for _ in range(self.config["vehicles_count"]):
                self.road.generate_random_vehicle(speed=self.config["vehicle_speed"],
                                                  lane_id=None,
                                                  spacing=self.config["vehicle_spacing"],
                                                  width=self.config["vehicle_width"])

    def _create_agent(self):
        """
        Create the agent.
        """
        self.agent_spawn = [10 + np.random.rand()*30.,  self.road.get_first_lane_Y() -4] if self.config["random_init"]\
        else [25,  self.road.get_first_lane_Y() - 4]
        
        self.goal = [10 + np.random.rand()*30, self.road.get_last_lane_Y() + 5] \
            if self.config["random_goals"] else [25, self.road.get_last_lane_Y() + 5]
        self.agent = Agent(np.array(self.agent_spawn), radius=0.75, speed=4,
                           goal=self.goal, action_map=self.ACTIONS_MAP,
                           world_bounds=self.config["world_bounds"])

      
    def _reward(self, action):
        """
        The reward is received when the agent the goal.
        """
        sq_distance = np.sum(np.square(self.agent.position - self.goal))
        goal_reward = self.GOAL_REWARD * int(sq_distance < (self.GOAL_EPSILON) ** 2)
        dist_reward = self.DISTANCE_REWARD * np.dot(self.agent.velocity_direction, self.agent.goal_direction)
        collision_reward = self.COLLISION_REWARD if self.agent.crashed else 0
        return goal_reward + dist_reward + collision_reward


    def _is_terminal(self):
        """
        The episode is over if the agent collides, or the episode duration is met, 
        or the goal is reached.
        """
        return self.agent.crashed or \
            (self.time >= self.config["duration"] and not self.config["manual_control"]) or \
            np.sum(np.square(self.agent.position - self.goal)) < self.GOAL_EPSILON**2 or \
            not (self.config["world_bounds"][0] < self.agent.position[0] < self.config["world_bounds"][2]) or \
            not (self.config["world_bounds"][1] < self.agent.position[1] < self.config["world_bounds"][3])

register(
    id='reacher-v0',
    entry_point='frogger_env.envs:ReachGoalEnv',
)

