## Environment
We have a frogger-like environment in which the agent is expected to safely cross a busy highway to reach the other side. The agent is initially positioned randomly on one side <br />
of the highway and can execute three discrete actions to reach the other side: 'stand still', 'move up', and 'move down'. The environment's state is represented as LiDAR scan <br />
observations that have a vector length of 60.

## Algorithms
**Deep Q Learning (DQN):** <br />
Value-iteration <br />
![](https://github.com/rprasan/Reinforcement-Learning/blob/main/Value-based%20algorithms/Videos%20of%20Results/Value%20Iteration.gif) <br /><br />
Policy iteration <br />
![](https://github.com/rprasan/Reinforcement-Learning/blob/main/Value-based%20algorithms/Videos%20of%20Results/Policy%20Iteration.gif) <br /><br /><br />

**Double Deep Q Learning (Double DQN):** <br />
On-policy first-visit MC control <br />
Environment 1: <br />
![](https://github.com/rprasan/Reinforcement-Learning/blob/main/Value-based%20algorithms/Videos%20of%20Results/MC%20Control%20-%201.gif) <br />
