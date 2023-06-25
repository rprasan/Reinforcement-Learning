## Environment
We have a frogger-like environment in which the agent is expected to cross a busy highway to reach the other side safely. The agent is initially positioned randomly on one side of the highway and can execute a set of discrete actions to reach the other side. The environment's state is represented as LiDAR scan observations that have a vector length of 60: <br />
Environment 1 <br />
In the environment's first version, the highway traffic is unidirectional, and the agent can execute only a set of three actions to reach the other side: 'stand still', 'move up', and 'move down'.
Environment 2 <br />
In the environment's second version, the highway traffic is bidirectional, and the agent can execute a set of five actions to reach the other side: 'stand still', 'move up', and 'move down'. As the traffic is bidirectional, this environment is more difficult to solve that Environment 1.

## Algorithms
**Deep Q Learning (DQN):** <br />
Environment 1 <br />
![](https://github.com/rprasan/Reinforcement-Learning/blob/main/Value-based%20algorithms/Videos%20of%20Results/Value%20Iteration.gif) <br /><br /><br />

**Double Deep Q Learning (Double DQN):** <br />
Environment 2 <br />
Environment 1: <br />
![](https://github.com/rprasan/Reinforcement-Learning/blob/main/Value-based%20algorithms/Videos%20of%20Results/MC%20Control%20-%201.gif) <br />
