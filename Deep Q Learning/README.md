## Environment
We have a frogger-like environment in which the agent is expected to cross a busy highway to reach the other side safely. The agent is initially positioned randomly on one side of the highway and can execute a set of discrete actions to reach the other side. The environment's state is represented as LiDAR scan observations that have a vector length of 60. <br /><br />
Environment 1: <br />
In the environment's first version, the highway traffic is unidirectional, and the agent can execute only a set of three actions to reach the other side: 'stand still', 'move up', and 'move down'. <br /> <br />
Environment 2: <br />
In the environment's second version, the highway traffic is bidirectional, and the agent can execute a set of five actions to reach the other side: 'stand still', 'move up', and 'move down'. As the traffic is bidirectional, this environment is more difficult to solve that Environment 1. <br />

## Algorithms
**Deep Q Learning (DQN):** <br />
Environment 1: <br />
![](https://github.com/rprasan/Reinforcement-Learning/blob/main/Deep%20Q%20Learning/Results/Environment%201/Seed%206/Mean%20Training%20Reward%20For%20Last%20100%20Episodes.png) <br />
![](https://github.com/rprasan/Reinforcement-Learning/blob/main/Deep%20Q%20Learning/Results/Environment%201/Seed%206/TimeStepEpisode.png) <br />
![](https://github.com/rprasan/Reinforcement-Learning/blob/main/Deep%20Q%20Learning/Results/Environment%201/Seed%206/Average%20Evaluation%20Score%20Over%20Last%20100%20Roll%20Outs.png) <br />
![](https://github.com/rprasan/Reinforcement-Learning/blob/main/Deep%20Q%20Learning/Results/Environment%201/Seed%206/Average%20Evaluation%20Score%20Over%20All%20Roll%20Outs.png) <br /><br />

**Double Deep Q Learning (Double DQN):** <br />
Environment 2: <br />
![](https://github.com/rprasan/Reinforcement-Learning/blob/main/Deep%20Q%20Learning/Results/Environment%202/Seed%206/Mean%20Training%20Reward%20For%20Last%20100%20Episodes.png) <br />
![](https://github.com/rprasan/Reinforcement-Learning/blob/main/Deep%20Q%20Learning/Results/Environment%202/Seed%206/TimeStepEpisode.png) <br />
![](https://github.com/rprasan/Reinforcement-Learning/blob/main/Deep%20Q%20Learning/Results/Environment%202/Seed%206/Average%20Evaluation%20Score%20Over%20Last%20100%20Roll%20Outs.png) <br />
![](https://github.com/rprasan/Reinforcement-Learning/blob/main/Deep%20Q%20Learning/Results/Environment%202/Seed%206/Average%20Evaluation%20Score%20Over%20All%20Roll%20Outs.png) <br /><br />
