## Environment
We use the following environments to demonstrate how black-box techniques can directly optimize an agent's policies: <br /><br />
Environment 1: <br />
The first environment has a frogger-like agent that navigates to a fixed goal on a static highway map by executing four discrete actions: 'move up', 'move down', 'move left', and 'move right'. The agent's observation of the environment's state is represented as a vector comprising the agent's position, the goal position, and the unit vector pointing from the former to the latter. <br /><br />
Environment 2: <br />
The second environment is 'LunarLanderContinuous-v2' from OpenAI gym, in which a spaceship-like agent navigates to its landing pad by controlling its main central engine and its two side engines. The agent's observation of the environment's state is represented as a vector comprising the agent's position, velocity, orientation, angular speed, and two Booleans indicating if its two legs are in contact with the ground. <br />

## Algorithms
Two black-box techniques will be employed to solve the above two environments by means of direct policy optimization: <br /><br />
**Cross-Entropy Method (CEM):** <br />
Environment 1: <br />
![](https://github.com/rprasan/Reinforcement-Learning/blob/main/Blackbox%20Optimization%20Techniques/Environment%201/Seed%201/Average%20Reward%20Versus%20Number%20of%20Iterations.png) <br />
![](https://github.com/rprasan/Reinforcement-Learning/blob/main/Blackbox%20Optimization%20Techniques/Environment%201/Seed%201/Average%20Reward%20Versus%20Number%20of%20Steps.png) <br />
![](https://github.com/rprasan/Reinforcement-Learning/blob/main/Blackbox%20Optimization%20Techniques/Environment%201/Seed%201/Number%20of%20Steps%20Versus%20Number%20of%20Iterations.png) <br />
![](https://github.com/rprasan/Reinforcement-Learning/blob/main/Blackbox%20Optimization%20Techniques/Environment%201/Seed%201/Test%20Video.gif) <br /><br />

**Evolution Strategies [Salimas et al, 2017](https://arxiv.org/pdf/1703.03864.pdf):** <br />
Environment 2: <br />
![](https://github.com/rprasan/Reinforcement-Learning/blob/main/Blackbox%20Optimization%20Techniques/Environment%202/Seed%201/Average%20Reward%20Versus%20Number%20of%20Iterations.png) <br />
![](https://github.com/rprasan/Reinforcement-Learning/blob/main/Blackbox%20Optimization%20Techniques/Environment%202/Seed%201/Average%20Reward%20Versus%20Number%20of%20Steps.png) <br />
![](https://github.com/rprasan/Reinforcement-Learning/blob/main/Blackbox%20Optimization%20Techniques/Environment%202/Seed%201/Number%20of%20Steps%20Versus%20Number%20of%20Iterations.png) <br />
![](https://github.com/rprasan/Reinforcement-Learning/blob/main/Blackbox%20Optimization%20Techniques/Environment%202/Seed%201/Average%20Reward%20For%20Last%2050%20Evaluations%20Versus%20Number%20of%20Iterations.png) <br />
![](https://github.com/rprasan/Reinforcement-Learning/blob/main/Blackbox%20Optimization%20Techniques/Environment%202/Seed%201/Average%20Reward%20For%20Last%2050%20Evaluations%20Versus%20Number%20of%20Steps.png) <br />
![](https://github.com/rprasan/Reinforcement-Learning/blob/main/Blackbox%20Optimization%20Techniques/Environment%202/Average%20Reward%20Versus%20Number%20of%20Iterations.png) <br />
![](https://github.com/rprasan/Reinforcement-Learning/blob/main/Blackbox%20Optimization%20Techniques/Environment%202/Average%20Reward%20Versus%20Number%20of%20Steps.png) <br />
![](https://github.com/rprasan/Reinforcement-Learning/blob/main/Blackbox%20Optimization%20Techniques/Environment%202/Number%20of%20Steps%20Versus%20Number%20of%20Iterations.png) <br />
![](https://github.com/rprasan/Reinforcement-Learning/blob/main/Blackbox%20Optimization%20Techniques/Environment%202/Seed%201/Test%20Video.gif) <br /><br />
