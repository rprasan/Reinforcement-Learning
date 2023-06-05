# crawler.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)
# Modified by Rahul Prasanna Kumar (rprasan@g.clemson.edu)
"""
In this file, you should test your Q-learning implementation on the robot crawler environment 
that we saw in class. It is suggested to test your code in the grid world environments before this one.

The package `matplotlib` is needed for the program to run.


The Crawler environment has discrete state and action spaces
and provides both of model-based and model-free access.

It has the following properties:
    env.observation_space.n     # the number of states
    env.action_space.n          # the number of actions
  

Once a terminal state is reached the environment should be (re)initialized by
    s = env.reset()
where s is the initial state.
An experience (sample) can be collected from s by taking an action a as follows:
    s_, r, terminal, info = env.step(a)
where s_ is the resulted state by taking the action a,
      r is the reward achieved by taking the action a,
      terminal is a boolean flag to indicate if s_ is a terminal state, and
      info is just used to keep compatible with openAI gym library.


A Logger instance is provided for each function, through which you can
visualize the process of the algorithm.
You can visualize the value, v, and policy, pi, for the i-th iteration by
    logger.log(i, v, pi)
"""


# use random library if needed
import random

def sarsa(env, logger):
    """
    Implement SARSA to return a deterministic policy for all states.

    Parameters
    ----------
    env: CrawlerEnv
        the environment
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process.
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """

    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    gamma = 0.95   

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 1
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    # maximum number of training iterations
    max_iterations = 5000
    #########################

### Please finish the code below ##############################################
###############################################################################
    import numpy as np
    import time as time
    tic=time.time()
    alpha=0.1                                                         #learning rate
    """LINEAR DECAY OF EPSILON"""
    #epsilon decays linearly from 1 to 0.1 within the first 80% iterations,
    #thereafter it remains equal to 0.1
    maxSteps=2000
    m,b=np.polyfit(np.array([0,np.ceil((0.8*maxSteps)-1)]),
                   np.array([1,0.1]),deg=1)                           #parameters for linear decay
    epsilon=[(m*i)+b for i in np.linspace(0,
             int(np.ceil((0.8*maxSteps)-1)),
             int(np.ceil((0.8*maxSteps)-1)))]                         #decaying part (80%) of epsilon
    epsilon=np.concatenate((epsilon,
                            0.1*np.ones((maxSteps-len(epsilon)))))    #concatenate with epislon for remaining 20%
    """SARSA (ON-POLICY TD)"""   
    Q=np.zeros((NUM_STATES,NUM_ACTIONS))                              #initialize all Q-states to zero
    numExperiences=0                                                  ##samples considered so far - not to exceed 'maxSteps'
    while numExperiences<maxSteps:                                    #'maxSteps' limits #experiences rather than #episodes
        state=env.reset()                                             #reset the game so that current state at the beginning of an episode is 'START'
        terminal=False                                                #status of current state - terminal or non-terminal    
        action=np.random.choice([[i for i in range(NUM_ACTIONS)],
                                  np.argmax(Q[state,:])],
        size=1,p=[epsilon[numExperiences],1-epsilon[numExperiences]]) #with probability (1-ε), choose action corresponding to max Q-value. With probability ε, choose list for random action
        action=np.stack(action).astype(None)                          #convert object to 1D or 2D array
        if action.ndim==1:                                            #if 1D, action is the one corresponding to max Q-value
            action=int(action)                                        #convert action to type 'int'
        else:                                                         #if 2D, list [i for i in range(NUM_ACTIONS)] is chosen
            action=int(np.random.choice(action.flatten(),size=1,
                                        p=[1/NUM_ACTIONS for i in range(NUM_ACTIONS)]))           #choose a random action from list [i for i in range(NUM_ACTIONS)] such that all actionss are equally likely
        while not terminal:                                           #loop over each experience until a terminal state is reached
            sPrime,reward,terminal,prob=env.step(action)              #generate an experience
            aPrime=np.random.choice([[i for i in range(NUM_ACTIONS)],
                                      np.argmax(Q[sPrime,:])],size=1,
                p=[epsilon[numExperiences],1-epsilon[numExperiences]])#with probability (1-ε), choose action corresponding to max Q-value. With probability ε, choose list for random action
            aPrime=np.stack(aPrime).astype(None)                      #convert object to 1D or 2D array
            if aPrime.ndim==1:                                        #if 1D, action is the one corresponding to max Q-value
                aPrime=int(aPrime)                                    #convert action to type 'int'
            else:                                                     #if 2D, list [i for i in range(NUM_ACTIONS)] is chosen
                aPrime=int(np.random.choice(aPrime.flatten(),size=1,
                                            p=[1/NUM_ACTIONS for i in range(NUM_ACTIONS)]))       #choose a random action from list [i for i in range(NUM_ACTIONS)] such that all elements are equally likely
            numExperiences+=1                                         #increment #collected experiences
            if terminal:                                              #if s' is terminal, ignore γQ(s',a')
                Q[state,action]+=(alpha*(reward-Q[state,action]))     #update value of current Q-state
            else:
                Q[state,action]+=(alpha*(reward+(gamma*Q[sPrime,aPrime])-Q[state,action]))        #update value of current Q-state
            pi=np.argmax(Q,axis=1).tolist()                           #update policy after processing each experience, and convert 'pi' to 'list'
            logger.log(numExperiences,np.max(Q,axis=1).tolist(),pi)   #for visualization
            state=sPrime                                              #next state becomes current state
            action=aPrime                                             #next action becomes current action
            if numExperiences>=maxSteps:                              #break if #experiences exceeds 'maxSteps'
                break
###############################################################################
    print('Execution time is '+str(round(time.time()-tic,2))+'s.\n')
    return pi



def q_learning(env, logger):
    """
    Implement Q-learning to return a deterministic policy for all states.

    Parameters
    ----------
    env: CrawlerEnv
        the environment
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process.
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """

    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    gamma = 0.95   

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 1
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    # maximum number of training iterations
    max_iterations = 5000
    #########################

### Please finish the code below ##############################################
###############################################################################
    import numpy as np
    import time as time
    tic=time.time()
    alpha=0.1                                                       #learning rate
    """LINEAR DECAY OF EPSILON"""
    #epsilon decays linearly from 1 to 0.1 within the first 80% iterations,
    #thereafter it remains equal to 0.1
    maxSteps=2000
    m,b=np.polyfit(np.array([0,np.ceil((0.8*maxSteps)-1)]),
                   np.array([1,0.1]),deg=1)                         #parameters for linear decay
    epsilon=[(m*i)+b for i in np.linspace(0,
             int(np.ceil((0.8*maxSteps)-1)),
             int(np.ceil((0.8*maxSteps)-1)))]                       #decaying part (80%) of epsilon
    epsilon=np.concatenate((epsilon,
                            0.1*np.ones((maxSteps-len(epsilon)))))  #concatenate with epislon for remaining 20%
    """Q-LEARNING (OFF-POLICY TD)"""   
    Q=np.zeros((NUM_STATES,NUM_ACTIONS))                            #initialize all Q-states to zero
    numExperiences=0                                                ##samples considered so far - not to exceed 'maxSteps'
    while numExperiences<maxSteps:                                  #'maxSteps' limits #experiences rather than #episodes
        state=env.reset()                                           #reset the game so that current state at the beginning of an episode is 'START'
        terminal=False                                              #status of current state - terminal or non-terminal
        while not terminal:                                         #loop over each experience until a terminal state is reached
            action=np.random.choice([[i for i in range(NUM_ACTIONS)],
                                      np.argmax(Q[state,:])],
        size=1,p=[epsilon[numExperiences],1-epsilon[numExperiences]])
            action=np.stack(action).astype(None)                    #convert object to 1D or 2D array
            if action.ndim==1:                                      #if 1D, action is the one corresponding to max Q-value
                action=int(action)                                  #convert action to type 'int'
            else:                                                   #if 2D, list [i for i in range(NUM_ACTIONS)] is chosen
                action=int(np.random.choice(action.flatten(),size=1,
                                            p=[1/NUM_ACTIONS for i in range(NUM_ACTIONS)]))#choose a random action from [0,1,2,3] such that all elements are equally likely
            sPrime,reward,terminal,prob=env.step(action)            #generate an experience
            numExperiences+=1                                       #increment #collected experiences
            if terminal:                                            #if s' is terminal, ignore γQ(s',a')
                Q[state,action]+=(alpha*(reward-Q[state,action]))   #update value of current Q-state
            else:
                Q[state,action]+=(alpha*(reward+(gamma*np.max(Q[sPrime,:]))-Q[state,action]))#update value of current Q-state
            pi=np.argmax(Q,axis=1).tolist()                         #update policy after processing each experience, and convert 'pi' to 'list'
            logger.log(numExperiences,np.max(Q,axis=1).tolist(),pi) #for visualization
            state=sPrime                                            #next state becomes current state
            if numExperiences>=maxSteps:                            #break if #experiences exceeds 'maxSteps'
                break
###############################################################################
    print('Execution time is '+str(round(time.time()-tic,2))+'s.\n')
    return pi


if __name__ == "__main__":
    from app.crawler import App
    import tkinter as tk
    
    algs = {
        "Q Learning": q_learning,
         "SARSA": sarsa
    }

    root = tk.Tk()
    App(algs, root)
    root.mainloop()