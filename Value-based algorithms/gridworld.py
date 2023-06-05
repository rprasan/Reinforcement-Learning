# grid_world.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)
# Modified by Rahul Prasanna Kumar (rprasan@g.clemson.edu)
"""
The package `matplotlib` is needed for the program to run.

The Grid World environment has discrete state and action spaces
and allows for both model-based and model-free access.

It has the following properties:
    env.observation_space.n     # the number of states
    env.action_space.n          # the number of actions
    env.trans_model             # the transition/dynamics model

In value_iteration and policy_iteration, you can access the transition model 
at a given state s and action by calling
    t = env.trans_model[s][a]
where s is an integer in the range [0, env.observation_space.n),
      a is an integer in the range [0, env.action_space.n), and
      t is a list of four-element tuples in the form of
        (p, s_, r, terminal)
where s_ is a new state reachable from the state s by taking the action a,
      p is the probability to reach s_ from s by a, i.e. p(s_|s, a),
      r is the reward of reaching s_ from s by a, and
      terminal is a boolean flag to indicate if s_ is a terminal state.

In mc_control, sarsa, q_learning, and double q-learning once a terminal state is reached, 
the environment should be (re)initialized by
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
You can also only update the visualization of the v values by
    logger.log(i, v)
"""


# use random library if needed
import random


def value_iteration(env, gamma, max_iterations, logger):
    """
    Implement value iteration to return a deterministic policy for all states.
    See lines 20-30 for details.  

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the reward discount factor
    max_iterations: integer
        the maximum number of value iterations that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        Hint: The value iteration may converge before reaching max_iterations.  
        In this case, you want to exit the algorithm earlier. A way to check 
        if value iteration has already converged is to check whether 
        the infinity norm between the values before and after an iteration is small enough. 
        In the gridworld environments, 1e-4 (theta parameter in the pseudocode) is an acceptable tolerance.
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    TRANSITION_MODEL = env.trans_model

    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    # Visualize the value and policy 
    logger.log(0, v, pi)
    # At each iteration, you may need to keep track of pi to perform logging
   
### Please finish the code below ##############################################
###############################################################################
    import numpy as np
    import time as time
    tic=time.time()
    VStar_km1=np.zeros((NUM_STATES))                         #stores V*(k-1) of all states
    VStar_k=np.zeros((NUM_STATES))                           #stores V*(k) of all states
    pi=np.zeros((NUM_STATES))                                #stores the policy (best action in each state)
    for k in range(max_iterations):                          #H='max_iterations'
        for state in range(NUM_STATES):                      #loop over all states
            for action in range(NUM_ACTIONS):                #loop over all actions
                transitions=TRANSITION_MODEL[state][action]  #T(s,a,s')
                if len(transitions)!=1:                      #if current state not terminal, #transitions>1. So use equation in notes to calculate 'temp'
                    temp=sum(tuple(i[0]*(i[2]+(gamma*VStar_km1[i[1]])) for i in transitions))#summation over all possible s' of current state s
                else:                                        #if current state is terminal, #transitions=1 as 'exit' is the only possible action. Ignore V*(k-1) as the episode terminates and 'r' incorporates the exit reward. If not ignored, V*(k-1) causes total reward to exceed maximum possible reward
                    temp=sum(tuple(i[0]*i[2] for i in transitions))                          #p x r since the episode terminates and V*(k-1) is ignored
                if action==0:                                #make V*(s)=sum from above statement and add corresponding action to policy 'pi'
                    VStar_k[state]=temp
                    pi[state]=action
                elif action!=0 and temp>VStar_k[state]:      #if sum higher for any other action, update V*(s) and policy 'pi'
                    VStar_k[state]=temp
                    pi[state]=action
        logger.log(k+1,VStar_k.tolist(),
                   [int(j) for j in pi.tolist()])            #for visualization
        time.sleep(.20)                                      #delay to improve viewability of GUI
        print('k='+str(k+1)+', \u221E-norm='+str(np.linalg.norm(VStar_k-VStar_km1,ord=np.inf)))
        if np.linalg.norm(VStar_k-VStar_km1,ord=np.inf)<1e-4:
            print('\nThe V-values have converged!! Convergence is at k='+str(k+1)+'.')
            break
        VStar_km1=VStar_k                                    #V*(k) becomes V*(k-1) for the next value of 'k'
        VStar_k=np.zeros((NUM_STATES))                       #reinitialize V*(k) to all zeros
    pi=[int(j) for j in pi.tolist()]                         #convert 'pi' to 'list' and its elements from 'float' to 'int'
###############################################################################
    print('The time taken for convergence is '+str(round(time.time()-tic,2))+'s.\n')
    print('The optimal policy \u03C0* is:')
    print(pi)
    return pi


def policy_iteration(env, gamma, max_iterations, logger):
    """
    Optional: Implement policy iteration to return a deterministic policy for all states.
    See lines 20-30 for details.  

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the reward discount factor
    max_iterations: integer
        the maximum number of policy iterations that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        Hint 1: Policy iteration may converge before reaching max_iterations. 
        In this case, you should exit the algorithm. A simple way to check 
        if the algorithm has already converged is by simply checking whether
        the policy at each state hasn't changed from the previous iteration.
        Hint 2: The value iteration during policy evaluation usually converges 
        very fast and policy evaluation should end upon convergence. A way to check 
        if policy evaluation has converged is to check whether the infinity norm 
        norm between the values before and after an iteration is small enough. 
        In the gridworld environments, 1e-4 is an acceptable tolerance.
    logger: app.grid_world.App.Logger
        a logger instance to record and visualize the iteration process.
        During policy evaluation, the V-values will be updated without changing the current policy; 
        here you can update the visualization of values by simply calling logger.log(i, v).
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    TRANSITION_MODEL = env.trans_model
    
    v = [0.0] * NUM_STATES
    pi = [random.randint(0, NUM_ACTIONS-1)] * NUM_STATES
    # Visualize the initial value and policy
    logger.log(0, v, pi)

### Please finish the code below ##############################################
###############################################################################
    import numpy as np
    import time as time
    tic=time.time()
    VPi_k=np.zeros((NUM_STATES))                           #stores VPi(k) of all states
    VPi_kp1=np.zeros((NUM_STATES))                         #stores VPi(k+1) of all states
    np.random.seed(1)
    pi_k=np.random.randint(0,NUM_ACTIONS,NUM_STATES)       #generate an initial random policy for Pi(k)
    pi_kp1=np.zeros((NUM_STATES))                          #stores the new policy Pi(k+1)
    logger.log(0,VPi_k.tolist(),[int(j) for j in pi_k.tolist()])
    time.sleep(0.20)                                       #delay to improve viewability of GUI
    for k in range(max_iterations):                        #H='max_iterations'
        """POLICY EVALUATION (IMPROVE VALUES)"""
        norm=1                                             #to check for value convergence
        while norm>1e-4:
            for state in range(NUM_STATES):                #loop over all states
                transitions=TRANSITION_MODEL[state][int(pi_k[state])]#T(s,pi(s),s') is the prescription
                if len(transitions)!=1:                    #if current state not terminal, #transitions>1. So use equation in notes to calculate 'temp'
                    VPi_kp1[state]=sum(tuple((i[0]*(i[2]+(gamma*VPi_k[i[1]]))) for i in transitions))#summation over all possible s' of current state s
                else:                                      #if current state is terminal, #transitions=1 as 'exit' is the only possible action. Ignore V*(k-1) as the episode terminates and 'r' incorporates the exit reward. If not ignored, V*(k-1) causes total reward to exceed maximum possible reward
                    VPi_kp1[state]=sum(tuple(i[0]*i[2] for i in transitions))                        #p x r since the episode terminates and V*(k-1) is ignored
            norm=np.linalg.norm(VPi_kp1-VPi_k,ord=np.inf)  #to check for value convergence
            VPi_k=VPi_kp1                                  #V*(k+1) becomes V*(k) for the next value of 'k'
            VPi_kp1=np.zeros((NUM_STATES))                 #reinitialize V*(k+1) to all zeros
        """POLICY IMPROVEMENT"""
        for state in range(NUM_STATES):                    #loop over all states
            for action in range(NUM_ACTIONS):              #loop over all actions
                transitions=TRANSITION_MODEL[state][action]#T('state','action',s')
                if len(transitions)!=1:                    #if current state not terminal, #transitions>1. So use equation in notes to calculate 'temp'
                    temp=sum(tuple((i[0]*(i[2]+(gamma*VPi_k[i[1]]))) for i in transitions))          #summation over all possible s' of current state s
                else:                                      #if current state is terminal, #transitions=1 as 'exit' is the only possible action. Ignore V*(k-1) as the episode terminates and 'r' incorporates the exit reward. If not ignored, V*(k-1) causes total reward to exceed maximum possible reward
                    temp=sum(tuple(i[0]*i[2] for i in transitions))                                  #p x r since the episode terminates and V*(k-1) is ignored
                if action==0:
                    pi_kp1[state]=action
                    value=temp
                elif action!=0 and temp>value:
                    pi_kp1[state]=action
                    value=temp
        logger.log(k+1,VPi_k.tolist(),
                   [int(j) for j in pi_kp1.tolist()])      #for visualization
        time.sleep(.20)                                    #delay to improve viewability of GUI
        if np.array_equal(pi_k,pi_kp1):                    #exit the loop if the policy has converged
            print('\nThe policy has converged!! Convergence is at k='+str(k+1)+'.')
            break
        pi_k=pi_kp1                                        #Pi(k+1) becomes Pi(k) for the next value of 'k'
        pi_kp1=np.zeros((NUM_STATES))                      #reinitialize Pi(k+1) to all zeros        
    pi=[int(j) for j in pi_k.tolist()]                     #convert 'pi' to 'list' and its elements from 'float' to 'int'
###############################################################################
    print('The time taken for convergence is '+str(round(time.time()-tic,2))+'s.\n')
    print('The optimal policy \u03C0* is:')
    print(pi)    
    return pi


def on_policy_mc_control(env, gamma, max_iterations, logger):
    """
    Implement on-policy first visit Monte Carlo control to return a deterministic policy for all states.
    Please note that in RL you do not have access to the transition model 
    and the reward function, i.e. you cannot call env.trans_model. 
    Instead you need to learn policies by collecting samples using env.step
    See lines 32-42 for more details. 

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the discount factor
    max_iterations: integer
        the maximum number of iterations (either training episodes or total steps) that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
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
    
    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    # Visualize the initial value and policy
    logger.log(0, v, pi)

    #########################
    # Adjust superparameters as you see fit
    #
    #parameter for the epsilon-greedy method to trade off exploration and exploitation
    epsilon = 1
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    #########################

    

### Please finish the code below ##############################################
###############################################################################
    """
    Ideal #iterations are as follows:
    world 1 - 3000
    world 2 - 3000
    world 3 - 5000
    """
    import numpy as np
    import time as time
    import matplotlib.pyplot as plt
    tic=time.time()
    alpha=0.1                                                       #learning rate
    """LINEAR DECAY OF EPSILON"""
    #epsilon decays linearly from 1 to 0.1 within the first 80% iterations,
    #thereafter it remains equal to 0.1
    m,b=np.polyfit(np.array([0,np.ceil((0.8*max_iterations)-1)]),
                   np.array([1,0.1]),deg=1)                         #parameters for linear decay
    epsilon=[(m*i)+b for i in np.linspace(0,
             int(np.ceil((0.8*max_iterations)-1)),
             int(np.ceil((0.8*max_iterations)-1)))]                 #decaying part (80%) of epsilon
    epsilon=np.concatenate((epsilon,
                            0.1*np.ones((max_iterations-len(epsilon)))))       #concatenate with epislon for remaining 20%
    plt.figure(1)                                                   #plot values of epsilon
    plt.grid('on')
    plt.ylabel('\u03B5',fontsize=14)
    plt.xlabel('#iterations',fontsize=14)
    plt.title('Linear decay of \u03B5',fontsize=18)
    plt.plot(np.linspace(0,len(epsilon)-1,len(epsilon)),epsilon)    
    """ON-POLICY FIRST VISIT MC"""
    Q=np.zeros((NUM_STATES,NUM_ACTIONS))                            #initialize all Q-states to zero
    numExperiences=0                                                ##samples considered so far - not to exceed 'max_iterations'
    while numExperiences<max_iterations:                            #'max_iterations' limits #experiences rather than #episodes
        episode=[]                                                  #to stores all experiences belonging to the current episode
        state=env.reset()                                           #reset the game so that current state at the beginning of an episode is 'START'
        terminal=False                                              #status of current state - terminal or non-terminal
        while not terminal:                                         #loop over each experience until a terminal state is reached
            action=np.random.choice([[i for i in range(NUM_ACTIONS)],
                                      np.argmax(Q[state,:])],
        size=1,p=[epsilon[numExperiences],
                  1-epsilon[numExperiences]])                       #with probability (1-ε), choose action corresponding to max Q-value. With probability ε, choose list for random action 
            action=np.stack(action).astype(None)                    #convert object to 1D or 2D array
            if action.ndim==1:                                      #if 1D, action is the one corresponding to max Q-value
                action=int(action)                                  #convert action to type 'int'
            else:                                                   #if 2D, list [i for i in range(NUM_ACTIONS)] is chosen
                action=int(np.random.choice(action.flatten(),
                                            size=1,p=[1/NUM_ACTIONS for i in range(NUM_ACTIONS)]))#choose a random action from list [i for i in range(NUM_ACTIONS)] such that all elements are equally likely
            sPrime,reward,terminal,prob=env.step(action)            #generating an experience
            episode.append((state,action,
                            reward,sPrime,terminal,prob))           #add generated experience to episode
            numExperiences+=1                                       #increment #collected experiences
            if numExperiences>=max_iterations:                      #break if 'max_terations' reached             
                break
            state=sPrime                                            #next state becomes current state
        visitedQStates=[]                                           #stores visited Q-states (helps satisfy first-visit property)
        for experience,index in zip(episode,range(len(episode))):   #loop over all experiences in episode
            if (experience[0],experience[1]) not in visitedQStates: #check if Q-state visited for the first time
                visitedQStates.append((experience[0],experience[1]))#add Q-state to visited Q-states
                Gt=experience[2]                                    #iteratively incrment 'Gt'
                n=1
                for i in range(index+1,len(episode)):
                    Gt+=((gamma**n)*episode[i][2])
                    n+=1
                Q[experience[0],experience[1]]+=(alpha*(Gt-Q[experience[0],experience[1]]))       #update value of current Q-state
        pi=np.argmax(Q,axis=1).tolist()                             #convert 'pi' to 'list' and its elements from 'float' to 'int'
        logger.log(numExperiences,np.max(Q,axis=1).tolist(),pi)     #for visualization
        time.sleep(.1)
###############################################################################
    print('Execution time is '+str(round(time.time()-tic,2))+'s.\n')
    print('The optimal policy \u03C0* is:')
    print(pi)
    return pi


def sarsa(env, gamma, max_iterations, logger):
    """
    Implement SARSA to return a deterministic policy for all states.
    Please note that in RL you do not have access to the transition model 
    and the reward function, i.e. you cannot call env.trans_model. 
    Instead you need to learn policies by collecting samples using env.step
    See lines 32-42 for more details. 

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the discount factor
    max_iterations: integer
        the maximum number of iterations (either training episodes or total steps) that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
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
    
    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    # Visualize the initial value and policy
    logger.log(0, v, pi)

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 1
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    #########################

    

### Please finish the code below ##############################################
###############################################################################
    """
    Ideal #iterations are as follows:
    world 1 - 3000
    world 2 - 3000
    world 3 - 10000
    """
    import numpy as np
    import time as time
    import matplotlib.pyplot as plt
    tic=time.time()
    alpha=0.1                                                         #learning rate
    """LINEAR DECAY OF EPSILON"""
    #epsilon decays linearly from 1 to 0.1 within the first 80% iterations,
    #thereafter it remains equal to 0.1
    m,b=np.polyfit(np.array([0,np.ceil((0.8*max_iterations)-1)]),
                   np.array([1,0.1]),deg=1)                           #parameters for linear decay
    epsilon=[(m*i)+b for i in np.linspace(0,
             int(np.ceil((0.8*max_iterations)-1)),
             int(np.ceil((0.8*max_iterations)-1)))]                   #decaying part (80%) of epsilon
    epsilon=np.concatenate((epsilon,
                            0.1*np.ones((max_iterations-len(epsilon)))))       #concatenate with epislon for remaining 20%
    plt.figure(1)                                                     #plot values of epsilon
    plt.grid('on')
    plt.ylabel('\u03B5',fontsize=14)
    plt.xlabel('#iterations',fontsize=14)
    plt.title('Linear decay of \u03B5',fontsize=18)
    plt.plot(np.linspace(0,len(epsilon)-1,len(epsilon)),epsilon)    
    """SARSA (ON-POLICY TD)"""   
    Q=np.zeros((NUM_STATES,NUM_ACTIONS))                              #initialize all Q-states to zero
    numExperiences=0                                                  ##samples considered so far - not to exceed 'max_iterations'
    while numExperiences<max_iterations:                              #'max_iterations' limits #experiences rather than #episodes
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
            if numExperiences>=max_iterations:                        #break if #experiences exceeds 'max_iterations'
                break
###############################################################################
    print('Execution time is '+str(round(time.time()-tic,2))+'s.\n')
    print('The optimal policy \u03C0* is:')
    print(pi)     
    return pi


def q_learning(env, gamma, max_iterations, logger):
    """
    Implement Q-learning to return a deterministic policy for all states.
    Please note that in RL you do not have access to the transition model 
    and the reward function, i.e. you cannot call env.trans_model. 
    Instead you need to learn policies by collecting samples using env.step
    See lines 32-42 for more details. 

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the discount factor
    max_iterations: integer
        the maximum number of iterations (either training episodes or total steps) that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
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
    
    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    # Visualize the initial value and policy
    logger.log(0, v, pi)

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 1
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    #########################

    

### Please finish the code below ##############################################
###############################################################################
    """
    Ideal #iterations are as follows:
    world 1 - 3000
    world 2 - 3000
    world 3 - 5000
    """
    import numpy as np
    import time as time
    import matplotlib.pyplot as plt
    tic=time.time()
    alpha=0.1                                                       #learning rate
    """LINEAR DECAY OF EPSILON"""
    #epsilon decays linearly from 1 to 0.1 within the first 80% iterations,
    #thereafter it remains equal to 0.1
    m,b=np.polyfit(np.array([0,np.ceil((0.8*max_iterations)-1)]),
                   np.array([1,0.1]),deg=1)                         #parameters for linear decay
    epsilon=[(m*i)+b for i in np.linspace(0,
             int(np.ceil((0.8*max_iterations)-1)),
             int(np.ceil((0.8*max_iterations)-1)))]                 #decaying part (80%) of epsilon
    epsilon=np.concatenate((epsilon,
                            0.1*np.ones((max_iterations-len(epsilon)))))       #concatenate with epislon for remaining 20%
    plt.figure(1)                                                   #plot values of epsilon
    plt.grid('on')
    plt.ylabel('\u03B5',fontsize=14)
    plt.xlabel('#iterations',fontsize=14)
    plt.title('Linear decay of \u03B5',fontsize=18)
    plt.plot(np.linspace(0,len(epsilon)-1,len(epsilon)),epsilon)    
    """Q-LEARNING (OFF-POLICY TD)"""   
    Q=np.zeros((NUM_STATES,NUM_ACTIONS))                            #initialize all Q-states to zero
    numExperiences=0                                                ##samples considered so far - not to exceed 'max_iterations'
    while numExperiences<max_iterations:                            #'max_iterations' limits #experiences rather than #episodes
        state=env.reset()                                           #reset the game so that current state at the beginning of an episode is 'START'
        terminal=False                                              #status of current state - terminal or non-terminal
        while not terminal:                                         #loop over each experience until a terminal state is reached
            action=np.random.choice([[i for i in range(NUM_ACTIONS)],np.argmax(Q[state,:])],size=1,p=[epsilon[numExperiences],1-epsilon[numExperiences]])
            action=np.stack(action).astype(None)                    #convert object to 1D or 2D array
            if action.ndim==1:                                      #if 1D, action is the one corresponding to max Q-value
                action=int(action)                                  #convert action to type 'int'
            else:                                                   #if 2D, list [i for i in range(NUM_ACTIONS)] is chosen
                action=int(np.random.choice(action.flatten(),size=1,p=[1/NUM_ACTIONS for i in range(NUM_ACTIONS)]))#choose a random action from [0,1,2,3] such that all elements are equally likely
            sPrime,reward,terminal,prob=env.step(action)            #generate an experience
            numExperiences+=1                                       #increment #collected experiences
            if terminal:                                            #if s' is terminal, ignore γQ(s',a')
                Q[state,action]+=(alpha*(reward-Q[state,action]))   #update value of current Q-state
            else:
                Q[state,action]+=(alpha*(reward+(gamma*np.max(Q[sPrime,:]))-Q[state,action]))#update value of current Q-state
            pi=np.argmax(Q,axis=1).tolist()                         #update policy after processing each experience, and convert 'pi' to 'list'
            logger.log(numExperiences,np.max(Q,axis=1).tolist(),pi) #for visualization
            state=sPrime                                            #next state becomes current state
            if numExperiences>=max_iterations:                      #break if #experiences exceeds 'max_iterations'
                break
###############################################################################
    print('Execution time is '+str(round(time.time()-tic,2))+'s.\n')
    print('The optimal policy \u03C0* is:')
    print(pi)         
    return pi


def double_q_learning(env, gamma, max_iterations, logger):
    """
    Implement double Q-learning to return a deterministic policy for all states.
    Please note that in RL you do not have access to the transition model 
    and the reward function, i.e. you cannot call env.trans_model. 
    Instead you need to learn policies by collecting samples using env.step
    See lines 32-42 for more details. 

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the discount factor
    max_iterations: integer
        the maximum number of iterations (either training episodes or total steps) that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
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
    
    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    # Visualize the initial value and policy
    logger.log(0, v, pi)

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 1
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    #########################

    

### Please finish the code below ##############################################
###############################################################################
    """
    Ideal #iterations are as follows:
    world 1 - 3000
    world 2 - 3000
    world 3 - 5000
    """
    import numpy as np
    import time as time
    import matplotlib.pyplot as plt
    tic=time.time()
    alpha=0.1                                                           #learning rate
    """LINEAR DECAY OF EPSILON"""
    #epsilon decays linearly from 1 to 0.1 within the first 80% iterations,
    #thereafter it remains equal to 0.1
    m,b=np.polyfit(np.array([0,np.ceil((0.8*max_iterations)-1)]),
                   np.array([1,0.1]),deg=1)                             #parameters for linear decay
    epsilon=[(m*i)+b for i in np.linspace(0,
             int(np.ceil((0.8*max_iterations)-1)),
             int(np.ceil((0.8*max_iterations)-1)))]                     #decaying part (80%) of epsilon
    epsilon=np.concatenate((epsilon,
                            0.1*np.ones((max_iterations-len(epsilon)))))#concatenate with epislon for remaining 20%
    plt.figure(1)                                                       #plot values of epsilon
    plt.grid('on')
    plt.ylabel('\u03B5',fontsize=14)
    plt.xlabel('#iterations',fontsize=14)
    plt.title('Linear decay of \u03B5',fontsize=18)
    plt.plot(np.linspace(0,len(epsilon)-1,len(epsilon)),epsilon)    
    """DOUBLE Q-LEARNING"""   
    QA=np.zeros((NUM_STATES,NUM_ACTIONS))                               #initialize all Q-states in QA to zero
    QB=np.zeros((NUM_STATES,NUM_ACTIONS))                               #initialize all Q-states in QB to zero
    numExperiences=0                                                    ##samples considered so far - not to exceed 'max_iterations'    
    while numExperiences<max_iterations:                                #'max_iterations' limits #experiences rather than #episodes
        state=env.reset()                                               #reset the game so that current state at the beginning of an episode is 'START'
        terminal=False                                                  #status of current state - terminal or non-terminal        
        while not terminal:                                             #loop over each experience until a terminal state is reached
            Q=(QA+QB)/2                                                 #define Q as the average of QA and QB
            action=np.random.choice([[i for i in range(NUM_ACTIONS)],np.argmax(Q[state,:])],size=1,p=[epsilon[numExperiences],1-epsilon[numExperiences]])
            action=np.stack(action).astype(None)                        #convert object to 1D or 2D array
            if action.ndim==1:                                          #if 1D, action is the one corresponding to max Q-value
                action=int(action)                                      #convert action to type 'int'
            else:                                                       #if 2D, list [i for i in range(NUM_ACTIONS)] is chosen
                action=int(np.random.choice(action.flatten(),size=1,p=[1/NUM_ACTIONS for i in range(NUM_ACTIONS)]))#choose a random action from [0,1,2,3] such that all elements are equally likely
            sPrime,reward,terminal,prob=env.step(action)                #generate an experience
            numExperiences+=1                                           #increment #collected experiences
            selection=np.random.choice(['A','B'],size=1,p=[0.5,0.5])    #choose QA or QB with equal probabilities
            if selection=='A':                                          #choice is QA
                aStar=np.argmax(QA[sPrime,:])                           #define a*
                if terminal:                                            #if s' is terminal, ignore γQB(s',a*)
                    QA[state,action]+=(alpha*(reward-QA[state,
                      action]))                                         #update value of current Q-state in QA
                else:                                               
                    QA[state,action]+=(alpha*(reward+(gamma*QB[sPrime,
                      aStar])-QA[state,action]))                        #update value of current Q-state in QA
            else:                                                       #choice is QB
                bStar=np.argmax(QB[sPrime,:])                           #define b*
                if terminal:                                            #if s' is terminal, ignore γQA(s',b*)
                    QB[state,action]+=(alpha*(reward-QB[state,action])) #update value of current Q-state in QB
                else:
                    QB[state,action]+=(alpha*(reward+(gamma*QA[sPrime,bStar])-QB[state,action]))#update value of current Q-state in QB
            pi=np.argmax((QA+QB)/2,axis=1).tolist()                     #update policy by making it greedy wrt mean of QA and QB
            logger.log(numExperiences,np.max((QA+QB)/2,axis=1).tolist(),pi)    #for visualization
            state=sPrime                                                #next state becomes current state
            if numExperiences>=max_iterations:                          #break if #experiences exceeds 'max_iterations'
                break
###############################################################################
    print('Execution time is '+str(round(time.time()-tic,2))+'s.\n')
    print('The optimal policy \u03C0* is:')
    print(pi)         
    return pi


if __name__ == "__main__":
    from app.grid_world import App
    import tkinter as tk

    algs = {
        "Value Iteration": value_iteration,
        "Policy Iteration": policy_iteration,
        "On-policy MC Control": on_policy_mc_control,
        "SARSA": sarsa,
        "Q-Learning": q_learning,
        "Double Q-Learning": double_q_learning
   }
    worlds = {
        # o for obstacle
        # s for start cell
        "world1": App.DEFAULT_WORLD,
        "world2": lambda : [
            ["_", "_", "_", "_", "_"],
            ["s", "_", "_", "_", 1],
            [-100, -100, -100, -100, -100],
        ],
        #"world2": lambda : [
        #    [10, "s", "s", "s", 1],
        #    [-10, -10, -10, -10, -10],
        #],
        "world3": lambda : [
            ["_", "_", "_", "_", "_"],
            ["_", "o", "_", "_", "_"],
            ["_", "o",   1, "_",  10],
            ["s", "_", "_", "_", "_"],
            [-10, -10, -10, -10, -10]
        ]
    }

    root = tk.Tk()
    App(algs, worlds, root)
    tk.mainloop()