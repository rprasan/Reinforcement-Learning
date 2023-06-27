import gym
import frogger_env
import pygame
import time
import numpy as np

# Models and computation
import torch # will use pyTorch to handle NN 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from random import sample

# Visualization
#import matplotlib
#import matplotlib.pyplot as plt


''' 
    Simple test to verify that you can import the environment.  
    The agent samples always the up action. 
    Change the code below if you want to randonly sample an action, 
    or manually control the agent with the keyboard (w = up, s = down, other_key = idle)!

'''
env = gym.make("reacher-v0")
mode = "up" #change this to manual or random
if mode == 'manual':
    env.config["manual_control"] = True


#The observation that the agent receives is its position, the goal position, 
#and a unit vector that points from the agent's position to the goal.
print('observation space:', env.observation_space)

#The action space consists of four actions 0: move up, 1: move down, 2: move right 3: move left
print('action space:', env.action_space)

for _ in range(5):
    done = False
    action = 1
    obs = env.reset()
    total_reward = 0
    while not done:
        obs_, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
        obs = obs_
        if env.config["manual_control"]:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                action = 0
            elif keys[pygame.K_s]:
                action = 1
            elif keys[pygame.K_d]:
                action = 2
            elif keys[pygame.K_a]:
                action = 3
        elif mode == 'up':
            action = 1
        elif mode == 'random':
            action = env.action_space.sample()
    print(total_reward)
env.close()

def save_checkpoint(model, filename):
    """
    Saves a model to a file 
        
    Parameters
    ----------
    model: your policy network
    filename: the name of the checkpoint file
    """
    torch.save(model.state_dict(), filename)



'''                         
    ############################### CEM ###############################  
    Implement the cross-entropy method to solve the reacher-v0 environment 
    and enable the agent to learn to navigate to a fixed goal position on 
    a static highway map. 
    Please, refer to the project descriprtion and lecture 13 for more details.
'''


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

######################## Your code ####################################

'''
    Implement a policy network that the agent can use to select actions.    
    Instead of training the parameters of the network, you will directly 
    set them based on your CEM optimization.
'''
class CEMPolicyNetwork(nn.Module): 
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        pass

    def forward(self, state):
        """
        Perform forward pass 
        """
        pass 
    
    def get_weights_dim(self):
        """
        Returns the number of the NN parameters (weights, biases)
        """
        return sum(p.numel() for p in self.parameters())
    
    def set_weights(self, weights):
        '''
        Function to copy a weight vector to the parameters (weights, biases) of the NN. 
        You can also set the weights outside of the class, e.g. using state_dict()

        Params
        ======
            weights (float): a flatten numpy array that contains the weight vector obtained through ES
        '''
        pass

'''
    Train a CEM agent
'''
class CEMAgent():
  pass