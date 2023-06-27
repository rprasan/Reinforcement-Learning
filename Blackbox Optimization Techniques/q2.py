import gym
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
    Simple test to verify that you can import the LunarLander environment.  
    The agent samples a random action

'''
env = gym.make("LunarLanderContinuous-v2")
for _ in range(5):
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
env.close()

#The observation that the agent receives is its position, velocity, 
#angular speed, and two 2 Boolean flags indicating whether the left 
#and right leg of the agent, respectively, is in contact with the ground.
print('observation space:', env.observation_space)

#Two continuous actions are used to control the agent. The first one controls 
#the main engine and the second one controls the left and right engines. 
#The expected range of each of the two actions is [-1, 1]. 
#Please make sure that your policy network outputs values in that range
print('action space:', env.action_space)


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
    ############################### ES ###############################  
    Implement below the non-distributed version of the evolution 
    strategies method from  [Salimans et al., 2017] to solve the 
    LunarLanderContinuous-v2 environment and enable the agent to 
    navigate to its landing pad. 
    Please, refer to the project descriprtion and lecture 13 for more details. 
'''


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

######################## Your code ####################################

'''
    Implement a policy network that the agent can use to select actions.    
    Instead of training the parameters of the network, you will directly 
    set them based on your CEM optimization.
'''
class ESPolicyNetwork(nn.Module): 
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
    Train an ES agent
'''
class ESAgent():
  pass